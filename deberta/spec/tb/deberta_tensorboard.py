'''

	"attention_probs_dropout_prob": 0.1,
	"hidden_act": "gelu",
	"hidden_dropout_prob": 0.1,
	"hidden_size": 768,
	"initializer_range": 0.02,
	"intermediate_size": 3072,
	"max_position_embeddings": 512,
	"relative_attention": true,
	"position_buckets": 256,
	"norm_rel_ebd": "layer_norm",
	"share_att_key": true,
	"pos_att_type": "p2c|c2p",
	"layer_norm_eps": 1e-7,
	"max_relative_positions": -1,
	"position_biased_input": false,
	"num_attention_heads": 12,
	"attention_head_size": 64,
	"num_hidden_layers": 12,
	"type_vocab_size": 0,
	"vocab_size": 128100
}

'''

import os
import requests
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from huggingface_hub import configure_http_backend
import argparse
import numpy as np
import random
import torch_npu

orch_org_gather = torch.gather

def our_custom_gather(*args, **kwargs):
  print(f'custom gather() shape: input={args[0].shape} {args[0].device}')
  return orch_org_gather(*args, **kwargs)

# === Optional: fix for SSL certificate errors when downloading from Hugging Face ===
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False  # Allow self-signed certs
    return session

# === Check if a device is available (CUDA, NPU, or CPU) ===
def device_exists(device_str):
    if device_str.startswith('cuda'):
        return torch.cuda.is_available() and int(device_str.split(':')[1]) < torch.cuda.device_count()
    elif device_str.startswith('npu'):
        try:
            import torch_npu
            return torch_npu.npu.is_available()
        except ImportError:
            return False
    elif device_str == 'cpu':
        return True
    return False

# === Get the first available device from priority list ===
def getDevice():
    for dev in ['cuda:0', 'npu:0']:
        if device_exists(dev):
            return dev
    return 'cpu'

# === Paths for local model and dataset ===
MODEL_NAME = "microsoft/deberta-v3-base"
MODEL_DIR = "./local_model"
DATASET_DIR = "./imdb_dataset"

# === Download and cache model if not already saved ===
if not os.path.isdir(MODEL_DIR):
    print("## Downloading model...")
    configure_http_backend(backend_factory=backend_factory)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)

# === Download and save IMDb dataset if not already available ===
if not os.path.isdir(DATASET_DIR):
    print("## Downloading dataset...")
    configure_http_backend(backend_factory=backend_factory)
    dataset = load_dataset("imdb")
    dataset.save_to_disk(DATASET_DIR)

# === Define a simple classifier model using DeBERTa as encoder ===
class DebertaSentimentClassifier(nn.Module):
    def __init__(self, model_dir, num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_dir)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits



def SET_SEED(seed: int, cuda : None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
       torch.cuda.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


parser = argparse.ArgumentParser(description="Run DeBERTa inference with configurable backend and label")
parser.add_argument("--compile", type=str, default="none", help="Torch compile backend (e.g. inductor, eager, aot_eager)")
parser.add_argument("--samples", type=int, default=64, help="Samples default 64")
parser.add_argument("--batch_size", type=int, default=8, help="Batch Size 8")
parser.add_argument("--profile", action="store_true", help="Enable performance profiling")
parser.add_argument("--dev", type=str, default="auto", help="Torch device")
parser.add_argument("--custom_gather", action="store_true", help="Replace custom op gather")
parser.add_argument("--seed", type=int, default=20, help="set seed")

args = parser.parse_args()

for key, value in vars(args).items():
    print(f"## Default setting for : {key}: {value}")


if args.custom_gather:
  torch.gather = our_custom_gather

# === Setup: device, tokenizer, model ===

dev_str = getDevice() if args.dev == "auto" else args.dev

device = torch.device( dev_str )

if args.seed:
   cdn=True if 'cuda' in dev_str else False 
   SET_SEED(args.seed, cdn)


print(f"## Execution device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = DebertaSentimentClassifier(MODEL_DIR).to(device).eval()

# === Optional: compile the model for better performance (requires PyTorch 2.0+) ===
if args.compile != 'none':
 print(f'Use torch.compile({args.compile})')
 model = torch.compile(model, backend=args.compile)

# === Load 1250 examples from the test split ===
dataset = load_from_disk(DATASET_DIR)["test"].select(range(args.samples))
batch_size = args.batch_size
predictions = []

# === Warm-up / tracing for the compiled model ===
example_texts = dataset[:batch_size]["text"]
enc = tokenizer(example_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
enc.pop("token_type_ids", None)
enc = {k: v.to(device) for k, v in enc.items()}
model(enc["input_ids"], enc["attention_mask"])  # Trace the model

# === Inference loop over dataset ===

def core_inference(prof=None):
   
   for i in tqdm(range(0, len(dataset), batch_size), desc="Running inference"):
       batch_texts = dataset[i:i + batch_size]["text"]
       inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
       inputs.pop("token_type_ids", None)  # Some models don't use this
       inputs = {k: v.to(device) for k, v in inputs.items()}
   
       with torch.no_grad():
           logits = model(inputs["input_ids"], inputs["attention_mask"])
           probs = torch.softmax(logits, dim=-1)
           preds = torch.argmax(probs, dim=-1).cpu().tolist()
           predictions.extend(preds)

       if prof:
           prof.step()    
   
   # === Print a few example predictions ===
   for i in range(5):
       print(f"\nText: {dataset[i]['text'][:100]}...")
       print(f"Predicted Sentiment: {'Positive' if predictions[i] else 'Negative'}")
   

experimental_config = torch_npu.profiler._ExperimentalConfig(
	aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
	profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
	l2_cache=False
)


if args.profile:
    with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(
                wait=1,
                warmup=1,
                active=2),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./tensorboard_result"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
      core_inference(prof)
      

    print(f"\n### Top operators by {device.type} time:")
#    print(prof.key_averages().table(
          #sort_by=f"{device.type}_time_total", 
          #row_limit=20
#           ))


else:
   core_inference()

