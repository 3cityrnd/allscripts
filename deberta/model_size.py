import torch
from transformers import AutoTokenizer, AutoModel

model_name = "microsoft/deberta-v3-base"

# Load model 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# move to GPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = model.to(device)
device = torch.device('cuda')
model=model.to(device)
model.eval()

# Sync VRAM
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

# forward pass, 
dummy_input = tokenizer("Example task", return_tensors="pt").to(device)
with torch.no_grad():
    _ = model(**dummy_input)

# Sync and read mem
torch.cuda.synchronize()
memory_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)  # MB

print(f"Model size {memory_allocated:.2f} MB VRAM")


