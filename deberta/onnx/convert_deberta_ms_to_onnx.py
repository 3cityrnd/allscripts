from transformers import AutoTokenizer, AutoModel
import torch

model_name = "microsoft/deberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

text = "Example text DeBERTa gather"
inputs = tokenizer(text, return_tensors="pt")

input_names = ["input_ids", "attention_mask"]
output_names = ["last_hidden_state"]

torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    "deberta.onnx",
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"}
    },
    opset_version=13,
    do_constant_folding=True
)

print(" Model saved as deberta.onnx")

