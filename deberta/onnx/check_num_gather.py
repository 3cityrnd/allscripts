import onnx

onnx_model_path = "deberta.onnx"

model = onnx.load(onnx_model_path)

gather_nodes = [node for node in model.graph.node if node.op_type == "Gather"]

print(f"'Gather' in model: {len(gather_nodes)}")

# Print nodes
for i, node in enumerate(gather_nodes, 1):
    print(f"{i}. Node number {node.name if node.name else 'no name'}")

