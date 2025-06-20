import torch

dev_string="npu"

a=torch.rand(2,4,device=dev_string)

print(f'Tensor has been created on {dev_string}')

