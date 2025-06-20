import torch
import torch_npu


dev_string="npu"

a=torch.rand(2,4,device=dev_string)
b=torch.randn(2,2).npu()

print(f'[ok] npu is working : tensor has been created on {dev_string}')

