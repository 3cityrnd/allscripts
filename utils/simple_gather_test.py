import torch
import torch_npu



a=torch.rand(3,111,256,512, device='npu')
b=torch.randint(low=1, high=250, size=[3,111,256,256],device='npu')
for i in range(1,10):
     print(f'Executed {i} {a.shape} {b.shape}')
     c= torch.gather(a, dim=-1, index=b)





