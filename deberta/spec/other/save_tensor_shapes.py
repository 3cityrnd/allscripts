import os
import random
import torch
import json


def save_info(*args, **kwargs):
    current_folder = os.getcwd()
    data_folder = os.path.join(current_folder, 'sample_data')
    os.makedirs(data_folder, exist_ok=True)

    sample_folder_name = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(12))
    sample_folder_path =os.path.join(data_folder, sample_folder_name)
    os.makedirs(sample_folder_path)

    # att_tensor = args[0].cpu().detach()
    index_tensor: torch.Tensor = kwargs['index']
    index_tensor = index_tensor.cpu().detach().int()
    # torch.save(att_tensor, os.path.join(sample_folder_path, 'input_tensor.pt'))
    torch.save(index_tensor, os.path.join(sample_folder_path, 'index_tensor.pt'))

    shape_data = {
        "att_shape": list(args[0].shape), 
        "index_shape": list(index_tensor.shape) 
    }
    with open(os.path.join(sample_folder_path, 'shapes.json'), "w") as f:
        json.dump(shape_data, f)    

    import pandas as pd
    df = pd.DataFrame(index_tensor)
    df.to_csv("index_tensor.csv", float_format="%.2f")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# To use it, change the our_custom_gather func in deberta_v3_base.py like so:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# def our_custom_gather(*args, **kwargs):
#     save_info(*args, **kwargs)
#     return torch_org_gather(*args, **kwargs)
