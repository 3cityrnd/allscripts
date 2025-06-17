import math
import torch
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def relative_position_v1(hidden_dim, max_relative_dist: Optional[int] = None): 
    # resulting matrix is hidden_dim x hidden_dim, max_relative_dist is k
    # assert max_relative_dist <= hidden_dim, "max_relative_dist needs to be smaller than hidden dim"
    position_matrix = torch.zeros(hidden_dim, hidden_dim)
    k = max_relative_dist if max_relative_dist else hidden_dim//2
    # k = torch.as_tensor(k)
    for i, row in enumerate(position_matrix):
        for j, value in enumerate(row):
            if i-j <= -k:
                position_matrix[i, j] = 0
            elif i-j >= k:
                position_matrix[i, j] = 2*k-1
            else:
                position_matrix[i, j] = i-j+k
    return position_matrix


def relative_position_v2v3(hidden_dim, max_relative_dist: Optional[int] = None):
    # assert max_relative_dist <= hidden_dim, "max_relative_dist needs to be smaller than hidden dim"
    position_matrix = torch.zeros(hidden_dim, hidden_dim)
    k = max_relative_dist if max_relative_dist else hidden_dim//2
    # k = torch.as_tensor(k)
    mid = k // 2

    def log_func(pos, hidden_dim):
        abs_pos = torch.abs(torch.as_tensor(pos))
        # if abs_pos < mid:
        #     abs_pos = torch.as_tensor(mid - 1)
        # numerator=torch.log(torch.as_tensor(abs_pos/mid))
        # denumerator=torch.log(torch.as_tensor((hidden_dim-1)/mid))
        numerator=math.log10(abs_pos/mid)
        denumerator=math.log10((hidden_dim-1)/mid)
        result = (numerator/denumerator)*(mid-1)+mid
        return torch.round(torch.as_tensor(result))#.int()

    # rel_pos_matrix = relative_position_v1(hidden_dim, k)

    for i, row in enumerate(position_matrix):
        for j, value in enumerate(row):
            # rel_pos = rel_pos_matrix[i, j]
            rel_pos = i-j
            if i-j <= -mid:
                position_matrix[i, j] = -log_func(rel_pos, hidden_dim)+k
            elif i-j >= mid:
                position_matrix[i, j] = log_func(rel_pos, hidden_dim)+k
            else:
                position_matrix[i, j] = i-j+k
    return position_matrix


if __name__ == "__main__":
    BATCH = 3
    SEQ_LEN = 111
    HIDDEN_DIM = 8
    EMBED_SIZE = 512
    QKV_DIM = 64
    SCALE = 1.0

    q = torch.rand(BATCH, SEQ_LEN, HIDDEN_DIM, QKV_DIM)
    k = torch.rand(BATCH, SEQ_LEN, HIDDEN_DIM, QKV_DIM)
    v = torch.rand(BATCH, SEQ_LEN, HIDDEN_DIM, QKV_DIM)

    c2p = torch.rand(SEQ_LEN, QKV_DIM, EMBED_SIZE)
    p2c = torch.rand(SEQ_LEN, QKV_DIM, EMBED_SIZE)
    pos = torch.randint(low=0, high=HIDDEN_DIM, size=[BATCH, SEQ_LEN, HIDDEN_DIM, HIDDEN_DIM])

    max_rel_dist = 6
    pos = relative_position_v1(hidden_dim=HIDDEN_DIM, max_relative_dist=max_rel_dist)
    pos = relative_position_v2v3(hidden_dim=HIDDEN_DIM, max_relative_dist=max_rel_dist)

    plt.matshow(pos.numpy(), cmap='Blues')
    for i in range(pos.shape[0]):
        for j in range(pos.shape[1]):
            value = f"{pos[i, j].int().item()}"
            x, y = j, i
            for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), 
                        (0,1), (1,-1), (1,0), (1,1)]:
                plt.text(x + dx*0.015, y + dy*0.015, value,
                        ha='center', va='center',
                        color='black', fontsize=10, weight='bold')
            plt.text(x, y, value, 
                    ha='center', va='center',
                    color='white', fontsize=10, weight='bold')
    plt.savefig(f'embeddings_sample.png')