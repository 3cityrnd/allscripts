import torch
import numpy as np
import random


def SET_SEED(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

SET_SEED(42)


def noopt_partial_disentangled_attention(q_states: torch.Tensor, 
                           k_states: torch.Tensor, 
                           v_states: torch.Tensor,
                           c2p_pos_embed: torch.Tensor, 
                           p2c_pos_embed: torch.Tensor,
                           pos_index: torch.Tensor, 
                           attn_mask: torch.Tensor,
                           score_scale: torch.Tensor
                           ):
    # c2p_pos_embed = [sequence_length, 64, 512]
    # q_states = [batch_size, sequence_length, 256, 64]

    c2p_pos_embed = c2p_pos_embed.unsqueeze(0)
    # c2p_pos_embed = [1, sequence_length, 64, 512]

    matmul_1 = torch.matmul(q_states, c2p_pos_embed)
    # matmul_1 = [batch_size, sequence_length, 256, 512]

    gather_c2p = torch.gather(matmul_1, dim=-1, index=pos_index)
    # gather_c2p = [batch_size, sequence_length, 256, 256]


    # p2c_pos_embed = [sequence_length, 64, 512]
    # k_states = [batch_size, sequence_length, 256, 64]

    p2c_pos_embed = p2c_pos_embed.unsqueeze(0)
    # p2c_pos_embed = [1, sequence_length, 64, 512]

    matmul_2 = torch.matmul(k_states, p2c_pos_embed)
    # matmul_2 = [batch_size, sequence_length, 256, 512]

    gather_p2c = torch.gather(matmul_2, dim=-1, index=pos_index)
    # gather_p2c = [batch_size, sequence_length, 256, 256]

    result = torch.add(gather_c2p, gather_p2c)
    # result = [batch_size, sequence_length, 256, 256]

    return result


def opt_partial_disentangled_attention(q_states, 
                           k_states, 
                           v_states,
                           c2p_pos_embed, 
                           p2c_pos_embed,
                           pos_index, 
                           attn_mask,
                           score_scale
                           ):
    # c2p_pos_embed = [sequence_length, 64, 512]
    # q_states = [batch_size, sequence_length, 256, 64]

    c2p_pos_embed = c2p_pos_embed.unsqueeze(0)
    # c2p_pos_embed = [1, sequence_length, 64, 512]

    matmul_1 = torch.matmul(q_states, c2p_pos_embed)
    # matmul_1 = [batch_size, sequence_length, 256, 512]


    # p2c_pos_embed = [sequence_length, 64, 512]
    # k_states = [batch_size, sequence_length, 256, 64]

    p2c_pos_embed = p2c_pos_embed.unsqueeze(0)
    # p2c_pos_embed = [1, sequence_length, 64, 512]

    matmul_2 = torch.matmul(k_states, p2c_pos_embed)
    # matmul_2 = [batch_size, sequence_length, 256, 512]

    add = torch.add(matmul_1, matmul_2)
    # add = [batch_size, sequence_length, 256, 512]

    gather = torch.gather(add, dim=-1, index=pos_index)
    # gather = [batch_size, sequence_length, 256, 256]

    return gather



if __name__ == "__main__":
    BATCH = 3
    SEQ_LEN = 111
    HIDDEN_DIM = 256
    EMBED_SIZE = 512
    QKV_DIM = 64
    SCALE = 1.0

    q = torch.rand(BATCH, SEQ_LEN, HIDDEN_DIM, QKV_DIM)
    k = torch.rand(BATCH, SEQ_LEN, HIDDEN_DIM, QKV_DIM)
    v = torch.rand(BATCH, SEQ_LEN, HIDDEN_DIM, QKV_DIM)

    c2p = torch.rand(SEQ_LEN, QKV_DIM, EMBED_SIZE)
    p2c = torch.rand(SEQ_LEN, QKV_DIM, EMBED_SIZE)
    pos = torch.randint(low=0, high=HIDDEN_DIM, size=[BATCH, SEQ_LEN, HIDDEN_DIM, HIDDEN_DIM])
    mask = torch.rand(BATCH, SEQ_LEN, HIDDEN_DIM, HIDDEN_DIM)

    device = torch.device('npu:0')

    q = q.to(device)
    k = k.to(device)
    v = v.to(device)
    c2p = c2p.to(device)
    p2c = p2c.to(device)
    pos = pos.to(device)
    mask = mask.to(device)

    with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=True) as profiler:
        noopt_attn_outputs = noopt_partial_disentangled_attention(q_states=q, 
                                                        k_states=k, 
                                                        v_states=v,
                                                        c2p_pos_embed=c2p,
                                                        p2c_pos_embed=p2c,
                                                        pos_index=pos,
                                                        attn_mask=mask,
                                                        score_scale=SCALE
                                                        )
    print(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=True) as profiler:

        opt_attn_outputs = opt_partial_disentangled_attention(q_states=q, 
                                                    k_states=k, 
                                                    v_states=v,
                                                    c2p_pos_embed=c2p,
                                                    p2c_pos_embed=p2c,
                                                    pos_index=pos,
                                                    attn_mask=mask,
                                                    score_scale=SCALE
                                                    )
    print(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))    
    assert torch.allclose(noopt_attn_outputs, opt_attn_outputs) == True
