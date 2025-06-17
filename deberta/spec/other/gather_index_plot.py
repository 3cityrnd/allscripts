import torch


# code from transformers library

def c2p_index_deberta_v1(query_layer, key_layer, max_relative_positions, rel_embeddings):
    @torch.jit.script
    def compute_attention_span(query_layer: torch.Tensor, key_layer: torch.Tensor, max_relative_positions: int):
        return torch.tensor(min(max(query_layer.size(-2), key_layer.size(-2)), max_relative_positions))

    @torch.jit.script
    def build_relative_position(query_layer, key_layer):
        query_size = query_layer.size(-2)
        key_size = key_layer.size(-2)

        q_ids = torch.arange(query_size, dtype=torch.long, device=query_layer.device)
        k_ids = torch.arange(key_size, dtype=torch.long, device=key_layer.device)
        rel_pos_ids = q_ids[:, None] - k_ids.view(1, -1).repeat(query_size, 1)
        rel_pos_ids = rel_pos_ids[:query_size, :]
        rel_pos_ids = rel_pos_ids.unsqueeze(0)
        return rel_pos_ids

    @torch.jit.script
    def c2p_dynamic_expand(c2p_pos: torch.Tensor, query_layer: torch.Tensor, relative_pos: torch.Tensor):
        return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])

    relative_pos = build_relative_position(query_layer, key_layer,
                                            # query_layer.device
                                           )
    if relative_pos.dim() == 2:
        relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
    elif relative_pos.dim() == 3:
        relative_pos = relative_pos.unsqueeze(1)
    att_span = compute_attention_span(query_layer, key_layer, max_relative_positions)
    relative_pos = relative_pos.long()
    rel_embeddings = rel_embeddings[
        max_relative_positions - att_span : max_relative_positions + att_span, :
    ].unsqueeze(0)

    c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
    index=c2p_dynamic_expand(c2p_pos, query_layer, relative_pos)
    return index


def c2p_index_deberta_v2v3(query_layer, key_layer, max_relative_positions, rel_embeddings, position_buckets):
    @torch.jit.script
    def make_log_bucket_position(relative_pos, bucket_size: int, max_position: int):
        sign = torch.sign(relative_pos)
        mid = bucket_size // 2
        abs_pos = torch.where(
            (relative_pos < mid) & (relative_pos > -mid),
            torch.tensor(mid - 1).type_as(relative_pos),
            torch.abs(relative_pos),
        )
        log_pos = (
            torch.ceil(torch.log(abs_pos / mid) / torch.log(torch.tensor((max_position - 1) / mid)) * (mid - 1)) + mid
        )
        bucket_pos = torch.where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)
        return bucket_pos

    def build_relative_position(query_layer, key_layer, bucket_size: int = -1, max_position: int = -1):
        query_size = query_layer.size(-2)
        key_size = key_layer.size(-2)

        q_ids = torch.arange(query_size, dtype=torch.long, device=query_layer.device)
        k_ids = torch.arange(key_size, dtype=torch.long, device=key_layer.device)
        rel_pos_ids = q_ids[:, None] - k_ids[None, :]
        if bucket_size > 0 and max_position > 0:
            rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
        rel_pos_ids = rel_pos_ids.to(torch.long)
        rel_pos_ids = rel_pos_ids[:query_size, :]
        rel_pos_ids = rel_pos_ids.unsqueeze(0)
        return rel_pos_ids
    relative_pos = build_relative_position(
    query_layer,
    key_layer,
    bucket_size=position_buckets,
    max_position=max_relative_positions,
    )
    if relative_pos.dim() == 2:
        relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
    elif relative_pos.dim() == 3:
        relative_pos = relative_pos.unsqueeze(1)
    relative_pos = relative_pos.long()
    att_span = position_buckets
    c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
    # index = c2p_pos.squeeze(0).expand([query_layer.size(0), query_layer.size(1), relative_pos.size(-1)])
    index = c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])
    return index


def plot_img(name, data):
    font_size = 5
    plt.matshow(data.numpy(), cmap="Blues")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = f"{data[i, j].int().item()}"
            x, y = j, i
            for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), 
                        (0,1), (1,-1), (1,0), (1,1)]:
                plt.text(x + dx*0.015, y + dy*0.015, value,
                        ha='center', va='center',
                        color='black', fontsize=font_size, weight='bold')
            plt.text(x, y, value, 
                    ha='center', va='center',
                    color='white', fontsize=font_size, weight='bold')
    plt.savefig(f'{name}.png', dpi=300)
    plt.close()


if __name__ == "__main__":

    DEBUG_SIZE = 64

    BATCH = 3
    SEQ_LEN = 111
    HIDDEN_DIM = DEBUG_SIZE // 2
    EMBED_SIZE = DEBUG_SIZE
    QKV_DIM = 64
    SCALE = 1.0

    q = torch.rand(BATCH, SEQ_LEN, HIDDEN_DIM, QKV_DIM)
    k = torch.rand(BATCH, SEQ_LEN, HIDDEN_DIM, QKV_DIM)
    v = torch.rand(BATCH, SEQ_LEN, HIDDEN_DIM, QKV_DIM)

    c2p = torch.rand(SEQ_LEN, QKV_DIM, EMBED_SIZE)
    p2c = torch.rand(SEQ_LEN, QKV_DIM, EMBED_SIZE)
    pos = torch.randint(low=0, high=HIDDEN_DIM, size=[BATCH, SEQ_LEN, HIDDEN_DIM, HIDDEN_DIM])

    rel_embeddings = torch.rand(HIDDEN_DIM, HIDDEN_DIM) # max_position_embeddings, hidden_size
    position_buckets = torch.tensor(HIDDEN_DIM)

    index_v1 = c2p_index_deberta_v1(q, k, HIDDEN_DIM, rel_embeddings)
    index_v2v3 = c2p_index_deberta_v2v3(q, k, HIDDEN_DIM, rel_embeddings, position_buckets)

    # assert torch.allclose(index_v1, index_v2v3) == True, "not equal"

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # import time

    # for i in range(BATCH):
    #     for j in range(SEQ_LEN):
    i, j = 0, 0
    diff = index_v1[i,j,...] - index_v2v3[i,j,...]
    plot_img('index_v1', index_v1[i,j,...])
    plot_img('index_v2v3', index_v2v3[i,j,...])
    plot_img('diff', diff)

