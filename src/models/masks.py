import torch

def create_causal_mask(seq_len, device=None):
    mask = torch.triu(
        torch.ones(seq_len,seq_len,dtype=torch.bool,device=device),
        diagonal=1
    )

    return mask.unsqueeze(0).unsqueeze(0)