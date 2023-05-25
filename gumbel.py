import torch
import torch.nn.functional as F

def gumbel_softmax(logits, temperature, top_k=1, normalize = True):
    softmax_sample = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
    if top_k > 1:
        _, indices = softmax_sample.topk(top_k, dim=-1)
        one_hot_sample = torch.zeros_like(softmax_sample).scatter_(-1, indices, 1.0)
        softmax_sample = one_hot_sample / top_k  # Normalize to sum to 1 over the top k values
    if normalize:
        return softmax_sample
    else:
        return softmax_sample * top_k
    
def gumbel_topk_binary_mask(mask, top_k, temperature):
    # Apply Gumbel-Softmax reparameterization
    gumbel_softmax_sample = gumbel_softmax(mask, temperature, top_k, normalize=False)
    return gumbel_softmax_sample.bool()