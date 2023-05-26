import torch
import torch.nn.functional as F

def gumbel_softmax(logits, temperature=1.0):
    gumbel_noise = torch.empty_like(logits).exponential_().log()  # Generate Gumbel noise
    gumbel_logits = (logits + gumbel_noise) / temperature  # Add Gumbel noise and apply temperature
    gumbel_softmax = F.softmax(gumbel_logits, dim=-1)  # Apply softmax to obtain Gumbel-Softmax
    return gumbel_softmax
    
def differentiable_softmax_k_pick_mask(logits, top_k, dim = -1):
    softmax = F.softmax(logits, dim=dim)  # Apply softmax

    # Differentiable approximation of binary representation for picking k-out-of-n
    values, indices = torch.topk(softmax, top_k, dim=dim)
    mask = torch.zeros_like(logits)
    mask.scatter_(dim, indices, values / values)

    return mask

def choose_mask(g_prompt, e_prompt, g_prompt_top_k=3, e_prompt_top_k=3, max_noise=0.0):
    prompts = torch.stack([g_prompt, e_prompt], dim=0)

    noise = torch.rand_like(prompts) * max_noise
    prompts = prompts + noise
    
    prompts_softmax = F.softmax(prompts, dim=0)
    g_prompt = prompts_softmax[0]
    e_prompt = prompts_softmax[1]

    g_prompt = differentiable_softmax_k_pick_mask(g_prompt, g_prompt_top_k, dim=-1)
    e_prompt = differentiable_softmax_k_pick_mask(e_prompt, e_prompt_top_k, dim=-1)

    return g_prompt, e_prompt


def test_collsion(n=1000, device = "cpu"):
    collision_count = 0
    for _ in range(n):
        mask1 = torch.randn(12, requires_grad=True).to(device)
        mask2 = torch.randn(12, requires_grad=True).to(device)
        
        g_mask, e_mask = choose_mask(mask1, mask2)
        if torch.sum(g_mask*e_mask) > 0:
            collision_count += 1
    print(collision_count/n)


if __name__ == "__main__":
    device = "cpu"

    mask1 = torch.randn(12, requires_grad=True).to(device)
    mask2 = torch.randn(12, requires_grad=True).to(device)
    
    g_mask, e_mask = choose_mask(mask1, mask2, max_noise=1.0)
    assert g_mask.grad_fn is not None
    assert e_mask.grad_fn is not None
    

    test_collsion(10, device=device)

    # smulate batch
    mask1 = mask1.expand(3, -1)
    mask2 = mask2.expand(3, -1)

    g_mask, e_mask = choose_mask(mask1, mask2, max_noise=0.3)

    print(g_mask)
    print(e_mask)