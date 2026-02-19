import math
from typing import Optional

import torch
from torch.func import vjp


def guided_action_chunk_inference(
    action_decoder: torch.nn.Module,
    cond_memory: torch.Tensor,
    discrete_semantic_input: Optional[torch.Tensor],
    prev_action_chunk: torch.Tensor,
    delay: int,
    executed_steps: int,
    num_ode_sim_steps: int,
    num_queries: int,
    action_dim: int,
    max_guidance_weight: float = 5.0,
    input_noise = None
) -> torch.Tensor:
    """
    Stateless real-time action chunking routine for flow-matching policies.
    """
    device = cond_memory.device
    batch_size = cond_memory.shape[0]
    original_dtype = prev_action_chunk.dtype if prev_action_chunk is not None else cond_memory.dtype

    if prev_action_chunk is None:
        prev_action_chunk = torch.zeros(batch_size, num_queries, action_dim, device=device, dtype=original_dtype)
    else:
        prev_action_chunk = prev_action_chunk[:, :num_queries]

    if num_ode_sim_steps <= 0:
        return prev_action_chunk.to(dtype=original_dtype)

    dt = 1.0 / float(num_ode_sim_steps)

    def compute_prefix_weights(delay_steps: int, executed: int, total: int, schedule: str = "exp") -> torch.Tensor:
        start = max(min(int(delay_steps), total), 0)
        span = max(int(executed), 1)
        end = max(total, start + span)
        if end < start:
            start = end
        indices = torch.arange(end, device=device, dtype=torch.float32)
        weights = torch.zeros(end, device=device, dtype=torch.float32)
        weights[:start] = 1.0
        if schedule == "ones":
            weights = torch.ones_like(indices)
        elif schedule == "zeros":
            weights = (indices < start).float()
        else:
            denom = end - span - start + 1
            c_i = (end - span - indices) / denom
            inter_vals = c_i * torch.expm1(c_i) / (math.e - 1.0)
            weights[start : end - span] = inter_vals[start : end - span]
            weights[end - span :] = 0.0
        return weights

    weights = compute_prefix_weights(delay, executed_steps, num_queries, schedule="exp").view(1, num_queries, 1)

    max_guidance_tensor = torch.tensor(max_guidance_weight, device=device, dtype=torch.float32)
    
    actions_hat = torch.randn(
        batch_size,
        num_queries,
        action_dim,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    ) if input_noise is None else input_noise
    
    with torch.enable_grad():
        for step in range(num_ode_sim_steps):
            tau = step / float(num_ode_sim_steps)
            tau_tensor = torch.full((batch_size,), tau, device=device, dtype=torch.float32)

            def denoiser(actions_in: torch.Tensor):
                velocity = action_decoder(time=tau_tensor, 
                                          noise=actions_in, 
                                          memory_input=cond_memory,
                                          discrete_semantic_input=discrete_semantic_input,)
                x1 = actions_in + (1.0 - tau) * velocity
                return x1, velocity

            (estimated_final, velocity), vjp_fn = vjp(denoiser, actions_hat)
            error = (prev_action_chunk - estimated_final) * weights
            pinv_correction = vjp_fn((error, torch.zeros_like(velocity)))[0]

            tau_scalar = torch.tensor(tau, device=device, dtype=torch.float32)
            inv_r2 = (tau_scalar.pow(2) + (1 - tau_scalar).pow(2)) / torch.clamp((1 - tau_scalar).pow(2), min=1e-6)
            base = torch.where(tau_scalar > 0, (1 - tau_scalar) / tau_scalar, max_guidance_tensor)
            guidance_weight = torch.minimum(base * inv_r2, max_guidance_tensor)

            velocity = velocity + guidance_weight * pinv_correction
            actions_hat = (actions_hat + dt * velocity).detach()
            actions_hat.requires_grad_(True)

    return actions_hat.detach().to(device=device, dtype=original_dtype)
