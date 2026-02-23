# env_actor/inference_engine_utils

Utility functions for policy inference, primarily **action inpainting** — the technique used to blend overlapping action chunks for smooth real-time control.

## Background

When a policy predicts actions in chunks (e.g., 40 timesteps at once) but GPU inference takes several control steps to complete, there is a gap between when actions were predicted and when they are executed. Action inpainting blends the previously predicted (but not yet executed) actions with the newly predicted chunk using decay weights, ensuring smooth transitions.

Reference: [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/pdf/2506.07339)

## Key Functions

### `compute_guided_prefix_weights()`

**File:** [`action_inpainting.py`](action_inpainting.py)

Numpy-based function that computes blending weights for two overlapping action chunks. Used by policies that handle inpainting externally (outside the model architecture).

```python
weights = compute_guided_prefix_weights(
    delay_steps=5,     # estimated inference latency in control steps
    executed=35,       # number of actions already consumed from old chunk
    total=40,          # total action chunk size
    schedule="exp",    # exponential decay schedule
)
# weights.shape: (40,)
# weights[:delay_steps] ≈ 1.0  (keep old actions — already committed)
# weights[delay_steps:total-executed] = exponential decay
# weights[total-executed:] = 0.0  (use new predictions)
```

The policy applies the weights as:
```python
blended = prev_actions * weights + new_actions * (1.0 - weights)
```

### `guided_action_chunk_inference()`

**File:** [`action_inpainting.py`](action_inpainting.py)

Torch-based function for models whose architecture supports inpainting directly during the denoising process. Uses flow-matching ODE integration with VJP-based guidance to steer the denoised action chunk toward the previously predicted one.

This is a more advanced approach where the guidance happens during the ODE solve (each denoising step), rather than as a post-hoc blend. It requires the model to expose an `action_decoder` module with a `(time, noise, memory_input, discrete_semantic_input)` interface.

**Parameters:**
- `action_decoder` — the denoising model
- `cond_memory` — conditioning features (e.g., from vision encoder)
- `prev_action_chunk` — the previous action chunk to guide toward
- `delay`, `executed_steps` — for computing prefix weights
- `num_ode_sim_steps` — number of Euler integration steps
- `max_guidance_weight` — clamp for the VJP correction strength

## Files

| File | Purpose |
|------|---------|
| [`action_inpainting.py`](action_inpainting.py) | `compute_guided_prefix_weights()` (numpy) and `guided_action_chunk_inference()` (torch) |
