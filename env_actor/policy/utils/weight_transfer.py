import torch

def load_state_dict_cpu_into_module(module: torch.nn.Module, sd_cpu: dict, strict: bool = True):
    """
    sd_cpu: state_dict with CPU tensors (from Ray shared memory / object store).
    Moves each tensor to the matching device+dtype of the target module's current state_dict.
    """
    # If wrapped (DDP, etc.)
    target = module.module if hasattr(module, "module") else module

    tgt_sd = target.state_dict()  # tensors on correct device/dtype
    sd = {}
    for k, v in sd_cpu.items():
        if isinstance(v, torch.Tensor) and k in tgt_sd:
            sd[k] = v.to(
                device=tgt_sd[k].device,
                dtype=tgt_sd[k].dtype,
                non_blocking=True,
            )
        else:
            sd[k] = v

    return target.load_state_dict(sd, strict=strict)