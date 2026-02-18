"""Adapter: runs vla_'s pi05_igris policy inside inference_engine.

Uses raw (unnormalized) observations from data_manager.get_raw_obs_arrays().
Returns raw (denormalized) actions as numpy array.
vla_'s Policy.infer() handles all normalization internally.
"""

import os
import sys
import pathlib
import numpy as np

# --- Single sys.path hook for vendored openpi ---
def _ensure_openpi_importable():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    third_party = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..",
            "trainer", "policy_constructor", "model_constructor",
            "blocks", "experiments", "third_party",
        )
    )
    if third_party not in sys.path:
        sys.path.insert(0, third_party)

_ensure_openpi_importable()

from openpi.policies import policy_config as _policy_config
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _train_config

# inference_engine camera key -> vla_ camera key
_CAM_MAP = {"head": "cam_head", "left": "cam_left", "right": "cam_right"}


def get_action_dim_from_config(train_config):
    """Extract action_dim from output data transforms, fallback to model config."""
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    for transform in data_config.data_transforms.outputs:
        if hasattr(transform, "action_dim"):
            return int(transform.action_dim)
    return int(train_config.model.action_dim)


def get_norm_stats_dim(norm_stats, key):
    """Get dimensionality of a norm_stats entry from its mean shape."""
    if not norm_stats:
        return None
    stats = norm_stats.get(key)
    if stats is None:
        return None
    return int(np.asarray(stats.mean, dtype=np.float32).reshape(-1).shape[0])


class Pi05IgrisVlaAdapter:
    """Wraps vla_'s pi05_igris for inference_engine's sequential loop.

    Expects RAW observations (from get_raw_obs_arrays). Returns RAW actions (denormalized).
    All normalization is handled internally by vla_'s Policy.infer().

    Args:
        ckpt_dir: Path to vla_ checkpoint (must contain model.safetensors + assets/)
        device: torch device string (e.g. "cuda", "cpu")
        train_config_name: openpi config name (default "pi05_igris")
        default_prompt: Optional language instruction
        camera_names: List of IE camera keys (default ["head", "left", "right"])
    """

    def __init__(
        self,
        ckpt_dir: str | pathlib.Path,
        device: str = "cuda",
        train_config_name: str = "pi05_igris",
        default_prompt: str | None = None,
        camera_names: tuple[str, ...] = ("head", "left", "right"),
        norm_stats=None,
    ):
        self.device = device
        self.camera_names = list(camera_names)

        # --- Load train config ---
        train_config = _train_config.get_config(train_config_name)
        ckpt_dir = pathlib.Path(ckpt_dir)

        # --- Robust norm_stats discovery (ported from pi_rtc.py) ---
        if norm_stats is None:
            data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
            asset_id = data_config.asset_id or data_config.repo_id
            if asset_id is None:
                raise ValueError("Checkpoint asset_id is missing; cannot infer state/action dimensions.")

            assets_dir = ckpt_dir / "assets"
            norm_stats_dir = None

            if assets_dir.exists():
                expected_dir = assets_dir / asset_id
                if (expected_dir / "norm_stats.json").exists():
                    norm_stats_dir = expected_dir
                else:
                    candidates = list(assets_dir.rglob("norm_stats.json"))
                    matching = [p for p in candidates if asset_id in str(p.parent)]
                    if matching:
                        candidates = matching
                    if len(candidates) == 1:
                        norm_stats_dir = candidates[0].parent
                    elif len(candidates) > 1:
                        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        norm_stats_dir = candidates[0].parent
                        print(f"[Pi05Igris] Warning: multiple norm_stats found; using {norm_stats_dir}")

            if norm_stats_dir is None:
                raise ValueError(f"norm_stats not found under {assets_dir} (asset_id={asset_id})")

            norm_stats_asset = str(norm_stats_dir.relative_to(assets_dir))
            norm_stats = _checkpoints.load_norm_stats(assets_dir, norm_stats_asset)
            print(f"[Pi05Igris] Loaded norm_stats from {norm_stats_dir}")

        # --- Extract state_dim from norm_stats ---
        stats_state_key = None
        state_dim = None
        if norm_stats:
            if "state" in norm_stats:
                stats_state_key = "state"
            else:
                state_candidates = [k for k in norm_stats if k != "actions"]
                stats_state_key = state_candidates[0] if state_candidates else None
            state_dim = get_norm_stats_dim(norm_stats, stats_state_key) if stats_state_key else None

        if state_dim is None:
            raise ValueError("Checkpoint norm_stats missing state statistics; cannot infer state_dim.")

        # --- Extract and validate action_dim ---
        action_dim_from_config = get_action_dim_from_config(train_config)
        action_dim_from_stats = get_norm_stats_dim(norm_stats, "actions") if norm_stats else None

        if action_dim_from_stats is not None and action_dim_from_config != action_dim_from_stats:
            raise ValueError(
                f"Checkpoint action_dim ({action_dim_from_stats}) does not match "
                f"config action_dim ({action_dim_from_config})."
            )
        action_dim = action_dim_from_stats or action_dim_from_config
        if action_dim < 24:
            raise ValueError(f"Expected action_dim >= 24 for Igris, got {action_dim}")

        # --- Extract action_horizon ---
        action_horizon = train_config.model.action_horizon

        # --- Expose extracted values ---
        self.norm_stats = norm_stats
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.stats_state_key = stats_state_key

        print(f"[Pi05Igris] state_dim={state_dim}, action_dim={action_dim}, "
              f"action_horizon={action_horizon}, state_key={stats_state_key}")

        # --- Create policy with resolved norm_stats ---
        self._vla_policy = _policy_config.create_trained_policy(
            train_config,
            str(ckpt_dir),
            default_prompt=default_prompt,
            pytorch_device=str(device),
            norm_stats=norm_stats,
        )

    def predict(self, obs: dict, noise: np.ndarray | None = None) -> np.ndarray:
        """Run pi05_igris inference on raw observations.

        Args:
            raw_obs: Dict from data_manager.get_raw_obs_arrays():
                'proprio': np.float32 (num_robot_obs, state_dim) -- raw, unnormalized
                'head': np.uint8 (num_img_obs, 3, H, W)
                'left': np.uint8 (num_img_obs, 3, H, W)
                'right': np.uint8 (num_img_obs, 3, H, W)

        Returns:
            np.float32 (action_horizon, action_dim) -- denormalized actions (e.g., 50x24)
        """
        # Latest state (index 0 = most recent in data_manager history)
        raw_state = obs["proprio"].squeeze().astype(np.float32)

        # Convert images: (3,H,W) uint8 CHW -> (H,W,3) uint8 HWC
        images = {}
        for ie_name in self.camera_names:
            img_chw = obs[ie_name].squeeze()               # (3,H,W) uint8
            img_hwc = np.transpose(img_chw, (1, 2, 0))  # (H,W,3) uint8
            images[_CAM_MAP.get(ie_name, ie_name)] = img_hwc

        # vla_ Policy.infer handles: IgrisInputs -> resize -> tokenize -> normalize
        #   -> model -> unnormalize -> IgrisOutputs(clip to 24 dims)
        result = self._vla_policy.infer(obs={"images": images, "state": raw_state},
                                        noise=noise)

        return result["actions"]  # np.float32 (50, 24) denormalized

    # --- Stubs for inference_engine policy interface ---
    def warmup(self) -> None:
        """Run a dummy forward pass to trigger torch.compile and cuDNN warmup."""
        dummy_obs = {
            "proprio": np.zeros((1, self.state_dim), dtype=np.float32),
        }
        for cam in self.camera_names:
            dummy_obs[cam] = np.zeros((1, 3, 224, 224), dtype=np.uint8)
        self.predict(dummy_obs)

    def freeze_all_model_params(self) -> None:
        """vla_ policy is already in eval mode after create_trained_policy."""
        pass

    def guided_inference(self, input_data):
        raise NotImplementedError("pi05_igris adapter does not support RTC guided inference")
