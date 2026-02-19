import tensordict
from tensordict import TensorDict
import numpy as np
import torch
import copy

class EpisodeRecorderBridge:
    def __init__(self):
        self.episodic_obs_state = []
        self.episodic_action = []

    def _split_by_control_mode(
        self,
        train_td: TensorDict,
        key: str = "control_mode",
        clone: bool = False,
        return_mode: bool = True,
    ):
        """
        Splits `train_td` into a list of (mode, sub_td) or just sub_td, where each sub_td
        corresponds to a consecutive run of identical values of train_td[key].

        Args:
            train_td: TensorDict with leading time/batch dim (T, ...)
            key: key that contains the per-step control mode
            clone: if True, returns independent copies; if False, slices/views may share storage
            return_mode: if True returns list[(mode, td)], else list[td]
        """
        cm = train_td.get(key)

        # Make sure cm is 1D [T]
        cm = cm.squeeze()
        if cm.ndim != 1:
            cm = cm.reshape(-1)

        T = cm.numel()
        if T == 0:
            return []

        # modes: unique consecutive values, counts: run lengths
        modes, counts = torch.unique_consecutive(cm, return_counts=True)
        ends = counts.cumsum(0)          # end indices (exclusive)
        starts = ends - counts           # start indices

        out = []
        for mode, s, e in zip(modes.tolist(), starts.tolist(), ends.tolist()):
            sub = train_td[s:e]
            if clone:
                sub = sub.clone()
            out.append((mode, sub) if return_mode else sub)

        return out
    
    def _split_by_control_mode_as_episodes(
        self,
        train_td: TensorDict,
        base_episode_id: int,
        key: str = "control_mode",
    ):
        chunks = self._split_by_control_mode(train_td, key=key, clone=True, return_mode=True)

        out = []
        for seg_idx, (mode, sub) in enumerate(chunks):
            # Make unique episode ids per segment (choose any scheme you like)
            new_episode_id = base_episode_id * 10_000 + seg_idx
            sub["episode"].fill_(new_episode_id)

            # Mark end of each segment as terminal for sampling purposes
            sub[("next", "done")].zero_()
            sub[("next", "done")][-1] = True

            # (optional) keep mode around explicitly if you want
            # sub["segment_control_mode"] = torch.full(sub.batch_size, mode, dtype=sub[key].dtype)

            out.append(sub)

        return out

    def serve_train_data_buffer(self, episode_id):
        """
        Returns a tensordict for SliceSampler replay buffer
        """
        train_data = [TensorDict({
            "episode": torch.tensor(episode_id, dtype=torch.int64),
            "reward": torch.zeros(1).squeeze(),
            ("next", "done"): torch.zeros(1, dtype=torch.bool)
        }) for i in range(len(self.episodic_obs_state))]
        train_data = torch.stack(train_data, dim=0)
        train_data[("next", "done")][-1] = True

        next_obs = copy.deepcopy(self.episodic_obs_state[1:])
        next_obs.append(next_obs[-1])
        next_obs = TensorDict({
            "next": torch.stack(next_obs, dim=0)
        })

        obs = torch.stack(self.episodic_obs_state, dim=0)
        actions = torch.stack(self.episodic_action, dim=0)

        train_data.update(obs)
        train_data.update(actions)
        train_data.update(next_obs)

        return self._split_by_control_mode_as_episodes(train_td=train_data, base_episode_id=episode_id)
    
    def add_obs_state(self, obs_data: dict[str, np.array]):
        """
        Fields of obs_data are:
            proprio
            head
            left
            right
        where head, left, right represent images.
        """
        obs_data_tensordict = TensorDict({
                data_name: torch.from_numpy(obs_data[data_name]) for data_name in obs_data.keys()
            }, batch_size=[])
        
        self.episodic_obs_state.append(obs_data_tensordict)
    
    def add_action(self, action: np.array, control_mode: int = 0):
        td = TensorDict({
            'action': torch.from_numpy(action),
            'control_mode': torch.tensor(control_mode, dtype=torch.int8),
        }, batch_size=[])
        self.episodic_action.append(td)
    
    def init_train_data_buffer(self):
        self.episodic_obs_state = []
        self.episodic_action = []
