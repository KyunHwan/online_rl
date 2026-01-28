import ray
import torch
import transformers
from collections import deque

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # small win sometimes

@ray.remote(num_gpus=1)
class AutoRewardLabelerActor:
    def __init__(self, episode_queue_handle, replay_buffer_actor, img_frame_key: str, reward_key: str):
        self.episode_queue_handle = episode_queue_handle
        self.replay_buffer_actor = replay_buffer_actor
        self.img_frame_key = img_frame_key
        self.reward_key = reward_key

        # Load your heavy VLM here
        self.model_name = "nvidia/Cosmos-Reason2-8B"
        self.model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="flash_attention_2",  # if available, "flash_attention_2" is often faster
        ).eval()

        self.model.generation_config.use_cache = True

        processor = transformers.AutoProcessor.from_pretrained(self.model_name)
        processor.image_processor.min_pixels = 224 * 224
        processor.image_processor.max_pixels = 224 * 224

    def start(self):
        while True:
            # Wait until there's data
            # Resolve the object (Zero-copy read from Plasma) internally
            episode_data = self.episode_queue_handle.get(block=True)
            
            # Label Rewards & Send labeled data
            self.process_episode(episode_data)

    def process_episode(self, episode_data):
        # Label rewards
        # Assume we modify the tensordict or create a new one with rewards
        # TODO: 
        #     episode_data["reward"] = reward
        
        # Push to Replay Buffer
        # We call .remote() and wait for the acknowledgment to ensure 
        # the disk write is queued before we declare victory.
        replay_buffer_write_ref = self.replay_buffer_actor.add.remote(episode_data)
        
        # Wait for the disk write to register so we know flow is moving
        data_added = ray.get(replay_buffer_write_ref)
