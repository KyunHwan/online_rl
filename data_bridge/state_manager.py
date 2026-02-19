import ray

@ray.remote
class StateManagerActor:
    """
    Manages the abstract state via reference handle to the shared memory.
    Think of this class as a data memory pointer manager.
    """
    def __init__(self):
        # dictionary of policy component weights
        self.current_state_ref = None

        # weight version tracking
        self.controller_version = 0
        self.trainer_version = 0

    def update_state(self, new_state_ref):
        self.current_state_ref = new_state_ref
        self.trainer_version += 1
        print("weight pushed to Plasma...")

    def get_state(self):
        """
        Returns reference to updated weight if weight has been updated.
        Returns None else
        """
        if self.current_state_ref is not None and self.controller_version != self.trainer_version:
            print("received weight from plasma...")
            self.controller_version = self.trainer_version
            return self.current_state_ref
        else:
            return None