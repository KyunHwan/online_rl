import ray

@ray.remote
class PolicyStateManagerActor:
    """
    Manages the weights of the components of a policy via reference handle to the shared memory, 
    in which the weights live. 
    Think of this class as a data memory pointer manager.
    """
    def __init__(self):
        # dictionary of policy component weights
        self.current_weights_ref = None

        # weight version tracking
        self.controller_version = 0
        self.trainer_version = 0

    def update_weights(self, new_weights_ref):
        self.current_weights_ref = new_weights_ref
        self.trainer_version += 1
        print("weight pushed to Plasma...")

    def get_weights(self):
        """
        Returns reference to updated weight if weight has been updated.
        Returns None else
        """
        if self.current_weights_ref is not None and self.controller_version != self.trainer_version:
            print("received weight from plasma...")
            self.controller_version = self.trainer_version
            return self.current_weights_ref
        else:
            return None