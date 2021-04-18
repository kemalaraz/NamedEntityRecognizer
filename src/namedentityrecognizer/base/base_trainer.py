
class Trainer:
    """Base class for trainers"""

    def train(self, *args, **kwargs):
        """Train loop"""
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        """Evaluation loop"""
        raise NotImplementedError

    def infer(self, *args, **kwargs):
        """Inference"""
        raise NotImplementedError