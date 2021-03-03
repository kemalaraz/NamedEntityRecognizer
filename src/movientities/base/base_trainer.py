
class Trainer:
    """Base Class for Training"""

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError