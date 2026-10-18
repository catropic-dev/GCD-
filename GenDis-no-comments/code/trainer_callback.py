from transformers import TrainerCallback

class DatasetUpdateCallback(TrainerCallback):

    def __init__(self, trainer, unlabeled_dataset):
        self.trainer = trainer
        self.unlabeled_dataset = unlabeled_dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        pass

    def _generate_pseudo_labels(self, model):
        pass

    def _update_training_set(self, pseudo_labels):
        pass
