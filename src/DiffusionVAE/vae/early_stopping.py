import numpy as np

class LossEarlyStopping:
    def __init__(self, patience: int, min_delta: float, smoothing_window: int, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.smoothing_window = smoothing_window
        self.verbose = verbose
        self.best_loss = None
        self.early_stop = False
        self.loss_history = []
        self.wait = 0
        self.min_delta = min_delta

    def get_smoothed_loss(self, losses):
        if len(losses) < self.smoothing_window:
            return np.mean(losses)
        else:
            return np.mean(losses[-self.smoothing_window:])

    def __call__(self, current_loss: float):
        self.loss_history.append(current_loss)
        smoothed_loss = self.get_smoothed_loss(self.loss_history)

        if self.best_loss is None:
            self.best_loss = smoothed_loss
            if self.verbose:
                print(f"Early stopping baseline set: {smoothed_loss:.4f}")
        elif smoothed_loss > self.best_loss - self.min_delta:
            self.wait += 1
            if self.verbose:
                print(f"No improvement: {smoothed_loss:.4f} vs {self.best_loss:.4f}, patience: {self.wait}/{self.patience}")
            if self.wait >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered! Best loss: {self.best_loss:.4f}")
        else:
            # Loss improvement detect
            self.best_loss = smoothed_loss
            self.wait = 0
            if self.verbose:
                print(f"Loss improved to {smoothed_loss:.4f}")
                print(100*"=")

        return self.early_stop