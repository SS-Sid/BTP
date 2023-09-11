class EarlyStopper:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.best_epoch = None
        self.counter = 0
    
    def check_stop(self, history):
        loss = history['val']['loss'][-1]
        if self.best_loss is None:
            self.best_loss = loss
            self.best_epoch = len(history['val']['loss'])
        elif loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = loss
            self.best_epoch = len(history['val']['loss'])
            self.counter = 0
        return False