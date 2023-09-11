# import the required packages
import wandb

# define the WandbProfiler class
# watch gradients and model parameters to W&B
# logs metrics and losses to W&B
class WandbProfiler:
    def __init__(self, **config):
        self.config = config
        self.profiler = wandb

        self.profiler.login()
        
        self.run = self.profiler.init(
            project = self.config['project'],
            name = self.config['name'],
        )

    def watch(self, model, criterion):
        self.profiler.watch(model, criterion, log = 'all', log_freq = self.config['log_freq'])

    def update(self, history, epoch):
        self.profiler.log(history, step = epoch)

    def end(self):
        self.profiler.finish()