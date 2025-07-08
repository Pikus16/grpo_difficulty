from transformers import TrainerCallback, TrainerState, TrainerControl
import wandb

class CumulativeSuccessCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self._cumulative = 0

    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # `state.log_history[-1]` is the most recent logged metrics dict
        latest_state = state.log_history[-1]
        if 'reward' in latest_state:
            num_successes = int(latest_state['reward'] * args.num_generations)
            self._cumulative += num_successes
            # push to W&B
            wandb.log({"train/cumulative_successes": self._cumulative})
        return control