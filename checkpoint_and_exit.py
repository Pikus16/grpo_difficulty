#!/usr/bin/env python3
# checkpoint_and_exit.py
# ----------------------
# When called (by spot_watcher.sh), this script will import the global Trainer
# from train.py, force a checkpoint, and then exit. Your training process
# should notice that it’s been killed shortly afterwards and come up again
# with a new instance.

import os
import sys
import torch

# Add the directory containing train.py to the Python path (adjust if needed)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from train import _global_trainer  # <-- This is the Trainer instance from train.py

def force_checkpoint_and_exit():
    if _global_trainer is None:
        print("[checkpoint_and_exit.py] ERROR: No trainer is active in memory.")
        return

    try:
        # Ask TRL’s Trainer to save a checkpoint (similar to what it does every `save_steps`).
        # By default, TRL uses `trainer.save_model()` → this will produce a new folder
        # like `runs/checkpoint-<current_step>`.
        ckpt_dir = _global_trainer.args.output_dir
        print(f"[checkpoint_and_exit.py] Forcing checkpoint to {ckpt_dir} ...")
        # The TRL Trainer has a `save_model()` method that will create a new checkpoint folder:
        _global_trainer.save_model()  
        print("[checkpoint_and_exit.py] Checkpoint forced!")

    except Exception as e:
        print(f"[checkpoint_and_exit.py] Exception while forcing checkpoint: {e}")

if __name__ == "__main__":
    force_checkpoint_and_exit()
    print("[checkpoint_and_exit.py] Exiting now.")
    # After this script returns, spot_watcher.sh will exit, and the main train.py process
    # will be killed by AWS within ~2 minutes.
