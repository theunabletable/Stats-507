import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from datetime import datetime
import signal
import sys
# Local imports
from ChessGame import ChessGame
from ChessEnvironment import ChessEnvironment, RewardStrategy, ChessRewardFunction
from ChessActionSpace import ChessActionSpace
from ChessAgent import RandomAgent, MCTSAgent
from policy_network import PolicyNetwork
from ExperienceReplay import ExperienceReplay
from ChessTrainer import ChessTrainer
import logging
from typing import Dict, List, Tuple, Any 

#Main training script for Chess AI
#Programmer: Andrew Jones and Claude
#November 2024

"""
This script:
Initializes neural network.
Sets up exit handling to save progress.
Runs main training loop, adjustable number of games using num_games.
Handles checkpointing.
"""



if __name__ == "__main__":
    policy_net = PolicyNetwork()
    trainer = ChessTrainer(policy_net, num_simulations=100, 
                           reward_strategy = RewardStrategy.AGGRESSIVE)
    
    # Define signal handler
    def signal_handler(sig, frame):
        print("\nCtrl+C detected! Saving checkpoint before exit...")
        if hasattr(trainer, 'save_checkpoint'):
            trainer.save_checkpoint(trainer.total_games, trainer.optimizer)
        print("Checkpoint saved. Exiting...")
        sys.exit(0)
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        trainer.train(
            num_games=100,
            batch_size=64,
            save_frequency=50
        )
    except Exception as e:
        print(f"Error during training: {e}")
        print("Saving checkpoint before exit...")
        if hasattr(trainer, 'save_checkpoint'):
            trainer.save_checkpoint(trainer.total_games, trainer.optimizer)
        raise e