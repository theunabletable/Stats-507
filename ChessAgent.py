import numpy as np
import random
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import ChessEnvironment
import ChessActionSpace
from mcts import MCTS

#Chess Agent implementation for Chess AI
#Programmer: Andrew Jones and Claude
#November 2024

"""
Chess agents form the decision-making layer of the system.
The abstract base class defines the interface, while concrete implementations
provide different strategies from random play to MCTS-based decision making.
"""
class ChessAgent(ABC):
    """Abstract base class for chess agents."""
    
    def __init__(self, action_space):
        """
        Initialize the chess agent.
        
        Args:
            action_space (ChessActionSpace): The action space object for move conversion
        """
        self.action_space = action_space
    
    @abstractmethod
    def select_action(self, state, legal_moves):
        """
        Select an action given the current state and legal moves.
        
        Args:
            state (np.ndarray): The current state (20x8x8)
            legal_moves (list): List of legal moves in UCI format
            
        Returns:
            int: The index of the selected action in the action space
        """
        pass
    
    def update(self, state, action, reward, next_state, done):
        """
        Update the agent's knowledge based on experience.
        
        Args:
            state (np.ndarray): The current state
            action (int): The action taken
            reward (float): The reward received
            next_state (np.ndarray): The resulting state
            done (bool): Whether the episode is complete
        """
        pass


#Random agent serves as both a baseline and an opponent during training.
#It helps evaluate if MCTS/neural network agents are actually learning.
class RandomAgent(ChessAgent):
    """Agent that selects random legal moves."""
    
    def select_action(self, state, legal_moves):
        """Select a random legal move."""
        # Convert legal moves to action indices
        legal_indices = []
        for move in legal_moves:
            try:
                index = self.action_space.move_to_index(move.uci())
                legal_indices.append(index)
            except ValueError:
                continue
        
        if not legal_indices:
            raise ValueError("No legal moves available")
        
        return random.choice(legal_indices)

#MCTSAgent is the main agent of this project, using Monte Carlo Tree Search to make decisions.
class MCTSAgent(ChessAgent):
    def __init__(self, action_space, policy_network, num_simulations=800):
        super().__init__(action_space)
        
        self.mcts = MCTS(policy_network, action_space, num_simulations, c_puct=5.0)
            
    def select_action(self, env, legal_moves, temperature=1.0):
        """
        Select an action using MCTS with temperature.
        
        Args:
            env: The environment
            legal_moves: List of legal moves
            temperature (float): Temperature parameter for exploration
        """
        move, _ = self.mcts.select_move(env, temperature=temperature)
        return self.action_space.move_to_index(move.uci())
    