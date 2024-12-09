import random
import numpy as np


#Experience replay buffer for training Chess AI
#Programmer: Andrew Jones and Claude
#November 2024

"""
ExperienceReplay stores training experiences from self-play games.
Each experience contains state, move probabilities from MCTS, action taken,
reward received, and game outcome. These experiences are randomly sampled 
during training to update the neural network.
"""
class ExperienceReplay:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add_experience(self, state, mcts_probs, action, reward, next_state, done, game_outcome=None):
        experience = (state, mcts_probs, action, reward, next_state, done, game_outcome)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            # Overwrite the oldest experience if buffer is full
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack the experiences into separate arrays
        states, mcts_probs, actions, rewards, next_states, dones, outcomes = zip(*batch)
        
        return {
            'states': states,
            'mcts_probs': mcts_probs,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'outcomes': outcomes
        }
    
    def __len__(self):
        return len(self.buffer)