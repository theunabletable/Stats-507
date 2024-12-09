import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from datetime import datetime

from ChessGame import ChessGame
from ChessEnvironment import ChessEnvironment, ChessRewardFunction, RewardStrategy
from ChessActionSpace import ChessActionSpace
from ChessAgent import RandomAgent, MCTSAgent
from policy_network import PolicyNetwork
from ExperienceReplay import ExperienceReplay

import logging
from typing import Dict, List, Tuple, Any 

#Chess Training implementation for Chess AI
#Programmer: Andrew Jones and Claude
#November 2024

"""
ChessTrainer orchestrates the training process. It manages games between MCTS and random agents,
collects game experiences, and updates the neural network.
"""

class ChessTrainer:
    def __init__(self, policy_network, save_dir="checkpoints", 
                 reward_strategy=RewardStrategy.STANDARD, reward_config=None,
                 num_simulations=800):
        """
        Initialize the chess trainer.
        
        Args:
            policy_network (PolicyNetwork): The neural network to be trained
            save_dir (str): Directory to save checkpoints
            num_simulations (int): Number of MCTS simulations per move
        """
        self.policy_network = policy_network
        self.save_dir = save_dir
        self.action_space = ChessActionSpace()
        
        # Initialize reward function
        self.reward_function = ChessRewardFunction(
            strategy=reward_strategy,
            config=reward_config
        )
        # Create agents
        self.mcts_agent = MCTSAgent(
            action_space=self.action_space,
            policy_network=policy_network,
            num_simulations=num_simulations
        )
        self.random_agent = RandomAgent(self.action_space)
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Training metrics
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_games = 0
        self.optimizer = None  # Will be set in train method        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network.to(self.device)
        
    def save_checkpoint(self, episode, optimizer):
        """Save training state."""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws
        }
        # Save both numbered and latest checkpoint
        path = os.path.join(self.save_dir, f"checkpoint_{episode}.pt")
        latest_path = os.path.join(self.save_dir, "checkpoint.pt")
        torch.save(checkpoint, path)
        torch.save(checkpoint, latest_path)  # Also save as latest
        print(f"Checkpoints saved: {path} and {latest_path}")
        
    def play_training_game(self, mcts_as_white=True, temperature=1.0):
        """
        Play one training game: MCTS vs Random.
        
        Args:
            mcts_as_white (bool): Whether MCTS agent plays white
        
        Returns:
            list: List of experiences (state, mcts_probs, action, reward, next_state, done, game_outcome)
        """
        env = ChessEnvironment(
            mcts_is_white=mcts_as_white,
            reward_function=self.reward_function
        )
        state = env.get_state()
        done = False
        
        white_agent = self.mcts_agent if mcts_as_white else self.random_agent
        black_agent = self.random_agent if mcts_as_white else self.mcts_agent
        
        game_experiences = []
        move_count = 0
        
        try:
            while not done and move_count < 200:
                current_agent = white_agent if env.game.board.turn else black_agent
                legal_moves = env.game.get_legal_moves()
                
                # Print board state
                current_color = "White" if env.game.board.turn else "Black"
                current_agent_type = "MCTS" if current_agent == self.mcts_agent else "Random"
                print(f"\nMove {move_count}: {current_color} ({current_agent_type}) to play")
                print(f"Board:\n{env.game.board}")
                
                if current_agent == self.mcts_agent:
                    mcts_probs, _ = self.mcts_agent.mcts.run_search(env)
                    action_index = self.mcts_agent.select_action(env, legal_moves, temperature=temperature)
                else:
                    action_index = self.random_agent.select_action(state, legal_moves)
                    mcts_probs = None
                
                move = self.action_space.index_to_move(action_index)
                print(f"Move made: {move}")
                try:
                    next_state, reward, done = env.step(action_index)
                    print(f"Move {move_count}: Reward = {reward}")
                    
                    if current_agent == self.mcts_agent:
                        # Only store experiences when MCTS makes a move, with unmodified reward
                        experience = (
                            state,
                            mcts_probs,
                            action_index,
                            reward,  # Don't negate the reward
                            next_state,
                            done,
                            None  # Placeholder - will be updated after game ends
                        )
                        game_experiences.append(experience)
                        print(f"Experience stored: reward={reward}")
                    
                except Exception as e:
                    print(f"Error making move: {e}")
                    break
                
                state = next_state
                move_count += 1
                
                # Force draw if game is too long
                if move_count >= 200:
                    done = True
                    print("Game drawn by move limit")
                    
                    # Add final experience with draw outcome
                    if current_agent == self.mcts_agent:
                        game_experiences.append((
                            state,
                            mcts_probs,
                            action_index,
                            reward,
                            next_state,
                            True,  # done
                            0  # draw outcome
                        ))
                    
        except Exception as e:
            print(f"Game error: {e}")
            import traceback
            traceback.print_exc()
        

        # Update statistics
        if done:
            result = env.game.get_game_result()
            print(f"\nGame Result: {result}")
            
            # Use reward function's configured values for final outcome
            if result == '1-0':  # White won
                final_outcome = self.reward_function.config['win_reward'] if mcts_as_white else self.reward_function.config['loss_penalty']
            elif result == '0-1':  # Black won
                final_outcome = self.reward_function.config['loss_penalty'] if mcts_as_white else self.reward_function.config['win_reward']
            else:  # Draw
                final_outcome = self.reward_function.config['draw_penalty']
                
            # Update ALL experiences with the final outcome
            game_experiences = [
                (exp[0], exp[1], exp[2], exp[3], exp[4], exp[5], final_outcome)
                for exp in game_experiences
            ]
            
            # Rest of the statistics updating code remains the same
            if result == '1-0':
                if mcts_as_white:
                    self.wins += 1
                    print("Added win for MCTS (White)")
                else:
                    self.losses += 1
                    print("Added loss for MCTS (Black)")
            elif result == '0-1':
                if mcts_as_white:
                    self.losses += 1
                    print("Added loss for MCTS (White)")
                else:
                    self.wins += 1
                    print("Added win for MCTS (Black)")
            else:
                self.draws += 1
                print("Added draw")
            
            print(f"After update - W/L/D: {self.wins}/{self.losses}/{self.draws}")
                
            print(f"\nGame Statistics:")
            print(f"Total moves: {move_count}")
            print(f"MCTS played as: {'White' if mcts_as_white else 'Black'}")
            print(f"Current W/L/D: {self.wins}/{self.losses}/{self.draws}")
        
        return game_experiences
    
    def train(self, num_games=2, batch_size=64, save_frequency=50):
        """
        Main training loop.
        
        Args:
            num_games (int): Number of games to play
            batch_size (int): Size of training batches (not used anymore since we train on full games)
            save_frequency (int): Save checkpoint every N games
        """
        # Try to load previous checkpoint
        checkpoint_path = "checkpoint.pt"
        if os.path.exists(checkpoint_path):
            print("Loading previous checkpoint...")
            checkpoint = torch.load(checkpoint_path)
            self.policy_network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_game = checkpoint['games_played']
            self.wins = checkpoint.get('wins', 0)
            self.losses = checkpoint.get('losses', 0)
            self.draws = checkpoint.get('draws', 0)
            print(f"Resuming from game {start_game} with W/L/D: {self.wins}/{self.losses}/{self.draws}")
        else:
            start_game = 0
            print("Starting fresh training...")
        
        # Initialize optimizer and scaler
        self.optimizer = torch.optim.Adam(self.policy_network.parameters())
        scaler = torch.amp.GradScaler()
        
        # Training metrics
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        games_played = start_game
        
        print(f"Starting training for {num_games} games...")
        print(f"Save frequency: {save_frequency}")
        
        for game_num in range(start_game, num_games):
            try:
                # Alternate colors and calculate temperature
                mcts_as_white = (game_num % 2 == 0)
                temperature = max(1.0 * (1 - games_played/num_games), 0.05)
                
                print(f"\nGame {game_num + 1}/{num_games}")
                print(f"MCTS playing as {'White' if mcts_as_white else 'Black'}")
                print(f"Temperature: {temperature:.2f}")
                
                # Play game and get experiences
                experiences = self.play_training_game(mcts_as_white, temperature=temperature)
                print(f"Experiences from this game: {len(experiences)}")
                
                # Create fresh buffer for this game's experiences
                buffer = ExperienceReplay(capacity=len(experiences))
                for exp in experiences:
                    buffer.add_experience(*exp)

                # Train on all experiences from this game
                if len(buffer) > 0:
                    print(f"Training on {len(buffer)} experiences from this game...")
                    batch = buffer.sample(len(buffer))  # Get all experiences
                    print(f"Rewards in batch: {[f'{r:.2f}' for r in batch['rewards']]}")
                    print(f"Outcomes in batch: {[f'{o:.2f}' if o is not None else 'None' for o in batch['outcomes']]}")
                    print("\nSample of states and predictions:")
                  #  print(f"Value predictions: {value_pred[:5].squeeze().cpu().numpy()}")
                 #   print(f"Actual outcomes: {outcomes[:5].cpu().numpy()}")
                # Tensor conversion with None handling
                if isinstance(batch['states'][0], torch.Tensor):
                    states = torch.stack(batch['states']).to(self.device)
                else:
                    states = torch.FloatTensor(np.array(batch['states'])).to(self.device)

                # Handle mcts_probs with None values
                valid_moves = [i for i, probs in enumerate(batch['mcts_probs']) if probs is not None]
                if len(valid_moves) > 0:
                    # Only use experiences where we have MCTS probabilities
                    states = states[valid_moves]
                    valid_mcts_probs = [batch['mcts_probs'][i] for i in valid_moves]
                    
                    if isinstance(valid_mcts_probs[0], torch.Tensor):
                        mcts_probs = torch.stack(valid_mcts_probs).to(self.device)
                    else:
                        mcts_probs = torch.FloatTensor(np.array(valid_mcts_probs)).to(self.device)

                    actions = torch.LongTensor([batch['actions'][i] for i in valid_moves]).to(self.device)
                    rewards = torch.FloatTensor([batch['rewards'][i] for i in valid_moves]).to(self.device)
                    outcomes = [batch['outcomes'][i] if batch['outcomes'][i] is not None else 0 for i in valid_moves]
                    outcomes = torch.FloatTensor(outcomes).to(self.device)
                    
                    # Training step with mixed precision
                    with torch.amp.autocast('cuda'):
                        policy_logits, value_pred = self.policy_network(states)
                        
                        # Use mcts_probs for policy learning instead of actions
                        c_value = 1
                        c_policy = 1
                        policy_loss = -torch.sum(mcts_probs * F.log_softmax(policy_logits, dim=1)) / len(valid_moves)
                        value_loss = F.mse_loss(value_pred.squeeze(-1), outcomes)
                        loss = c_policy * policy_loss + c_value * value_loss
                    
                    # Store losses
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_loss += loss.item()
                    
                    # Optimize
                    self.optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    
                    print(f"Game losses - Policy: {policy_loss.item():.4f}, Value: {value_loss.item():.4f}, Overall: {loss.item():.4f}")

                # Progress reporting and checkpoints
                games_played += 1
                if games_played % 10 == 0:
                    avg_loss = total_loss / games_played if games_played > 0 else 0
                    avg_policy_loss = total_policy_loss / games_played if games_played > 0 else 0
                    avg_value_loss = total_value_loss / games_played if games_played > 0 else 0
                    
                    print(f"\nTraining Progress:")
                    print(f"Games Played: {games_played}")
                    print(f"Average Total Loss: {avg_loss:.4f}")
                    print(f"Average Policy Loss: {avg_policy_loss:.4f}")
                    print(f"Average Value Loss: {avg_value_loss:.4f}")
                    print(f"Win Rate: {(self.wins / games_played) * 100:.1f}%")
                    print(f"W/L/D: {self.wins}/{self.losses}/{self.draws}")
                
                if games_played % save_frequency == 0:
                    self.save_checkpoint(games_played, self.optimizer)
                    
            except KeyboardInterrupt:
                print("\nTraining interrupted by user")
                self.save_checkpoint(games_played, self.optimizer)
                break
                
            except Exception as e:
                print(f"\nError during training: {e}")
                import traceback
                traceback.print_exc()
                self.save_checkpoint(games_played, self.optimizer)
                continue
        
        print("\nTraining completed!")
        print(f"Final Statistics:")
        print(f"Games Played: {games_played}")
        print(f"Final Win Rate: {(self.wins / games_played) * 100:.1f}%")
        print(f"W/L/D: {self.wins}/{self.losses}/{self.draws}")
        
        # Save final checkpoint
        self.save_checkpoint(games_played, self.optimizer)
        return games_played