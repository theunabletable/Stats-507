from ChessGame import ChessGame
import chess
import torch
import numpy as np
import chess.pgn
from ChessActionSpace import ChessActionSpace
from datetime import datetime
import os
import time
from typing import Dict, Optional
from enum import Enum

#Chess Environment implementation for Chess AI
#Programmer: Andrew Jones and Claude
#November 2024

"""
ChessEnvironment manages the game state, move application, and tensor state generation.
It serves as the interface between the MCTS algorithm and the actual chess game,
providing state tensors for the neural network and handling reward calculation.

The state tensor is a 20x8x8 representation where:
- Planes 0-11: Piece positions (6 pieces * 2 colors)
- Plane 12: Current turn
- Planes 13-16: Castling rights
- Plane 17: En passant
- Plane 18: Move counter
- Plane 19: Repetition state
"""
class ChessEnvironment:
    # Class-level shared action space
    _shared_action_space = None
    
    def __init__(self, shared_action_space=None, reward_function = None, mcts_is_white=True):
        """Initialize the Chess Environment with a new ChessGame instance."""
        self.game = ChessGame()
        self.done = False
        self.reward_function = reward_function or ChessRewardFunction()
        self.mcts_is_white = mcts_is_white  # Store MCTS color
        # Use provided action space or get/create shared one
        if shared_action_space is not None:
            self.action_space = shared_action_space
        else:
            if ChessEnvironment._shared_action_space is None:
                ChessEnvironment._shared_action_space = ChessActionSpace()
            self.action_space = ChessEnvironment._shared_action_space
            
        self.moves = []  # Track moves made in the game
        self.replay_buffer = []  # Replay buffer to store experience tuples
        self.previous_board = self.game.board.copy()  # Store previous state

    #Reset the environment to start a new game.
    def reset(self):
        self.game.reset_game()
        self.done = False
        self.moves = []  # Clear moves for new game
        self.replay_buffer = []  # Clear replay buffer for new game
        return self.get_state()


    #Step applies a move, returns a new state, reward, and done flag.
    def step(self, action_index):
        if self.done:
            raise Exception("Cannot take a step in a finished game. Please reset the environment.")
            
        # Store previous board for reward calculation
        self.previous_board = self.game.board.copy()
        
        # Convert action index to move
        move = self.action_space.index_to_move(action_index)
        uci_move = chess.Move.from_uci(move)
        
        # Make move
        if uci_move in self.game.board.legal_moves:
            move_success = self.game.make_move(uci_move)
            if not move_success:
                raise Exception(f"Move {uci_move} was legal but failed to apply")
            self.moves.append(uci_move)
        else:
            raise Exception(f"Illegal move attempted: {uci_move}")
            
        # Calculate reward using mcts_is_white
        reward = self.reward_function.calculate_reward(
            self.game.board,
            self.previous_board,
            self.game.is_game_over(),
            mcts_is_white=self.mcts_is_white
        )
        
        # Update game state
        self.done = self.game.is_game_over()
        state = self.get_state()
        
        # Save experience
        self.replay_buffer.append((state, action_index, reward, self.done))
        
        if self.done:
            self.save_game_to_pgn()
            
        return state, reward, self.done
    

    #creates/updates the 20x8x8 tensor representing current game state
    def get_state(self, out=None) -> torch.Tensor:

        if not hasattr(self, 'device'):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
        state = out if out is not None else torch.zeros((20, 8, 8), device=self.device)
        state.zero_()
        board = self.game.board
        
        
        # Create piece array with numpy
        piece_array = np.zeros((12, 8, 8), dtype=np.float32)
        piece_map = board.piece_map()
        
        # Convert to numpy indices
        for square, piece in piece_map.items():
            rank, file = square // 8, square % 8
            plane_idx = (piece.piece_type - 1) * 2 + (0 if piece.color else 1)
            piece_array[plane_idx, rank, file] = 1.0
        
        # Convert to tensor in one operation
        state[:12].copy_(torch.from_numpy(piece_array).to(self.device))
        

        # Castling setup       
        castling_values = torch.tensor([
            self.game.board.has_kingside_castling_rights(chess.WHITE),
            self.game.board.has_queenside_castling_rights(chess.WHITE),
            self.game.board.has_kingside_castling_rights(chess.BLACK),
            self.game.board.has_queenside_castling_rights(chess.BLACK)
        ], device=self.device)
        state[13:17].copy_(castling_values.view(-1, 1, 1).expand(-1, 8, 8))

        # Remaining planes

        state[12].fill_(1 if self.game.board.turn == chess.WHITE else 0)
        if self.game.board.ep_square is not None:
            state[17, self.game.board.ep_square // 8, self.game.board.ep_square % 8] = 1
        if self.game.board.halfmove_clock > 0:
            pos = min(self.game.board.halfmove_clock, 63)
            state[18, pos // 8, pos % 8] = 1
        if self.game.board.is_repetition(2):
            state[19, 0, 0] = 1
            if self.game.board.is_repetition(3):
                state[19, 0, 1] = 1


        return state

    




#Helper function used in potential reward calculations.
    def get_piece_value(self, piece):
        if piece is None:
            return 0
        if piece.piece_type == chess.PAWN:
            return 1
        elif piece.piece_type == chess.KNIGHT or piece.piece_type == chess.BISHOP:
            return 3
        elif piece.piece_type == chess.ROOK:
            return 5
        elif piece.piece_type == chess.QUEEN:
            return 9
        else:
            return 0  # King is not captured in chess, so no value is assigned

#Helper function to save full games as a PGN file.
    def save_game_to_pgn(self):
        # Create a new PGN game object
        game = chess.pgn.Game()

        # Set headers for metadata
        game.headers["Event"] = "Reinforcement Learning Game"
        game.headers["Site"] = "Local"
        game.headers["Date"] = datetime.now().strftime("%Y-%m-%d")
        game.headers["Round"] = "?"
        game.headers["White"] = "RL Agent"
        game.headers["Black"] = "Random Opponent"
        game.headers["Result"] = self.game.get_game_result()

        # Add moves to the PGN object
        node = game
        for move in self.moves:
            node = node.add_variation(move)

        # Create a unique filename using the current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        directory = r"C:\Users\Drew\Desktop\classes\Stats 507\Final Project\Chess Engine\replays"

        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = os.path.join(directory, f"{timestamp}.pgn")

        # Save the game to the file
        with open(filename, 'w') as pgn_file:
            print(game, file=pgn_file)

        print(f"Game saved to {filename}")
        
"""
The reward system consists of an Enum of strategies and a reward calculation class.
Different reward strategies encourage different playing styles, from pure win/loss
rewards to complex positional evaluation.
"""
class RewardStrategy(Enum):
    STANDARD = "standard"  # Original reward scheme
    ASYMMETRIC_MATERIAL = "asymmetric_material"  # Reward captures, don't punish losses
    CHECKMATE_ONLY = "checkmate_only"  # Only reward for winning/losing
    AGGRESSIVE = "aggressive"  # Heavily reward captures and attacking moves
    POSITIONAL = "positional"  # Reward center control and piece development
    
#Each strategy has unique config values that determine how different events are rewarded.
#For example, AGGRESSIVE has high capture_reward and win_reward to encourage attacking play,
#while POSITIONAL gives small rewards for controlling key squares.


class ChessRewardFunction:
    def __init__(self, strategy: RewardStrategy = RewardStrategy.STANDARD, config: Optional[Dict] = None):
        self.strategy = strategy
        
        # Default configurations for different strategies
        self.strategy_configs = {
            RewardStrategy.STANDARD: {
                'material_weight': 0.01,
                'win_reward': 1.5,
                'loss_penalty': -1.5,
                'draw_penalty': -0.15,
                'use_material': True
            },
            RewardStrategy.ASYMMETRIC_MATERIAL: {
                'capture_reward': 0.02,  # Higher reward for captures
                'win_reward': 1.5,
                'loss_penalty': -1.5,
                'draw_penalty': -0.15,
                'use_material': True
            },
            RewardStrategy.CHECKMATE_ONLY: {
                'win_reward': 2.0,  # Higher win reward since it's the only focus
                'loss_penalty': -2.0,
                'draw_penalty': -0.1,
                'use_material': False
            },
            RewardStrategy.AGGRESSIVE: {
                'capture_reward': 0.4,
                'win_reward': 3,
                'loss_penalty': -3,
                'draw_penalty': -0.5,
                'use_material': True
            },
            RewardStrategy.POSITIONAL: {
                'center_control_reward': 0.02,
                'development_reward': 0.01,
                'win_reward': 1.5,
                'loss_penalty': -1.5,
                'draw_penalty': -0.15,
                'use_material': True
            }
        }
        
        # Use provided config or default for selected strategy
        self.config = config or self.strategy_configs[strategy]
        
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        
        # Center squares for positional evaluation
        self.center_squares = {chess.E4, chess.D4, chess.E5, chess.D5}
        self.extended_center = {
            chess.E4, chess.D4, chess.E5, chess.D5,  # Center
            chess.C3, chess.D3, chess.E3, chess.F3,  # Extended center
            chess.C4, chess.F4, chess.C5, chess.F5,
            chess.C6, chess.D6, chess.E6, chess.F6
        }
        
    def calculate_material_change(self, board: chess.Board, previous_board: chess.Board) -> tuple:
        """Calculate material change for both colors."""
        def get_material_count(board):
            white_material = 0
            black_material = 0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    value = self.piece_values.get(piece.piece_type, 0)
                    if piece.color == chess.WHITE:
                        white_material += value
                    else:
                        black_material += value
            return white_material, black_material
        
        current_white, current_black = get_material_count(board)
        previous_white, previous_black = get_material_count(previous_board)
        
        white_change = current_white - previous_white
        black_change = current_black - previous_black
        
        return white_change, black_change
        
    def calculate_reward(self, board: chess.Board, previous_board: chess.Board, 
                        game_over: bool, mcts_is_white: bool) -> float:
        """Calculate reward based on the selected strategy."""
        reward = 0.0
        
        if self.strategy == RewardStrategy.ASYMMETRIC_MATERIAL:
            # Calculate material changes
            white_change, black_change = self.calculate_material_change(board, previous_board)
            
            # Only reward captures, don't punish losses
            if mcts_is_white:
                if black_change < 0:  # White captured black piece
                    reward += abs(black_change) * self.config['capture_reward']
            else:
                if white_change < 0:  # Black captured white piece
                    reward += abs(white_change) * self.config['capture_reward']
                    
        elif self.strategy == RewardStrategy.AGGRESSIVE:
            # Calculate material changes
            white_change, black_change = self.calculate_material_change(board, previous_board)
            
            # Only reward captures of enemy pieces
            if mcts_is_white:
                if black_change < 0:  # White captured black piece
                    reward += abs(black_change) * self.config['capture_reward']
            else:
                if white_change < 0:  # Black captured white piece
                    reward += abs(white_change) * self.config['capture_reward']
            
            # Handle game-over conditions
            if game_over:
                if board.is_checkmate():
                    if board.turn != mcts_is_white:  # MCTS won
                        reward += self.config['win_reward']
                    else:  # MCTS lost
                        reward += self.config['loss_penalty']
                elif board.is_stalemate() or board.is_insufficient_material() or \
                    board.is_fifty_moves() or board.is_repetition():
                    reward += self.config['draw_penalty']
                    
            return reward
                            
        elif self.strategy == RewardStrategy.POSITIONAL:
            # Reward center control
            for square in self.extended_center:
                piece = board.piece_at(square)
                if piece and piece.color == mcts_is_white:
                    reward += self.config['center_control_reward']
            
            # Reward piece development
            if not previous_board.move_stack:  # Skip for first move
                return 0
                
            last_move = previous_board.move_stack[-1]
            if last_move:
                piece = board.piece_at(last_move.to_square)
                if piece and piece.color == mcts_is_white:
                    if piece.piece_type in [chess.KNIGHT, chess.BISHOP] and \
                       last_move.from_square in [chess.B1, chess.G1, chess.B8, chess.G8]:
                        reward += self.config['development_reward']
        
        # Handle game-over conditions for all strategies
        if game_over:
            if board.is_checkmate():
                if board.turn != mcts_is_white:  # MCTS just delivered checkmate
                    reward += self.config['win_reward']
                else:  # MCTS was checkmated
                    reward += self.config['loss_penalty']
            elif board.is_stalemate() or board.is_insufficient_material() or \
                 board.is_fifty_moves() or board.is_repetition():
                reward += self.config['draw_penalty']
                
        return reward