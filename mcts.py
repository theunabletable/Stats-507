import math
import numpy as np
import chess
import torch
import torch.nn.functional as F
from ChessEnvironment import ChessEnvironment
from ChessGame import ChessGame
import time
from collections import defaultdict
import cProfile
import pstats
from pstats import SortKey
#MCTS algorithm implementation for use in a Chess AI
#Programmer: Andrew Jones and Claude AI
#November 2024

"""
An MCTSNode object is a single node in our tree, representing a single position.
The MCTS tree starts at a root node and creates new MCTSNodes for all possible legal moves
from that position. 
"""
class MCTSNode:
    # Class-level shared objects
    _cached_turn = True  # True for White
    _board = chess.Board()  # For move validation/generation
    _traverse_board = chess.Board()  # For tree traversal
    _game = ChessGame()
    _env = None
    _action_space = None
    _legal_moves_cache = {}
    _max_cache_size = 10000
    _working_tensor = None
    _move_stack = []
    def __init__(self, game_state, prior_probability=0, parent=None, device=None, c_puct=1.0, last_move=None):
        self.game_state = game_state
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.c_puct = c_puct
        self.last_move = last_move
        
        # Initialize shared components if needed
        if MCTSNode._env is None:
            MCTSNode._env = ChessEnvironment(shared_action_space=game_state.action_space)
            MCTSNode._action_space = game_state.action_space
            MCTSNode._working_tensor = torch.zeros((20, 8, 8), device=self.device)

        
        # Store board state as FEN (lightweight)
        self.board_fen = game_state.game.board.fen()
        
        # Initialize state tensor
        if parent is None:  # Root node
            state = game_state.get_state()
            MCTSNode._working_tensor.copy_(state)
            # Initialize traverse board at root
            MCTSNode._traverse_board.set_fen(self.board_fen)
        elif last_move:
            self.apply_move_to_working_tensor(last_move, parent.game_state.game.board)
            
        self.stats = torch.zeros(3, device=self.device)
        self.stats[2] = prior_probability
        self.parent = parent
        self.children = {}
        self._legal_moves = self.get_legal_moves()
        self._legal_move_indices = None
        self.is_terminal = game_state.game.is_game_over()
        



#apply_move_to_working_tensor takes the current tensor state and changes every plane based on
#the move chosen. We try to reuse the same tensor while traversing the tree to reduce tensor creation.
    @classmethod
    def apply_move_to_working_tensor(cls, move, board):
        from_square = move.from_square
        to_square = move.to_square
        from_rank, from_file = from_square // 8, from_square % 8
        to_rank, to_file = to_square // 8, to_square % 8
        
        # Get moving piece
        piece = board.piece_at(from_square)
        if piece is None:
            return
                
        # Calculate piece plane index once
        piece_idx = (piece.piece_type - 1) * 2 + (0 if piece.color else 1)
        
        # Queue tensor operations without syncing
        with torch.no_grad():
            # Clear source and set target
            cls._working_tensor[piece_idx, from_rank, from_file] = 0
            cls._working_tensor[piece_idx, to_rank, to_file] = 1
            
            # Handle capture
            captured = board.piece_at(to_square)
            if captured:
                cap_idx = (captured.piece_type - 1) * 2 + (0 if captured.color else 1)
                cls._working_tensor[cap_idx, to_rank, to_file] = 0
            
            # Handle castling
            if piece.piece_type == chess.KING and abs(to_square - from_square) == 2:
                rook_idx = 4 + (0 if piece.color else 1)
                if to_square > from_square:  # Kingside
                    cls._working_tensor[rook_idx, from_rank, 7] = 0
                    cls._working_tensor[rook_idx, from_rank, 5] = 1
                else:  # Queenside
                    cls._working_tensor[rook_idx, from_rank, 0] = 0
                    cls._working_tensor[rook_idx, from_rank, 3] = 1
                
                # Clear castling rights
                if piece.color:
                    cls._working_tensor[13:15] = 0  # White
                else:
                    cls._working_tensor[15:17] = 0  # Black
                    
            # Update castling rights for rook moves
            elif piece.piece_type == chess.ROOK:
                if from_square == chess.A1:
                    cls._working_tensor[14] = 0  # White queenside
                elif from_square == chess.H1:
                    cls._working_tensor[13] = 0  # White kingside
                elif from_square == chess.A8:
                    cls._working_tensor[16] = 0  # Black queenside
                elif from_square == chess.H8:
                    cls._working_tensor[15] = 0  # Black kingside
            
            # Clear old en passant
            cls._working_tensor[17] = 0
            
            # Set new en passant
            if piece.piece_type == chess.PAWN and abs(to_square - from_square) == 16:
                ep_square = (to_square + from_square) // 2
                cls._working_tensor[17, ep_square // 8, ep_square % 8] = 1
            
            # Simple turn flip using cached value
            cls._cached_turn = not cls._cached_turn
            cls._working_tensor[12] = float(cls._cached_turn)
            
            # Update move counter
            if piece.piece_type == chess.PAWN or captured:
                cls._working_tensor[18] = 0
            else:
                new_clock = min(board.halfmove_clock + 1, 63)
                if new_clock > 0:
                    cls._working_tensor[18] = 0
                    cls._working_tensor[18, new_clock // 8, new_clock % 8] = 1



#These three properties form the core statistics that MCTS uses to balance exploration and exploitation.
#Each node tracks how often it's been visited, the cumulative values from those visits, and the 
#policy network's initial estimate. These combine in the UCB formula to select promising moves.
    @property
    def visit_count(self):
        return int(self.stats[0].item())
    
    @property
    def value_sum(self):
        return self.stats[1].item()
    
    @property
    def prior_probability(self):
        return self.stats[2].item()



    
    @visit_count.setter
    def visit_count(self, value):
        self.stats[0] = value
        
            


    @value_sum.setter
    def value_sum(self, value):
        self.stats[1] = value



    
    def get_legal_moves(self):
        """Get legal moves using shared board"""
        if self.board_fen in self._legal_moves_cache:
            return self._legal_moves_cache[self.board_fen]
        
        # Use shared board for move generation
        MCTSNode._board.set_fen(self.board_fen)
        moves = list(MCTSNode._board.legal_moves)
        
        if len(self._legal_moves_cache) < self._max_cache_size:
            self._legal_moves_cache[self.board_fen] = moves[:]
        return moves
    

    


#batch_expand expands a leaf node creating child nodes, which each have prior probabilities
#given by the policy network.
    def batch_expand(self, policy_network, batch_size=32, policy_logits=None):
        """Ultra-minimal batch expansion"""
        if self.is_terminal:
            return
                
        # Get policy predictions using working tensor
        if policy_logits is None:
            with torch.no_grad():
                policy_logits, _ = policy_network(MCTSNode._working_tensor.unsqueeze(0))
        
        # Calculate probabilities
        with torch.amp.autocast("cuda"):
            with torch.no_grad():
                policy_probs = F.softmax(policy_logits, dim=1).squeeze(0)
                if self._legal_move_indices is None:
                    self._legal_move_indices = torch.tensor([
                        self._action_space.move_to_index(move.uci())
                        for move in self._legal_moves
                    ], device=self.device)
                probs = policy_probs[self._legal_move_indices]
                probs = probs / probs.sum().item()
        
        # Create children without creating new states
        for move, prob in zip(self._legal_moves, probs):
            child = MCTSNode(
                game_state=self.game_state,  # Just pass parent's game state
                prior_probability=prob,
                parent=self,
                device=self.device,
                c_puct=self.c_puct,
                last_move=move
            )
            self.children[move] = child
            

#select_child is the method which visits nodes guided by the Upper Confidence Bound formula. The node which
#se
#UCB is calculated combining exploitation (q_values, average value from previous visits)
#and exploration (prior probability weighted by sqrt(parent visits) to increase exploration as we visit parent more,
# and 1/(1 + child visits) to reduce exploration of heavily visited children.)
#c_puct is a constant which controls the balance between exploration and exploitation.
    def select_child(self):
        """Select child using stored c_puct value with randomness for exploration"""
        if not self.children:
            return None
            
        children = list(self.children.values())
        
        # Stack child statistics
        child_stats = torch.stack([child.stats for child in children])
        visit_counts = child_stats[:, 0]
        values = child_stats[:, 1]
        priors = child_stats[:, 2]
        
        # Q values with visited vs unvisited handling
        q_values = torch.where(
            visit_counts > 0,
            values / visit_counts,
            torch.zeros_like(values)
        )
        
        # Use stored c_puct value
        parent_sqrt = torch.sqrt(torch.tensor(self.visit_count + 1, device=self.device))
        exploration = self.c_puct * priors * parent_sqrt / (1 + visit_counts)
        
        # Add small random noise to break ties
        ucb_scores = q_values + exploration + torch.randn_like(values) * 1e-5
        
        # Progressive widening - reduce exploration of heavily visited nodes
        if self.visit_count > 4:
            max_visits = int(torch.sqrt(torch.tensor(self.visit_count)).item())
            over_visited = visit_counts > max_visits
            ucb_scores = torch.where(
                over_visited,
                torch.full_like(ucb_scores, float('-inf')),
                ucb_scores
            )
        
        best_idx = ucb_scores.argmax().item()
        
        return children[best_idx]
    

"""
The MCTS class implements Monte Carlo Tree Search for chess moves.
It uses a policy network to guide search and for value evaluation.
The search process creates a tree of MCTSNode objects, explores promising variations,
and selects moves based on visit counts.
"""    

class MCTS:
    def __init__(self, network, action_space, num_simulations=800, c_puct=5.0, batch_size=128):
            self.policy_network = network
            self.action_space = action_space
            self.num_simulations = num_simulations
            self.c_puct = c_puct
            self.batch_size = batch_size
            self.device = next(network.parameters()).device
            # Initialize working tensor for state reuse
            self.working_tensor = torch.zeros((20, 8, 8), device=self.device)
            # Track move history for backtracking
            self.move_stack = []

            
            # Add warmup phase
            print("Warming up policy network...")
            warmup_input = torch.zeros((1, 20, 8, 8), device=self.device)
            for _ in range(10):  # Multiple warmup passes
                with torch.no_grad():
                    self.policy_network(warmup_input)
            torch.cuda.synchronize()
            print("Warmup complete")
            
        

    #Converts visit counts at root node into move probabilities
    def get_move_probabilities(self, root, temperature=2.0):
        visits = torch.zeros(len(self.action_space.all_possible_moves), device=self.device)
        
        for move, child in root.children.items():
            move_idx = self.action_space.move_to_index(move.uci())
            visits[move_idx] = child.visit_count
        
        if temperature == 0:  # Deterministic choice
            max_visit_idx = visits.argmax()
            probs = torch.zeros_like(visits)
            probs[max_visit_idx] = 1.0
        else:
            # Apply temperature
            visits = visits ** (1.0 / temperature)
            # Normalize on GPU
            if visits.sum() > 0:
                probs = visits / visits.sum()
            else:
                # Fallback to uniform distribution for legal moves
                legal_moves = list(root.game_state.game.board.legal_moves)
                move_indices = torch.tensor([
                    self.action_space.move_to_index(move.uci())
                    for move in legal_moves
                ], device=self.device)
                probs = torch.zeros_like(visits)
                probs[move_indices] = 1.0 / len(legal_moves)
        
        return probs.cpu().numpy()  # Convert to CPU only at the end
    

     
    #Executes the MCTS algorithm from a given root_state.
    def run_search(self, root_state):
        # Get initial state tensor from environment
        state_tensor = root_state.get_state()
        # Initialize working tensor from state
        self.working_tensor.copy_(state_tensor)
        self.move_stack.clear()
        
        # Create root node
        root = MCTSNode(root_state, device=self.device, c_puct=self.c_puct)
        
        # Root expansion
        root.batch_expand(self.policy_network, batch_size=32)
        if root.children:
            noise = torch.tensor(
                np.random.dirichlet([0.3] * len(root.children)),
                device=self.device
            )
            for i, (_, child) in enumerate(root.children.items()):
                child.stats[2] = 0.75 * child.stats[2] + 0.25 * noise[i]
        
        # Main simulation loop
        for sim in range(self.num_simulations):
            node = root
            path = [node]
            
            while node.children and not node.is_terminal:
                node = node.select_child()
                # Use the class method properly
                MCTSNode.apply_move_to_working_tensor(move=node.last_move, board=node.game_state.game.board)
                path.append(node)
            
            # Expansion and evaluation
            if not node.is_terminal:
                with torch.no_grad():
                    policy_logits, value = self.policy_network(MCTSNode._working_tensor.unsqueeze(0))
                    node.batch_expand(self.policy_network, batch_size=1, policy_logits=policy_logits)
                    value = value.item()
            else:
                # Terminal node evaluation
                value = 1.0 if node.game_state.game.is_checkmate() else 0.0
                if not node.game_state.game.board.turn:  # If black to move, negate value
                    value = -value
                    
            # Backpropagate
            for path_node in reversed(path):
                path_node.value_sum += value
                path_node.visit_count += 1
                value = -value  # Flip value for opponent
                
            # NOTE: need to implement a way to undo moves at the MCTSNode level
            # for now, restore from root state
            MCTSNode._working_tensor.copy_(state_tensor)
        
        return self.get_move_probabilities(root), root
    

    #select_move ultimately chooses the move in an actual game, by picking the node which was visited the most
    #times in the search.
    def select_move(self, game_state, temperature=1.5):
        probabilities, root = self.run_search(game_state)
        legal_moves = list(game_state.game.board.legal_moves)
        
        # Get probabilities for legal moves
        legal_move_probs = np.zeros(len(legal_moves))
        for i, move in enumerate(legal_moves):
            move_idx = self.action_space.move_to_index(move.uci())
            legal_move_probs[i] = probabilities[move_idx]
        
        # Add noise to break symmetry
        legal_move_probs += np.random.dirichlet([0.3] * len(legal_moves))
        
        # Apply temperature
        if temperature != 0:
            legal_move_probs = legal_move_probs ** (1.0 / temperature)
        
        # Normalize
        legal_move_probs = legal_move_probs / legal_move_probs.sum() if legal_move_probs.sum() > 0 else np.ones(len(legal_moves)) / len(legal_moves)
        
        # Select move
        selected_idx = legal_move_probs.argmax() if temperature == 0 else np.random.choice(len(legal_moves), p=legal_move_probs)
        
        return legal_moves[selected_idx], legal_move_probs


