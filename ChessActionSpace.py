import torch
import chess
import numpy as np
#ChessActionSpace move encoding implementation for Chess AI
#Programmer: Andrew Jones and Claude
#November 2024

"""
ChessActionSpace creates and manages the mapping between chess moves as they are in the ChessGame
class (e.g. "e2e4") and action indices. We create a fixed list of all possible moves and reference them by index.

"""
class ChessActionSpace:
    def __init__(self):
        self.all_possible_moves = self.generate_all_moves()
        
    def generate_all_moves(self):
        """
        Creates a comprehensive list of all possible chess moves using:
        All possible queen moves.
        All knight moves.
        Promotion handling
        Castling handling.
        """
        moves = set()
        board = chess.BaseBoard.empty()

        # Generate all possible from->to moves using queen + knight patterns
        for from_square in chess.SQUARES:
            # Get queen moves (covers straight and diagonal moves)
            board.set_piece_at(from_square, chess.Piece.from_symbol('Q'))
            queen_moves = board.attacks(from_square)
            board.remove_piece_at(from_square)
            
            # Get knight moves
            board.set_piece_at(from_square, chess.Piece.from_symbol('N'))
            knight_moves = board.attacks(from_square)
            board.remove_piece_at(from_square)
            
            # Combine moves
            all_moves = queen_moves | knight_moves
            
            # Convert to UCI notation
            from_square_name = chess.square_name(from_square)
            for to_square in chess.SQUARES:
                if all_moves.tolist()[to_square]:
                    moves.add(f"{from_square_name}{chess.square_name(to_square)}")

        # Add pawn promotion moves
        promotion_pieces = ['n', 'b', 'r', 'q']
        
        # White pawn promotions
        for file in range(8):
            from_square = chess.square(file, 6)  # 7th rank
            # Forward promotions
            to_square = chess.square(file, 7)
            base_move = f"{chess.square_name(from_square)}{chess.square_name(to_square)}"
            moves.add(base_move + 'q')  # Queen promotion already counted
            for piece in ['n', 'b', 'r']:  # Underpromotions
                moves.add(base_move + piece)
            
            # Capture promotions
            for file_offset in [-1, 1]:
                if 0 <= file + file_offset <= 7:
                    to_square = chess.square(file + file_offset, 7)
                    base_move = f"{chess.square_name(from_square)}{chess.square_name(to_square)}"
                    moves.add(base_move + 'q')  # Queen promotion already counted
                    for piece in ['n', 'b', 'r']:  # Underpromotions
                        moves.add(base_move + piece)

        # Black pawn promotions
        for file in range(8):
            from_square = chess.square(file, 1)  # 2nd rank
            # Forward promotions
            to_square = chess.square(file, 0)
            base_move = f"{chess.square_name(from_square)}{chess.square_name(to_square)}"
            moves.add(base_move + 'q')  # Queen promotion already counted
            for piece in ['n', 'b', 'r']:  # Underpromotions
                moves.add(base_move + piece)
            
            # Capture promotions
            for file_offset in [-1, 1]:
                if 0 <= file + file_offset <= 7:
                    to_square = chess.square(file + file_offset, 0)
                    base_move = f"{chess.square_name(from_square)}{chess.square_name(to_square)}"
                    moves.add(base_move + 'q')  # Queen promotion already counted
                    for piece in ['n', 'b', 'r']:  # Underpromotions
                        moves.add(base_move + piece)

        # Add castling moves (can be represented as king moves e1g1, e1c1, e8g8, e8c8)
        castling_moves = ['e1g1', 'e1c1', 'e8g8', 'e8c8']
        moves.update(castling_moves)

        return sorted(list(moves))

    def move_to_index(self, move):
        """
        Convert a UCI move string to an index in the action space.
        """
        try:
            return self.all_possible_moves.index(move)
        except ValueError:
            raise ValueError(f"Move '{move}' is not recognized in the action space.")

    def index_to_move(self, index):
        """
        Convert an index in the action space to a UCI move string.
        """
        if index < 0 or index >= len(self.all_possible_moves):
            raise IndexError(f"Index {index} is out of bounds for action space.")
        return self.all_possible_moves[index]

    def mask_legal_moves(self) -> torch.Tensor:
        """
        Create a mask for legal moves based on the current game state, mapping legal moves to legal indices.
        """
        mask = [0] * len(self.all_possible_moves)
        for move in legal_moves:
            try:
                index = self.move_to_index(move)
                mask[index] = 1
            except ValueError:
                print(f"Warning: Move {move} not found in action space")
        return mask