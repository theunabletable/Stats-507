from datetime import datetime
import chess
import os

#core chess game logic using the chess library, for use in a variety of applications
#Programmer: Andrew Jones and Claude AI
#November 2024

class ChessGame:
    def __init__(self):
        self.board = chess.Board()

    def get_legal_moves(self):
        return list(self.board.legal_moves)

    def make_move(self, move):
        """Make a move if it's legal, returns True if successful, else False"""
        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        return False

    def reset_game(self):
        self.board.reset()

    def get_board_state(self):
        """Return a copy of the current board state (e.g., for analysis or GUI updates)"""
        return self.board

    def is_game_over(self):
        """Check if the game is over (checkmate, stalemate, draw, etc.)"""
        return self.board.is_game_over()

    def get_game_result(self):
        """Return a message indicating the result of the game"""
        if self.board.is_checkmate():
            return "1-0" if self.board.turn == chess.BLACK else "0-1"
        elif self.board.is_stalemate() or self.board.can_claim_fifty_moves() or self.board.can_claim_threefold_repetition() or self.board.is_insufficient_material():
            return "1/2-1/2"
        return "Game over!"


    def get_move_list(self):
        """Return a list of moves in algebraic notation"""
        move_list = []
        temp_board = chess.Board()  # Create a temporary board to replay moves
        for move in self.board.move_stack:
            move_list.append(temp_board.san(move))
            temp_board.push(move)
        return move_list

    def save_game(self):
        """Save the current game in PGN format with a timestamped filename"""
        try:
            # Create a new PGN game object
            game = chess.pgn.Game()

            # Set headers for metadata
            game.headers["Event"] = "Self-Play"
            game.headers["Site"] = "?"
            game.headers["Date"] = datetime.now().strftime("%Y-%m-%d")
            game.headers["Round"] = "?"
            game.headers["White"] = "?"
            game.headers["Black"] = "?"
            game.headers["Result"] = "*"
            game.headers["Time"] = datetime.now().strftime("%H:%M:%S")

            # Add moves from the board to the PGN game
            node = game
            for move in self.board.move_stack:
                node = node.add_variation(move)

            # Create a unique filename with millisecond precision
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

        except Exception as e:
            print(f"Failed to save game: {e}")

    def copy(self):
        """Create a deep copy of the game state"""
        new_game = ChessGame()
        new_game.board = self.board.copy()
        return new_game