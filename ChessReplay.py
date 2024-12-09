import chess.pgn

class ChessReplay:
    def __init__(self):
        self.moves = []
        self.current_index = 0
        self.temp_board = chess.Board()

    def load_game(self, filename):
        """Load a PGN file and prepare it for replay"""
        try:
            with open(filename, 'r') as pgn_file:
                game = chess.pgn.read_game(pgn_file)
                self.moves = list(game.mainline_moves())
                print(f"Loaded {len(self.moves)} moves from {filename}")  # Debugging line
            self.reset()
        except FileNotFoundError:
            print(f"File not found: {filename}")
        except Exception as e:
            print(f"Failed to load game: {e}")

    def next_move(self):
        """Move to the next move in the replay"""
        if self.current_index < len(self.moves):
            move = self.moves[self.current_index]
            self.temp_board.push(move)
            self.current_index += 1
            return True
        return False

    def previous_move(self):
        """Undo the last move in the replay"""
        if self.current_index > 0:
            self.temp_board.pop()
            self.current_index -= 1
            return True
        return False

    def reset(self):
        """Reset the replay to the start of the game"""
        self.temp_board.reset()
        self.current_index = 0

    def get_current_board(self):
        """Return the current board state for rendering"""
        return self.temp_board
