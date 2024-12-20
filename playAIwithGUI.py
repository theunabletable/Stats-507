import tkinter as tk
from tkinter import ttk
from ChessGUI import ChessGUI
from ChessGame import ChessGame
from ChessEnvironment import ChessEnvironment
from ChessActionSpace import ChessActionSpace
from ChessAgent import RandomAgent
import chess

class ChessGUIWithAI(ChessGUI):
    def __init__(self, root, chess_game):
        # Create main frames first
        # Store the root window
        self.root = root
        self.root.title("Chess AI Interface")
        
        # Create main frames
        self.left_frame = tk.Frame(root)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.right_frame = tk.Frame(root)
        self.right_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Initialize game and other basic attributes without calling parent's __init__
        self.game = chess_game
        self.selected_square = None
        self.promotion_square = None
        self.game_over = False
        
        # Create canvas for chess board
        self.canvas = tk.Label(self.left_frame)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Create status label
        self.status_label = tk.Label(self.left_frame, text="", font=("Arial", 14))
        self.status_label.pack()
        
        # Initialize AI components
        self.env = ChessEnvironment()
        self.action_space = ChessActionSpace()
        self.agent = RandomAgent(self.action_space)
        
        # Create turn indicator
        self.turn_indicator = tk.Label(
            self.left_frame,
            text="Current Turn: White",
            font=("Arial", 14, "bold"),
            bg="white",
            relief=tk.RIDGE,
            pady=5
        )
        self.turn_indicator.pack(pady=10)
        
        # Create move log
        self.move_log_frame = tk.Frame(self.right_frame)
        self.move_log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add move log label
        tk.Label(
            self.move_log_frame,
            text="Move Log",
            font=("Arial", 12, "bold")
        ).pack()
        
        # Create move log text widget with scrollbar
        self.move_log = tk.Text(
            self.move_log_frame,
            height=20,
            width=30,
            font=("Courier", 10),
            wrap=tk.WORD
        )
        scrollbar = tk.Scrollbar(self.move_log_frame, command=self.move_log.yview)
        self.move_log.configure(yscrollcommand=scrollbar.set)
        
        self.move_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add AI control buttons
        self.ai_control_frame = tk.Frame(self.left_frame)
        self.ai_control_frame.pack(side=tk.BOTTOM, pady=10)
        
        self.ai_move_button = tk.Button(
            self.ai_control_frame, 
            text="Make AI Move", 
            command=self.make_ai_move
        )
        self.ai_move_button.pack(side=tk.LEFT, padx=5)
        
        self.auto_play_button = tk.Button(
            self.ai_control_frame, 
            text="Start Auto Play", 
            command=self.toggle_auto_play
        )
        self.auto_play_button.pack(side=tk.LEFT, padx=5)
        
        self.move_delay = tk.Scale(
            self.ai_control_frame,
            from_=0.1,
            to=3.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            label="Move Delay (seconds)",
            length=200
        )
        self.move_delay.set(1.0)
        self.move_delay.pack(side=tk.LEFT, padx=5)
        
        self.is_auto_playing = False
        self.move_speed_ms = 1000  # Default 1 second between moves
        
        # Initial board display
        self.update_board()
        
    def update_turn_indicator(self):
        """Update the turn indicator based on current board state"""
        current_turn = "White" if self.game.board.turn == chess.WHITE else "Black"
        self.turn_indicator.config(
            text=f"Current Turn: {current_turn}",
            bg="white" if current_turn == "White" else "gray"
        )

    def log_move(self, move_uci, is_ai=False):
        """Add a move to the move log"""
        move_number = len(self.game.board.move_stack) // 2 + 1
        current_turn = "White" if len(self.game.board.move_stack) % 2 == 1 else "Black"
        player_type = "AI" if is_ai else "Human"
        
        # Convert UCI to algebraic notation
        board = chess.Board()
        for past_move in list(self.game.board.move_stack)[:-1]:
            board.push(past_move)
        move = chess.Move.from_uci(move_uci)
        algebraic = board.san(move)
        
        log_entry = f"{move_number}. {current_turn} ({player_type}): {algebraic}\n"
        self.move_log.insert(tk.END, log_entry)
        self.move_log.see(tk.END)  # Auto-scroll to the bottom
        
    def make_ai_move(self):
        """Have the AI make a single move"""
        if not self.game_over:
            try:
                # Sync environment with current game state
                self.env.game.board = self.game.board.copy()
                
                # Get current state and legal moves
                state = self.env.get_state()
                legal_moves = list(self.game.board.legal_moves)
                
                if not legal_moves:
                    print("No legal moves available")
                    self.game_over = True
                    return
                
                # Get AI's move selection
                action_index = self.agent.select_action(state, legal_moves)
                move_uci = self.action_space.index_to_move(action_index)
                
                # Verify the move is legal
                move = chess.Move.from_uci(move_uci)
                if move not in legal_moves:
                    print(f"Selected move {move_uci} is not in legal moves: {[m.uci() for m in legal_moves]}")
                    # Try to select a different move
                    action_index = self.agent.select_action(state, legal_moves)
                    move_uci = self.action_space.index_to_move(action_index)
                    move = chess.Move.from_uci(move_uci)
                
                # Make the move in both game and environment
                if self.game.make_move(move):
                    print(f"AI played: {move_uci}")
                    self.env.step(action_index)
                    
                    # Log the move
                    self.log_move(move_uci, is_ai=True)
                    
                    # Update the display and turn indicator
                    self.update_board()
                    self.update_turn_indicator()
                    
                    # Check for game end
                    if self.game.is_game_over():
                        self.game_over = True
                        result = self.game.get_game_result()
                        self.status_label.config(text=f"Game Over: {result}")
                        self.output_game_result()
                        self.game.save_game()
                        self.is_auto_playing = False
                        self.auto_play_button.config(text="Start Auto Play")
                        # Log the game result
                        self.move_log.insert(tk.END, f"\nGame Over: {result}\n")
                else:
                    print(f"Failed to make move {move_uci}")
                    self.is_auto_playing = False
                    self.auto_play_button.config(text="Start Auto Play")
                
            except Exception as e:
                print(f"Error making AI move: {e}")
                self.status_label.config(text="Error making move")
                self.is_auto_playing = False
                self.auto_play_button.config(text="Start Auto Play")
    
    def toggle_auto_play(self):
        """Toggle automatic AI play"""
        self.is_auto_playing = not self.is_auto_playing
        if self.is_auto_playing:
            self.auto_play_button.config(text="Stop Auto Play")
            self.auto_play()
        else:
            self.auto_play_button.config(text="Start Auto Play")
    
    def auto_play(self):
        """Make AI moves automatically with delay"""
        if self.is_auto_playing and not self.game_over:
            self.make_ai_move()
            delay_ms = int(self.move_delay.get() * 1000)  # Convert seconds to milliseconds
            self.root.after(delay_ms, self.auto_play)

    def on_canvas_click(self, event):
        old_move_count = len(self.game.board.move_stack)
        # Call the parent class's method
        super().on_canvas_click(event)
        # After a human move, sync the environment and update displays
        if len(self.game.board.move_stack) > old_move_count:
            self.env.game.board = self.game.board.copy()
            last_move = self.game.board.move_stack[-1]
            self.log_move(last_move.uci(), is_ai=False)
            self.update_turn_indicator()

def main():
    root = tk.Tk()
    root.title("Chess AI Interface")
    game = ChessGame()
    gui = ChessGUIWithAI(root, game)
    root.mainloop()

if __name__ == "__main__":
    main()