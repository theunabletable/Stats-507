import tkinter as tk
from tkinter import filedialog  # Import filedialog to use it properly
from PIL import Image, ImageTk
import chess
import chess.svg
import cairosvg
import io
import time
from bs4 import BeautifulSoup
from ChessReplay import ChessReplay

#GUI for playing games. Saves a PGN file of the game after completion.
#Programmer: Andrew Jones and Claude AI
#November 2024

class ChessGUI:
    def __init__(self, root, chess_game):
        self.game = chess_game
        self.replay = ChessReplay()  # Add instance of ChessReplay
        self.root = root
        self.root.title("Chess Game with Visual Board")

        # Track the selected piece and its square
        self.selected_square = None
        self.promotion_square = None  # Track where the promotion is happening
        self.game_over = False  # Track if the game is over

        # Create a Canvas for Chess Board Image
        self.canvas = tk.Label(self.root)
        self.canvas.pack()

        # Bind mouse click event to the canvas
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Create a label to display game status messages
        self.status_label = tk.Label(self.root, text="", font=("Arial", 14))
        self.status_label.pack()

        # Add Replay Control Buttons
        self.load_button = tk.Button(self.root, text="Load Replay", command=self.load_replay)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.play_button = tk.Button(self.root, text="Play Replay", command=self.play_replay)
        self.play_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(self.root, text="Next Move", command=self.next_replay_move)
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.prev_button = tk.Button(self.root, text="Previous Move", command=self.prev_replay_move)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        # Initial Board Display
        self.update_board()



    def load_replay(self):
        """Open file dialog to select and load a replay"""
        filename = filedialog.askopenfilename(initialdir=r"C:\Users\Drew\Desktop\classes\Stats 507\Final Project\Chess Engine\replays", filetypes=[("PGN Files", "*.pgn")])
        if filename:
            self.replay.load_game(filename)
            self.update_board_with_replay()

    def play_replay(self):
        """Play the replay automatically"""
        def play():
            while self.replay.current_index < len(self.replay.moves):
                self.replay.next_move()
                self.update_board_with_replay()
                time.sleep(1)  # Adjust the speed of replay

        self.root.after(0, play)

    def next_replay_move(self):
        """Step forward in the replay"""
        if self.replay.next_move():
            self.update_board_with_replay()

    def prev_replay_move(self):
        """Step backward in the replay"""
        if self.replay.previous_move():
            self.update_board_with_replay()

    def update_board_with_replay(self):
        """Update the board with the current replay state"""
        svg_data = chess.svg.board(self.replay.get_current_board())
        png_data = cairosvg.svg2png(bytestring=svg_data)
        image = Image.open(io.BytesIO(png_data))
        image = ImageTk.PhotoImage(image)
        self.canvas.configure(image=image)
        self.canvas.image = image
        

    def on_canvas_click(self, event):
        # If the game is over, no moves should be processed
        if self.game_over:
            return

        if self.promotion_square is not None:
            # Handle promotion piece selection
            self.handle_promotion_selection(event)
            return

        # Get the dimensions of the current image on the canvas
        image_width = self.canvas.winfo_width()
        square_size = image_width // 8
        col = event.x // square_size
        row = event.y // square_size

        # Translate GUI row/column to chess notation (0-indexed to 1-indexed, and flip for board perspective)
        square = chess.square(col, 7 - row)

        # Handle selecting and moving pieces
        if self.selected_square is None:
            # Select a piece if it exists
            if self.game.get_board_state().piece_at(square):
                self.selected_square = square
                self.update_board()  # Update to show highlighted square
        else:
            # Attempt to make a move
            if self.is_pawn_promotion(self.selected_square, square):
                # Set promotion square and show promotion options
                self.promotion_square = square
                self.update_board()  # Update board to show promotion choices
            else:
                move = chess.Move(self.selected_square, square)
                if self.game.make_move(move):
                    print(f"Moved to {chess.square_name(square)}")
                else:
                    print("Illegal move")
                # Reset selection and update the board
                self.selected_square = None
                self.update_board()

                # Check if the game is over
                if self.game.is_game_over():
                    self.game_over = True
                    result = self.game.get_game_result()
                    self.status_label.config(text=f"Game Over: {result}")
                    self.output_game_result()
                    self.game.save_game()  # Save the game when it ends


    def handle_promotion_selection(self, event):
        # If the game is over, no moves should be processed
        if self.game_over:
            return

        # Get the dimensions of the current image on the canvas
        image_width = self.canvas.winfo_width()
        square_size = image_width // 8

        col = event.x // square_size
        row = event.y // square_size

        # Determine which promotion piece was selected based on the clicked square
        promotion_map = {
            2: chess.KNIGHT,  # c file -> knight
            3: chess.BISHOP,  # d file -> bishop
            4: chess.ROOK,    # e file -> rook
            5: chess.QUEEN    # f file -> queen
        }

        if row == 4 and col in promotion_map:
            promotion_piece = promotion_map[col]
            move = chess.Move(self.selected_square, self.promotion_square, promotion=promotion_piece)
            if self.game.make_move(move):
                print(f"Pawn promoted to {chess.piece_name(promotion_piece)}")
            else:
                print("Illegal promotion move")
            # Reset selection and promotion
            self.selected_square = None
            self.promotion_square = None
            self.update_board()

            # Check if the game is over after promotion
            if self.game.is_game_over():
                self.game_over = True
                result = self.game.get_game_result()
                self.status_label.config(text=f"Game Over: {result}")
                self.output_game_result()
                self.game.save_game()  # Save the game when it ends

    def update_board(self):
        # Generate the SVG image of the current board state
        svg_data = chess.svg.board(self.game.get_board_state())

        # Modify the SVG to add a circle highlight if needed
        if self.selected_square is not None:
            svg_data = self.add_circle_highlight(svg_data, self.selected_square)

        # If a pawn is being promoted, show promotion options
        if self.promotion_square is not None:
            svg_data = self.add_promotion_options(svg_data)

        # Convert SVG to PNG using cairosvg
        png_data = cairosvg.svg2png(bytestring=svg_data)

        # Convert the PNG bytes to a format Tkinter understands
        image = Image.open(io.BytesIO(png_data))
        image = ImageTk.PhotoImage(image)

        # Update the canvas with the new board image
        self.canvas.configure(image=image)
        self.canvas.image = image

    def is_pawn_promotion(self, from_square, to_square):
        """Check if the move is a pawn promotion."""
        piece = self.game.get_board_state().piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN:
            # Check if the pawn is moving to the last rank
            if (piece.color == chess.WHITE and chess.square_rank(to_square) == 7) or \
               (piece.color == chess.BLACK and chess.square_rank(to_square) == 0):
                return True
        return False

    def add_circle_highlight(self, svg_data, square):
        """
        Modify the SVG to add a circle highlight to the selected square.
        """
        # Parse the SVG data using BeautifulSoup
        soup = BeautifulSoup(svg_data, "xml")

        # Convert the square number to rank and file
        file, rank = chess.square_file(square), chess.square_rank(square)

        # Calculate SVG coordinates for the circle
        offset = 9
        cx = 22.5 + 45 * file + offset
        cy = 22.5 + 45 * (7 - rank) + offset

        # Create a new circle element to add to the SVG
        circle_tag = soup.new_tag(
            "circle",
            cx=str(cx),
            cy=str(cy),
            r="12",
            fill="#ffcc00",
            opacity="0.5"
        )

        # Add the circle to the root SVG tag
        svg_tag = soup.find("svg")
        if svg_tag:
            svg_tag.append(circle_tag)

        # Convert the modified SVG tree back to a string
        return str(soup)

    def add_promotion_options(self, svg_data):
        """
        Modify the SVG to add labels of promotion options on squares c5, d5, e5, f5.
        """
        # Parse the SVG data using BeautifulSoup
        soup = BeautifulSoup(svg_data, "xml")
        svg_tag = soup.find("svg")

        if not svg_tag:
            return str(soup)

        # Coordinates for the promotion options on row 4 (i.e., visually rank 5)
        options = [
            (2, "N"),  # c file -> knight
            (3, "B"),  # d file -> bishop
            (4, "R"),  # e file -> rook
            (5, "Q")   # f file -> queen
        ]

        for file, label in options:
            # Calculate SVG coordinates for the label
            x = 45 * file
            y = 45 * 4  # row 4 (0-indexed) represents rank 5 visually

            # Add the text label to the SVG to indicate the promotion option
            text_tag = soup.new_tag(
                "text",
                x=str(x + 15),  # Center the label in the square
                y=str(y + 30),  # Adjust to place text vertically centered in the square
                fill="black",
                style="font-size:20px;font-family:sans-serif;font-weight:bold;"
            )
            text_tag.string = label
            svg_tag.append(text_tag)

        # Convert the modified SVG tree back to a string
        return str(soup)
    
    def output_game_result(self):
        """Outputs the result and the movelist of the game"""
        # Output the result
        result = self.game.get_game_result()
        print(f"Game Result: {result}")

        # Output the move list in pairs
        move_list = self.game.get_move_list()
        print("Moves Played:")
        for i in range(0, len(move_list), 2):
            if i + 1 < len(move_list):
                # Output both White and Black moves on the same line
                print(f"{i // 2 + 1}. {move_list[i]} {move_list[i + 1]}")
            else:
                # Output only the White move if there's no corresponding Black move (odd number of moves)
                print(f"{i // 2 + 1}. {move_list[i]}")