import tkinter as tk
from ChessGame import ChessGame
from ChessGUI import ChessGUI
from ChessReplay import ChessReplay

if __name__ == "__main__":
    root = tk.Tk()
    chess_game = ChessGame()  # Create an instance of the game logic
    gui = ChessGUI(root, chess_game)  # Create an instance of the GUI with the game logic
    root.mainloop()