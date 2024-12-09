Python implementation of Chess AI, using Monte Carlo Tree Search
and reinforcement learning

Files:
main.py (runs training loop)
policy_network.py (neural network architecture)

mcts.py (implements the MCTS algorithm)
MCTSAnalyzer.py (tools for testing the performance of the MCTS algorithm)

ChessGame.py (chess game logic)
ChessActionSpace.py (encodes moves from ChessGame into number idices)
ChessEnvironment.py (Environment wrapper, contains reward function, provides
                     interface for the agent to make moves)
ChessAgent.py (implements different agents, such as MCTS agent, 
               or random agent).