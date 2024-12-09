import chess
import torch
from ChessActionSpace import ChessActionSpace
from collections import defaultdict
from typing import Dict, List, Tuple
from ChessGame import ChessGame
from ChessEnvironment import ChessEnvironment
from policy_network import PolicyNetwork
from ChessAgent import MCTSAgent
from mcts import MCTS, MCTSNode
import numpy as np
import time
import chess
import time
from collections import defaultdict
import numpy as np
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
from io import StringIO
import cProfile
import pstats

class MCTSOptimizer:
    def __init__(self, base_mcts_class):
        self.base_mcts = base_mcts_class
        self.variants = {}
        self.results = defaultdict(list)
        
    def add_variant(self, name, optimization_params):
        """Add a variant of MCTS with specific optimizations"""
        variant = deepcopy(self.base_mcts)
        
        # Apply optimization parameters
        for param, value in optimization_params.items():
            setattr(variant, param, value)
            
        self.variants[name] = variant
        
    def benchmark_position(self, fen_position, num_runs=3):
        """Benchmark all variants on a specific position"""
        board = chess.Board(fen_position)
        
        # Create game state
        from ChessGame import ChessGame
        from ChessEnvironment import ChessEnvironment
        
        game = ChessGame()
        game.board = board.copy()
        env = ChessEnvironment()
        env.game = game
        
        metrics = ['time', 'depth', 'nodes', 'nodes_per_second', 'avg_branching']
        results = {name: {metric: [] for metric in metrics} for name in self.variants}
        
        for variant_name, mcts in self.variants.items():
            print(f"\nTesting {variant_name}:")
            for run in range(num_runs):
                torch.cuda.empty_cache()  # Clear GPU memory
                
                # Time the search
                start_time = time.perf_counter()
                probs, root = mcts.run_search(env)
                end_time = time.perf_counter()
                
                # Collect metrics
                search_time = end_time - start_time
                max_depth = self.get_max_depth(root)
                total_nodes = self.count_nodes(root)
                nodes_per_second = total_nodes / search_time
                avg_branching = self.avg_branching_factor(root)
                
                # Store results
                results[variant_name]['time'].append(search_time)
                results[variant_name]['depth'].append(max_depth)
                results[variant_name]['nodes'].append(total_nodes)
                results[variant_name]['nodes_per_second'].append(nodes_per_second)
                results[variant_name]['avg_branching'].append(avg_branching)
                
                print(f"Run {run + 1}: Depth={max_depth}, "
                      f"Nodes={total_nodes}, Time={search_time:.2f}s, "
                      f"Nodes/s={nodes_per_second:.0f}")
                
        return results
    
    @staticmethod
    def get_max_depth(node):
        max_depth = 0
        stack = [(node, 0)]
        while stack:
            current, depth = stack.pop()
            if not current.children:
                max_depth = max(max_depth, depth)
            else:
                stack.extend((child, depth + 1) for child in current.children.values())
        return max_depth
    
    @staticmethod
    def count_nodes(node):
        count = 1
        stack = [node]
        while stack:
            current = stack.pop()
            stack.extend(current.children.values())
            count += len(current.children)
        return count
    
    @staticmethod
    def avg_branching_factor(node):
        if not node.children:
            return 0
        total_branches = 0
        total_nodes = 0
        stack = [node]
        while stack:
            current = stack.pop()
            if current.children:
                total_branches += len(current.children)
                total_nodes += 1
                stack.extend(current.children.values())
        return total_branches / max(1, total_nodes)
    
    def plot_results(self, results):
        """Plot comparison of variants"""
        metrics = ['time', 'depth', 'nodes', 'nodes_per_second']
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MCTS Optimization Comparison')
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            data = [results[variant][metric] for variant in self.variants]
            ax.boxplot(data, labels=list(self.variants.keys()))
            ax.set_title(metric.replace('_', ' ').title())
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('mcts_optimization_comparison.png')
        plt.close()

def test_optimizations():
    """Test different MCTS optimizations"""
    from policy_network import PolicyNetwork
    from ChessActionSpace import ChessActionSpace
    from mcts import MCTS
    
    # Create base instances
    policy_net = PolicyNetwork()
    action_space = ChessActionSpace()
    base_mcts = MCTS(policy_net, action_space, num_simulations=800)
    
    # Create optimizer
    optimizer = MCTSOptimizer(base_mcts)
    
    # Add variants to test
    variants = {
        'Baseline': {
            'num_simulations': 800,
            'batch_size': 32,
            'c_puct': 1.0
        },
        'Larger Batch': {
            'num_simulations': 800,
            'batch_size': 64,
            'c_puct': 1.0
        },
        'More Exploration': {
            'num_simulations': 800,
            'batch_size': 32,
            'c_puct': 2.0
        },
        'Fast Search': {
            'num_simulations': 400,
            'batch_size': 64,
            'c_puct': 1.5
        }
    }
    
    for name, params in variants.items():
        optimizer.add_variant(name, params)
    
    # Test positions
    test_positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"  # Complex midgame
    ]
    
    all_results = {}
    for i, fen in enumerate(test_positions):
        print(f"\nTesting Position {i+1}")
        results = optimizer.benchmark_position(fen, num_runs=3)
        all_results[fen] = results
        optimizer.plot_results(results)
    
    return all_results





class MCTSAnalyzer:
    def __init__(self, mcts_instance):
        self.mcts = mcts_instance
        self.metrics = defaultdict(list)
        self.device = next(mcts_instance.policy_network.parameters()).device
        
    def analyze_position(self, fen_position, num_runs=5):
        """Analyze MCTS performance on a specific position"""
        board = chess.Board(fen_position)
        self.metrics.clear()
        
        for run in range(num_runs):
            # Create game state from position
            from ChessGame import ChessGame
            from ChessEnvironment import ChessEnvironment
            
            game = ChessGame()
            game.board = board.copy()
            env = ChessEnvironment()
            env.game = game
            
            # Time the search
            start_time = time.perf_counter()
            try:
                probs, root = self.mcts.run_search(env)
                search_time = time.perf_counter() - start_time
                
                # Collect metrics
                self.metrics['search_time'].append(search_time)
                self.metrics['max_depth'].append(self.get_max_depth(root))
                self.metrics['num_nodes'].append(self.count_nodes(root))
                self.metrics['branching'].append(self.avg_branching_factor(root))
                self.metrics['value_spread'].append(self.value_distribution(root))
                
                print(f"Run {run + 1}: Depth={self.metrics['max_depth'][-1]}, "
                      f"Nodes={self.metrics['num_nodes'][-1]}, "
                      f"Time={search_time:.2f}s")
                      
            except Exception as e:
                print(f"Error in run {run + 1}: {str(e)}")
                continue
            
        return self.summarize_metrics()
    
    def get_max_depth(self, node):
        """Get maximum depth of the tree"""
        max_depth = 0
        stack = [(node, 0)]
        
        while stack:
            current, depth = stack.pop()
            if not current.children:
                max_depth = max(max_depth, depth)
            else:
                stack.extend((child, depth + 1) for child in current.children.values())
        
        return max_depth
    
    def count_nodes(self, node):
        """Count total nodes in tree"""
        count = 1
        stack = [node]
        
        while stack:
            current = stack.pop()
            stack.extend(current.children.values())
            count += len(current.children)
            
        return count
    
    def avg_branching_factor(self, node):
        """Calculate average branching factor"""
        if not node.children:
            return 0
        
        total_branches = 0
        total_nodes = 0
        stack = [node]
        
        while stack:
            current = stack.pop()
            if current.children:
                total_branches += len(current.children)
                total_nodes += 1
                stack.extend(current.children.values())
                
        return total_branches / max(1, total_nodes)
    
    def value_distribution(self, node):
        """Analyze value distribution across nodes"""
        values = []
        stack = [node]
        
        while stack:
            current = stack.pop()
            if current.visit_count > 0:
                values.append(current.value_sum / current.visit_count)
            stack.extend(current.children.values())
            
        return np.std(values) if values else 0
    
    def summarize_metrics(self):
        """Summarize collected metrics"""
        summary = {}
        for metric, values in self.metrics.items():
            if values:  # Only process metrics that have values
                summary[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)) if len(values) > 1 else 0.0,
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        return summary

def test_positions():
    """Test positions known to require deep search"""
    test_positions = [
        # Starting position
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        # Position requiring deep tactics
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        # Position with clear material advantage
        "rnbqkbnr/ppp2ppp/8/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 3"
    ]
    
    # Create all necessary instances
    from policy_network import PolicyNetwork
    from ChessActionSpace import ChessActionSpace
    from mcts import MCTS
    
    policy_net = PolicyNetwork()
    action_space = ChessActionSpace()
    mcts = MCTS(policy_net, action_space, num_simulations=800)
    
    analyzer = MCTSAnalyzer(mcts)
    
    results = {}
    for i, fen in enumerate(test_positions):
        print(f"\nAnalyzing position {i+1}...")
        results[fen] = analyzer.analyze_position(fen)
        
    return results

class MCTSProfiler:
    def __init__(self, mcts_instance):
        self.mcts = mcts_instance
        
    def profile_search(self, fen_position):
        """Profile a single MCTS search"""
        # Setup board
        board = chess.Board(fen_position)
        from ChessGame import ChessGame
        from ChessEnvironment import ChessEnvironment
        game = ChessGame()
        game.board = board.copy()
        env = ChessEnvironment()
        env.game = game
        
        # Profile the search
        profiler = cProfile.Profile()
        profiler.enable()
        probs, root = self.mcts.run_search(env)
        profiler.disable()
        
        # Sort and format results
        s = StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.sort_stats('cumulative').print_stats(20)  # Top 20 time-consuming functions
        
        # Also collect GPU memory stats if available
        gpu_stats = {}
        if torch.cuda.is_available():
            gpu_stats['max_memory'] = torch.cuda.max_memory_allocated() / 1024**2
            gpu_stats['current_memory'] = torch.cuda.memory_allocated() / 1024**2
            
        return s.getvalue(), gpu_stats

def run_profiling():
    """Run detailed profiling of MCTS search"""
    from policy_network import PolicyNetwork
    from ChessActionSpace import ChessActionSpace
    from mcts import MCTS
    
    # Create instances
    policy_net = PolicyNetwork()
    action_space = ChessActionSpace()
    mcts = MCTS(policy_net, action_space, num_simulations=800)
    
    profiler = MCTSProfiler(mcts)
    
    # Profile starting position
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print("Profiling MCTS search...")
    profile_output, gpu_stats = profiler.profile_search(fen)
    
    print("\nProfile Results:")
    print("-" * 50)
    print(profile_output)
    
    if gpu_stats:
        print("\nGPU Memory Usage:")
        print(f"Peak memory: {gpu_stats['max_memory']:.1f} MB")
        print(f"Current memory: {gpu_stats['current_memory']:.1f} MB")
        



def test_tensor_updates():
    """Granular testing of tensor update performance"""
    import time
    import chess
    import torch
    from collections import defaultdict
    from mcts import MCTSNode
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MCTSNode._working_tensor = torch.zeros((20, 8, 8), device=device)
    board = chess.Board()
    timings = defaultdict(float)
    counts = defaultdict(int)
    
    def time_operation(name, fn):
        start = time.perf_counter()
        result = fn()
        end = time.perf_counter()
        timings[name] += end - start
        counts[name] += 1
        return result

    def create_ep_set_timer(move):
        def time_ep_set():
            ep_square = (move.to_square + move.from_square) // 2
            MCTSNode._working_tensor[17, ep_square // 8, ep_square % 8] = 1
        return time_ep_set

    def create_castling_timer(move, piece):
        def time_castling():
            rook_idx = 4 + (0 if piece.color else 1)
            from_rank = move.from_square // 8
            if move.to_square > move.from_square:  # Kingside
                MCTSNode._working_tensor[rook_idx, from_rank, 7] = 0
                MCTSNode._working_tensor[rook_idx, from_rank, 5] = 1
            else:  # Queenside
                MCTSNode._working_tensor[rook_idx, from_rank, 0] = 0
                MCTSNode._working_tensor[rook_idx, from_rank, 3] = 1
        return time_castling
    
    for move in board.legal_moves:
        from_rank, from_file = move.from_square // 8, move.from_square % 8
        to_rank, to_file = move.to_square // 8, move.to_square % 8
        
        # Initial move analysis
        time_operation('move_setup', lambda: (
            move.from_square // 8,
            move.from_square % 8,
            move.to_square // 8,
            move.to_square % 8
        ))
        
        # Piece lookup and index calculation
        piece = time_operation('piece_access', lambda: board.piece_at(move.from_square))
        time_operation('piece_idx_calc', lambda: (piece.piece_type - 1) * 2 + (0 if piece.color else 1))
        piece_idx = (piece.piece_type - 1) * 2 + (0 if piece.color else 1)
        
        # Core move operations
        time_operation('tensor_initial_access', lambda: MCTSNode._working_tensor[piece_idx])
        time_operation('source_clear', lambda: MCTSNode._working_tensor[piece_idx, from_rank, from_file].fill_(0))
        
        # Capture handling
        captured = time_operation('capture_lookup', lambda: board.piece_at(move.to_square))
        if captured:
            cap_idx = (captured.piece_type - 1) * 2 + (0 if captured.color else 1)
            time_operation('capture_clear', lambda: MCTSNode._working_tensor[cap_idx, to_rank, to_file].fill_(0))
            
        # Target square update
        time_operation('target_set', lambda: MCTSNode._working_tensor[piece_idx, to_rank, to_file].fill_(1))
        
        # Special move timing
        if piece.piece_type == chess.KING and abs(move.to_square - move.from_square) == 2:
            time_operation('castling_calc', lambda: 4 + (0 if piece.color else 1))
            time_operation('castling_move', create_castling_timer(move, piece))
            time_operation('castling_rights', lambda: 
                MCTSNode._working_tensor[13:15 if piece.color else 15:17].zero_())
                
        elif piece.piece_type == chess.ROOK:
            def time_rook_rights():
                rights_idx = (14 if move.from_square == chess.A1 else 
                            13 if move.from_square == chess.H1 else
                            16 if move.from_square == chess.A8 else 
                            15 if move.from_square == chess.H8 else -1)
                if rights_idx != -1:
                    MCTSNode._working_tensor[rights_idx].fill_(0)
            time_operation('rook_rights', time_rook_rights)
        
        # En passant timing
        time_operation('ep_clear', lambda: MCTSNode._working_tensor[17].zero_())
        
        if piece.piece_type == chess.PAWN and abs(move.to_square - move.from_square) == 16:
            time_operation('ep_set', create_ep_set_timer(move))
        
        # Turn update timing
        time_operation('turn_cache', lambda: not MCTSNode._cached_turn)
        time_operation('turn_tensor', lambda: MCTSNode._working_tensor[12].fill_(float(not MCTSNode._cached_turn)))
        
        # Move counter timing
        if piece.piece_type == chess.PAWN or captured:
            time_operation('counter_reset', lambda: MCTSNode._working_tensor[18].zero_())
        else:
            def time_counter_update():
                new_clock = min(board.halfmove_clock + 1, 63)
                MCTSNode._working_tensor[18].zero_()
                MCTSNode._working_tensor[18, new_clock // 8, new_clock % 8].fill_(1)
            time_operation('counter_update', time_counter_update)
        
        # Device synchronization check
        time_operation('device_sync', lambda: torch.cuda.synchronize() if torch.cuda.is_available() else None)
        
        # Full move for comparison
        time_operation('full_move', lambda: MCTSNode.apply_move_to_working_tensor(move, board))
    
    # Print results sorted by time
    print("\nTiming Results:")
    print("-" * 60)
    print(f"{'Operation':<25} {'Time (ms)':<10} {'Count':<10} {'Avg (μs)':<10}")
    print("-" * 60)
    
    for op in sorted(timings.keys(), key=lambda x: timings[x], reverse=True):
        total_ms = timings[op] * 1000
        count = counts[op]
        avg_us = (total_ms * 1000) / count if count > 0 else 0
        print(f"{op:<25} {total_ms:>10.2f} {count:>10} {avg_us:>10.2f}")
        
    return timings, counts

if __name__ == "__main__":
    run_profiling()


