import torch
import torch.nn as nn
import torch.nn.functional as F
#Neural network implementation for Chess AI
#Programmer: Andrew Jones and Claude
#November 2024

"""
policy_network is a deep convolutional neural network that takes a chess position
and outputs both a policy (probabilities for each possible move) and a value
(evaluation of the position).
Input layer processes 20 8x8 planes representing board state
Residual tower with repeated conv blocks builds chess understanding
Splits into policy head and value head.
"""
class ConvBlock(nn.Module):
    """
    Basic building block using convolution, batch norm, and ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
def get_device():
        """Get the best available device (CUDA GPU if available, else CPU)"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device

class PolicyNetwork(nn.Module):
    def __init__(self, action_size=1968):  # 1968 is typical size from ChessActionSpace
        super().__init__()
        self.device = get_device()    
        # Enable automatic mixed precision
        self.mixed_precision = True    
        # Input layer: 20 planes of 8x8

        self.input_conv = ConvBlock(20, 256)
        
        # Residual tower
        self.residual_tower = nn.Sequential(
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256)
        )
        
        # Policy head
        self.policy_conv = ConvBlock(256, 256)
        self.policy_fc = nn.Linear(256 * 8 * 8, action_size)
        
        # Value head
        self.value_conv = ConvBlock(256, 1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.to(self.device)  # Move model to GPU if available
 
    
    def forward(self, x):
        """
        Process a batch of board positions.
        Args:
          x: batch of 20x8x8 board tensors
        
        returns:
          tuple of: policy_logits (raw move probabilities before softmax) 
          and value (position evaluation between -1 and 1).
        """
        with torch.amp.autocast('cuda', enabled=self.mixed_precision):
            # Common layers
            x = self.input_conv(x)
            x = self.residual_tower(x)
            
            # Policy head
            policy = self.policy_conv(x)
            policy = policy.view(policy.size(0), -1)  # Flatten
            policy_logits = self.policy_fc(policy)
            
            # Value head
            value = self.value_conv(x)
            value = value.view(value.size(0), -1)  # Flatten
            value = F.relu(self.value_fc1(value))
            value = torch.tanh(self.value_fc2(value))
            
            return policy_logits, value

def create_simple_network():
    """Create a policy network with default parameters"""
    return PolicyNetwork()
