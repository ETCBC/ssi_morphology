"""Global configuration.

These configuration settings are related to hardware; they should not affect the network.

device   Used by torch to select GPU/CPU.
"""
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
