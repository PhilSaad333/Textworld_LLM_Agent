from dataclasses import dataclass, field
import torch

@dataclass
class RLConfig:
    # Environment parameters
    max_steps: int = 50
    scale: int = 10  # Changed to int to match TaskConfig expectation
    
    # Model parameters
    learning_rate: float = 5e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    max_output_length: int = 128
    max_input_length: int = 512
    num_epochs: int = 3
    max_grad_norm: float = 1.0
    
    # Logging and checkpointing
    log_steps: int = 10
    save_steps: int = 100
    checkpoint_dir: str = "./checkpoints/rl"
    
    # Optimizer selection
    optimizer_type: str = "custom"  # 'custom' or 'huggingface'
    
    # GRPO specific parameters
    num_samples: int = 8  # G in the writeup (number of completions per prompt)
    num_generations: int = 4  # Legacy parameter for compatibility
    epsilon: float = 0.2  # PPO clipping parameter
    beta: float = 0.01  # KL penalty coefficient
    
    # Reward parameters
    gamma: float = 0.99  # Discount factor
    format_reward: float = 0.5  # Reward for correct format
    format_penalty: float = -1.0  # Penalty for incorrect format
    room_reward: float = 0.5  # Reward for correct room prediction
    room_penalty: float = -0.5  # Penalty for incorrect room prediction
    
    # Training parameters
    num_iterations: int = 3  # Number of training iterations
    num_episodes_per_iteration: int = 5  # Number of episodes to collect per iteration
    
    # Data collection parameters
    difficulties: list = None  # List of difficulties to collect data from
    episodes_per_difficulty: int = 5  # Number of episodes to collect per difficulty
    
    # Agent parameters
    temperature: float = 0.7
    top_p: float = 0.9
    use_map: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Initialize default values for lists"""
        if self.difficulties is None:
            self.difficulties = [1, 5, 10]  # Default difficulties
