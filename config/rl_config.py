from dataclasses import dataclass

@dataclass
class RLConfig:
    # Environment parameters
    max_steps: int = 50
    scale: int = 10  # Changed to int to match TaskConfig expectation
    
    # Model parameters
    learning_rate: float = 1e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_output_length: int = 128  # Used as max_completion_length in GRPOConfig
    max_input_length: int = 512  # Used as max_prompt_length in GRPOConfig
    num_epochs: int = 3
    
    # Logging and checkpointing
    log_steps: int = 10
    save_steps: int = 100
    checkpoint_dir: str = "./checkpoints/rl"
    
    # GRPO specific parameters
    beta: float = 0.1  # KL penalty coefficient
    # Removed target_kl and kl_penalty as they're not used by GRPOConfig
    use_vllm: bool = False  # Use vLLM for faster generation
    
    # Reward parameters
    format_success_reward: float = 0.5  # Reward for correct format
    format_failure_penalty: float = -0.2  # Penalty for incorrect format
    episode_completion_reward: float = 2.0  # Reward for completing an episode
    
    # Data collection parameters
    difficulties: list = None  # List of difficulties to collect data from
    episodes_per_difficulty: int = 5  # Number of episodes to collect per difficulty
    
    # Agent parameters
    temperature: float = 0.7  # Used by GRPOConfig
    top_p: float = 0.9  # Not directly used by GRPOConfig but might be useful elsewhere
    use_map: bool = True
    
    def __post_init__(self):
        """Initialize default values for lists"""
        if self.difficulties is None:
            self.difficulties = [1, 5, 10]  # Default difficulties
