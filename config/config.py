from dataclasses import dataclass
from enum import Enum
import os
import subprocess
import textworld
import torch


class RewardType(Enum):
    DENSE = "dense"
    BALANCED = "balanced"
    SPARSE = "sparse"

class GoalType(Enum):
    DETAILED = "detailed"
    BRIEF = "brief"
    NONE = "none"

class GameType(Enum):
    STANDARD = "standard"  # Current tw-simple games
    TREASURE = "treasure"  # New type
    SIMPLE = "basic"        # Basic tw-simple
    COINS = "coins"


@dataclass
class EvalConfig:
    """Configuration for model evaluation and testing"""
    
    # Generation parameters
    num_beams: int = 1                # Number of beams for beam search
    num_return_sequences: int = 1     # Number of sequences to return
    do_sample: bool = False           # Whether to use sampling
    temperature: float = 0.7          # Temperature for sampling
    top_p: float = 0.9                # Top-p for nucleus sampling
    top_k: int = 50                   # Top-k for top-k sampling
    
    # Evaluation behavior
    print_completions: bool = False   # Whether to print all completions
    print_format_check: bool = False  # Whether to print format check results
    print_action_selection: bool = False  # Whether to print action selection details
    
    # Action selection strategy
    action_selection: str = "best_beam"  # Options: "best_beam", "best_format", "sample"
    
    # Logging
    log_to_file: bool = False         # Whether to log to a file
    log_path: str = None              # Path to log file
    
    # Evaluation metrics
    track_success_rate: bool = True   # Whether to track success rate
    track_steps_to_completion: bool = True  # Whether to track steps to completion
    track_format_correctness: bool = True   # Whether to track format correctness
    
    def __post_init__(self):
        """Initialize log path if logging is enabled"""
        if self.log_to_file and self.log_path is None:
            # Create a default log directory if none provided
            os.makedirs("logs/eval", exist_ok=True)
            self.log_path = "logs/eval/evaluation_log.txt"
            
    def get_beam_search_params(self):
        """Get parameters for beam search generation"""
        return {
            "num_beams": self.num_beams,
            "num_return_sequences": min(self.num_return_sequences, self.num_beams),
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k if self.do_sample else None
        } 



@dataclass
class ModelConfig:
    # Pretrained model settings
    model_name: str = "google/flan-t5-large" #"FLAN-T5-BASE" #"TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    freeze_obs_base: bool = True
    unfreeze_last_n_obs_layers: int = 2

    # Network architecture
    obs_summary_dim: int = 256  # Dimension for observation embeddings
    cmd_summary_dim: int = 256  # Dimension for command embeddings
    room_dim: int = 256  # Dimension for room embeddings
    num_cs_heads: int = 4  # Number of heads in command scoring attention
    num_cs_layers: int = 2  # Number of layers in command scoring
    dropout: float = 0.1

    # Training/optimization
    learning_rate: float = 8e-5  # Slightly reduced for larger model
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Text processing
    max_sequence_length: int = 512
    max_history_length: int = 50  # Maximum number of history items to keep



@dataclass
class GameConfig:
    reward_type: RewardType
    goal_type: GoalType
    game_type: GameType = GameType.STANDARD
    max_history_actions: int = 3
    seed: int = 18
    output_dir: str = "."
    test: bool = True
    silent: bool = True
    force: bool = True
    treasure_level: int = 3  # from 1-30
    coin_level: int = 3  # from 1-30

    @property
    def output_name(self) -> str:
        game_types = {
            GameType.STANDARD: "standard",
            GameType.TREASURE: "treasure",
            GameType.SIMPLE: "basic",
            GameType.COINS: "coins"
        }
        if self.game_type == GameType.TREASURE:
            return f"tw-{game_types[self.game_type]}_level_{self.treasure_level}_seed_{self.seed}.z8"
        elif self.game_type == GameType.COINS:
            return f"tw-{game_types[self.game_type]}_level_{self.coin_level}_seed_{self.seed}.z8"
        else:
            return f"tw-{game_types[self.game_type]}_{self.seed}.z8"

    @property
    def output_path(self) -> str:
        return os.path.join(self.output_dir, self.output_name)





@dataclass
class SFTConfig:
    # Model settings
    model_name: str = "google/flan-t5-large"      #"FLAN-T5-BASE" #"TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Layer freezing settings
    freeze_layers: bool = True  # Whether to freeze most layers
    unfreeze_last_n_layers: int = 4  # Number of layers to unfreeze from the end
    
    # Training hyperparameters
    learning_rate: float = 1e-5  # Reduced from 3e-5
    batch_size: int = 4  # Reduced from 8
    num_epochs: int = 1  # We'll run multiple epochs manually with tag checking
    warmup_steps: int = 200  # Increased from 100
    weight_decay: float = 0.01
    max_grad_norm: float = 0.5  # Reduced from 1.0
    
    # Sequence lengths
    max_input_length: int = 512
    max_output_length: int = 128
    
    # Optimizer settings
    optimizer_type: str = "adamw"
    scheduler_type: str = "linear"  # linear warmup with decay
    
    # Training features
    gradient_accumulation_steps: int = 8  # Increased from 4
    mixed_precision: bool = False  # Disabled to prevent numerical instability
    
    # Validation
    validation_split: float = 0.1  # 10% of data for validation
    eval_steps: int = 100  # Evaluate every N steps
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_steps: int = 500  # Save model every N steps
    keep_checkpoint_max: int = 3  # Maximum number of checkpoints to keep
    
    # Logging
    use_wandb: bool = False  # Whether to use Weights & Biases logging
    wandb_project: str = "textworld-sft"
    wandb_entity: str = None
    log_steps: int = 10  # Log metrics every N steps
    
    # Data processing
    num_workers: int = 4  # Number of workers for data loading
    pin_memory: bool = True  # Pin memory for faster data transfer to GPU
    
    # Tag checking
    check_tags: bool = True  # Whether to check for command and room tags
    tag_check_samples: int = 20  # Number of samples to check for tags
    
    def __post_init__(self):
        """Validate and process config after initialization"""
        # Create checkpoint directory if it doesn't exist
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
        # Adjust batch size and gradient accumulation for small GPUs
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            if gpu_mem < 8:  # Less than 8GB GPU
                self.batch_size = min(4, self.batch_size)
                self.gradient_accumulation_steps = max(8, self.gradient_accumulation_steps)
                self.mixed_precision = True
                
        # Validate device setting
        if self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
            
        # Adjust workers for CPU
        if self.device == "cpu":
            self.num_workers = min(2, self.num_workers)
            self.pin_memory = False

@dataclass
class TextWorldConfig:
    def __init__(self,
                 game_config: GameConfig,
                 model_config: ModelConfig,
                 sft_config: SFTConfig = None,
                 eval_config = None):  # Add EvalConfig as optional parameter
        self.game_config = game_config
        self.model_config = model_config
        self.sft_config = sft_config or SFTConfig()  # Use default if not provided
        
        # Use default EvalConfig if not provided
        self.eval_config = eval_config or EvalConfig()  
        self.excluded_actions = ["look", "inventory"]

        self.requested_infos = textworld.EnvInfos(
            description=True,
            inventory=True,
            command_templates=False,
            admissible_commands=True,
            entities=True,
            verbs=True,
            location=True,
            max_score=True
        )

        os.makedirs(self.game_config.output_dir, exist_ok=True)
        
        # Set game_file and env_id based on output path
        self.game_file = self.game_config.output_path
        self.env_id = f"tw-v3"  # Standard env_id for TextWorld v3

        self.max_steps: int = 10  # Maximum steps per episode

    def create_game(self) -> bool:
        """Create game file if it doesn't exist"""
        if os.path.exists(self.game_file):
            return True
        
        try:
            if self.game_config.game_type == GameType.COINS:
                # Import coin collector specific functions
                from textworld.challenges.coin_collector import make_game_from_level
                
                # Create coin collector game
                options = textworld.GameOptions()
                options.seeds = self.game_config.seed
                game = make_game_from_level(self.game_config.coin_level, options)
                
            else:
                # Create standard/treasure game as before
                options = {
                    "reward_type": self.game_config.reward_type.value,
                    "goal_type": self.game_config.goal_type.value,
                    "test": self.game_config.test,
                    "silent": self.game_config.silent,
                    "force": self.game_config.force,
                }
                
                if self.game_config.game_type == GameType.TREASURE:
                    options["level"] = self.game_config.treasure_level
                    
                game = textworld.make(options=options)
                
            # Compile and save the game
            textworld.generator.compile_game(game, self.game_file)
            print(f"Game created successfully: {self.game_file}, env_id: {self.env_id}")
            return True
            
        except Exception as e:
            print(f"Failed to create game: {str(e)}")
            return False




def get_game_config(
    reward_type: RewardType,
    goal_type: GoalType,
    max_history_actions: int = 3,  # Add parameter with default
    eval_config = None  # Add EvalConfig as optional parameter
) -> TextWorldConfig:
    game_config = GameConfig(
        reward_type=reward_type,
        goal_type=goal_type,
        game_type=GameType.TREASURE,
        max_history_actions=max_history_actions  # Pass through to GameConfig
    )
    
    model_config = ModelConfig()
    
    # Import here to avoid circular imports if needed
    if eval_config is not None and not isinstance(eval_config, object):
        # No need to import EvalConfig since it's defined in this file
        eval_config = EvalConfig(**eval_config) if isinstance(eval_config, dict) else eval_config
    
    return TextWorldConfig(
        game_config=game_config,
        model_config=model_config,
        eval_config=eval_config
    )
def create_all_games():
    """Create all game variants with different reward and goal types"""
    configs = []

    for reward_type in RewardType:
        for goal_type in GoalType:
            if reward_type == RewardType.SPARSE and goal_type == GoalType.NONE:
                continue

            config = get_game_config(
                reward_type=reward_type,
                goal_type=goal_type,
                max_history_actions=3  # Add default value here too
            )
            if config.create_game():
                configs.append(config)
            else:
                print(f"Failed to create game with rewards={reward_type.value}, goal={goal_type.value}")

    return configs