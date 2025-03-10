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
class ModelConfig:
    # Pretrained model settings
    model_name: str = "FLAN-T5-BASE" #"TinyLlama/TinyLlama-1.1B-Chat-v1.0"
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
    learning_rate: float = 1e-4
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
    model_name: str = "FLAN-T5-BASE"      #"TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Layer freezing settings
    freeze_layers: bool = True  # Whether to freeze most layers
    unfreeze_last_n_layers: int = 2  # Number of layers to unfreeze from the end
    
    # Training hyperparameters
    learning_rate: float = 5e-5  # Slightly higher learning rate for TinyLlama
    batch_size: int = 16  # More conservative batch size
    num_epochs: int = 1  # We'll run multiple epochs manually with tag checking
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Sequence lengths
    max_input_length: int = 512
    max_output_length: int = 128
    
    # Optimizer settings
    optimizer_type: str = "adamw"
    scheduler_type: str = "linear"  # linear warmup with decay
    
    # Training features
    gradient_accumulation_steps: int = 2  # Use gradient accumulation for effective larger batch
    mixed_precision: bool = False  # Disabled for simplicity
    
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
                 sft_config: SFTConfig = None):  # Add SFTConfig as optional parameter
        self.game_config = game_config
        self.model_config = model_config
        self.sft_config = sft_config or SFTConfig()  # Use default if not provided

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
    max_history_actions: int = 3  # Add parameter with default
) -> TextWorldConfig:
    game_config = GameConfig(
        reward_type=reward_type,
        goal_type=goal_type,
        game_type=GameType.TREASURE,
        max_history_actions=max_history_actions  # Pass through to GameConfig
    )
    
    model_config = ModelConfig()
    #training_config = TrainingConfig()
    
    return TextWorldConfig(
        game_config=game_config,
        model_config=model_config
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