import os
import sys
import torch
import numpy as np
import random
from datetime import datetime

# Add the project root to the Python path to import your modules
if not '/content/Textworld_LLM_Agent' in sys.path:
    sys.path.append('/content/Textworld_LLM_Agent')

# Import necessary modules
from config.config import TextWorldConfig, ModelConfig, GameConfig, RewardType, GoalType, GameType, get_game_config
from config.rl_config import RLConfig
from training.trainer import TextWorldRLTrainer
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# Define paths
fine_tuned_model_path = '/content/drive/MyDrive/textworld_rl_models/flan_t5_large_finetuned/best_model.pt'
gameplay_data_dir = '/content/drive/MyDrive/textworld_rl_data'
grpo_model_save_dir = '/content/drive/MyDrive/textworld_rl_models/grpo_trained_model'

# Create directories if they don't exist
os.makedirs(gameplay_data_dir, exist_ok=True)
os.makedirs(grpo_model_save_dir, exist_ok=True)

# Generate a timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
gameplay_data_path = os.path.join(gameplay_data_dir, f"gameplay_data_{timestamp}.json")

# Create the main config using get_game_config
config = get_game_config(
    reward_type=RewardType.DENSE,
    goal_type=GoalType.DETAILED,
    max_history_actions=3
)

# Create RLConfig with optimized hyperparameters
rl_config = RLConfig(
    # Environment parameters
    max_steps=20,
    
    # Model parameters
    learning_rate=1e-5,  # Slightly lower than fine-tuning
    batch_size=4,  # Reduced batch size
    gradient_accumulation_steps=2,
    max_output_length=128,
    max_input_length=512,
    num_epochs=3,
    max_grad_norm=0.5,  # More aggressive gradient clipping
    
    # Logging and checkpointing
    log_steps=10,
    save_steps=50,
    checkpoint_dir=grpo_model_save_dir,
    
    # Optimizer selection
    optimizer_type="custom",  # Use custom GRPO
    
    # GRPO specific parameters
    num_samples=6,  # G in the writeup (number of completions per prompt)
    num_generations=6,  # Legacy parameter for compatibility
    epsilon=0.2,  # PPO clipping parameter
    beta=0.01,  # KL penalty coefficient
    
    # Reward parameters
    gamma=0.99,  # Discount factor
    format_reward=0.0,  # No reward for correct format
    format_penalty=-1.0,  # Penalty for incorrect format
    room_reward=0.0,  # No reward for correct room prediction
    room_penalty=-0.5,  # Moderate penalty for incorrect room prediction
    
    # Training parameters
    num_iterations=3,  # Number of training iterations
    num_episodes_per_iteration=5,  # Number of episodes to collect per iteration
    
    # Data collection parameters
    difficulties=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # First 10 difficulties
    episodes_per_difficulty=2,  # 2 episodes per difficulty
    
    # Agent parameters
    temperature=0.7,
    top_p=0.9,
    use_map=True,
    
    # Device
    device="cuda" if torch.cuda.is_available() else "cpu"
)

print("Configurations created successfully")
print(f"Using device: {rl_config.device}")
print(f"Model: {config.model_config.model_name}")
print(f"Unfreezing last {config.model_config.unfreeze_last_n_obs_layers} layers")
print(f"GRPO batch size: {rl_config.batch_size}")
print(f"GRPO completions per prompt: {rl_config.num_samples}")
print(f"Total completions per batch: {rl_config.batch_size * rl_config.num_samples}")
print(f"Collecting {len(rl_config.difficulties)} difficulties × {rl_config.episodes_per_difficulty} episodes = {len(rl_config.difficulties) * rl_config.episodes_per_difficulty} episodes")

# Monkey patch the TextWorldRLTrainer.__init__ method to handle the nested checkpoint
original_init = TextWorldRLTrainer.__init__

def patched_init(self, rl_config, main_config=None, model_path=None, use_map=True):
    self.rl_config = rl_config
    self.main_config = main_config
    self.use_map = use_map
    
    # Set self.config to rl_config to fix AttributeError
    self.config = rl_config
    
    # Initialize task_config and env_manager
    from environment.task_env import TaskConfig, TaskEnvManager
    self.task_config = TaskConfig(
        max_steps=rl_config.max_steps,
        scale=rl_config.scale if hasattr(rl_config, 'scale') else 10
    )
    self.env_manager = TaskEnvManager(self.task_config)
    
    # Initialize agent with model path
    from agents.textworld_llm_agent import TextWorldLLMAgent
    self.agent = TextWorldLLMAgent(
        config=main_config,
        training_mode=True,
        model_path=None,  # Don't load from path
        use_map=use_map
    )
    
    # Manually initialize model and tokenizer since training_mode=True skips this
    model_name = main_config.model_config.model_name
    is_autoregressive = "t5" not in model_name.lower() and "bart" not in model_name.lower()
    
    # Initialize model based on type
    if is_autoregressive:
        self.agent.model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        self.agent.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Initialize tokenizer
    self.agent.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens for command and room tags
    special_tokens = {
        'additional_special_tokens': ['<command>', '</command>', '<room>', '</room>']
    }
    
    # Add pad token if it doesn't exist (for some autoregressive models)
    if is_autoregressive and self.agent.tokenizer.pad_token is None:
        self.agent.tokenizer.pad_token = self.agent.tokenizer.eos_token
    
    self.agent.tokenizer.add_special_tokens(special_tokens)
    
    # Resize the model's token embeddings to account for the new tokens
    self.agent.model.resize_token_embeddings(len(self.agent.tokenizer))
    
    # Set device
    self.agent.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {self.agent.device}")
    
    # Move model to device
    self.agent.model.to(self.agent.device)
    
    # Load model weights if provided
    if model_path:
        print(f"Loading model from {model_path}")
        try:
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Check if it's a nested checkpoint
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                print("Detected training checkpoint format. Loading model_state_dict.")
                model_state_dict = checkpoint["model_state_dict"]
            else:
                print("Using checkpoint directly as model_state_dict")
                model_state_dict = checkpoint
            
            # Load the state dict
            self.agent.model.load_state_dict(model_state_dict)
            print("Successfully loaded model weights.")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    # Set model to evaluation mode
    self.agent.model.eval()
    
    # Initialize other attributes
    self.model = self.agent.model
    self.tokenizer = self.agent.tokenizer
    self.device = self.agent.device
    
    # Initialize GRPO config
    from config.rl_config import RLConfig
    self.grpo_config = rl_config
    
    # Add missing attributes to grpo_config if they don't exist
    if not hasattr(self.grpo_config, 'max_prompt_length'):
        self.grpo_config.max_prompt_length = self.grpo_config.max_input_length
    if not hasattr(self.grpo_config, 'max_completion_length'):
        self.grpo_config.max_completion_length = self.grpo_config.max_output_length
    
    # Initialize optimizer
    self.optimizer_type = getattr(self.config, 'optimizer_type', 'custom')  # 'custom' or 'huggingface'
    
    if self.optimizer_type == 'huggingface':
        try:
            from trl import PPOConfig, PPOTrainer
            print("Using Hugging Face's PPO implementation")
            
            # Initialize Hugging Face PPO trainer
            ppo_config = PPOConfig(
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                mini_batch_size=self.config.mini_batch_size if hasattr(self.config, 'mini_batch_size') else 4,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps if hasattr(self.config, 'gradient_accumulation_steps') else 1,
                optimize_cuda_cache=True,
                early_stopping=self.config.early_stopping if hasattr(self.config, 'early_stopping') else False,
                target_kl=self.config.target_kl if hasattr(self.config, 'target_kl') else 0.1,
                ppo_epochs=self.config.ppo_epochs if hasattr(self.config, 'ppo_epochs') else 4,
                clip_range=self.config.clip_range if hasattr(self.config, 'clip_range') else 0.2,
                vf_coef=self.config.vf_coef if hasattr(self.config, 'vf_coef') else 0.1,
                horizon=self.config.horizon if hasattr(self.config, 'horizon') else 10000,
                target=self.config.target if hasattr(self.config, 'target') else 6,
                init_kl_coef=self.config.init_kl_coef if hasattr(self.config, 'init_kl_coef') else 0.2,
                adap_kl_ctrl=self.config.adap_kl_ctrl if hasattr(self.config, 'adap_kl_ctrl') else True,
            )
            
            self.ppo_trainer = PPOTrainer(
                config=ppo_config,
                model=self.agent.model,
                ref_model=None,  # Will be set during training
                tokenizer=self.agent.tokenizer,
                dataset=None,  # Will be set during training
                data_collator=None,  # Will be set during training
            )
            
        except ImportError:
            print("Warning: trl package not found. Falling back to custom GRPO implementation.")
            self.optimizer_type = 'custom'
    
    if self.optimizer_type == 'custom':
        from training.optimizer import MyGRPOOptimizer
        print("Using custom GRPO implementation")
        
        # Initialize custom GRPO optimizer
        self.grpo_optimizer = MyGRPOOptimizer(self.config)
    
    # Initialize gameplay data
    self.gameplay_data = []
    
    print(f"Initialized TextWorldRLTrainer with {self.optimizer_type} optimizer")

# Apply the monkey patch
TextWorldRLTrainer.__init__ = patched_init

# Initialize the trainer
print("\nInitializing TextWorldRLTrainer...")
trainer = TextWorldRLTrainer(
    rl_config=rl_config,
    main_config=config,
    model_path=fine_tuned_model_path,
    use_map=True
)

# Step 1: Collect gameplay data
print("\nCollecting gameplay data...")
try:
    trainer.collect_and_save_gameplay_data(
        difficulties=rl_config.difficulties,
        episodes_per_difficulty=rl_config.episodes_per_difficulty,
        save_path=gameplay_data_path
    )
    print(f"Gameplay data saved to {gameplay_data_path}")
except Exception as e:
    print(f"Error collecting gameplay data: {str(e)}")
    raise

# Step 2: Train with GRPO using the collected data
print("\nTraining with GRPO...")
try:
    # Load the gameplay data
    trainer.load_gameplay_data(gameplay_data_path)
    
    # Train with custom GRPO
    final_model_path = os.path.join(grpo_model_save_dir, "grpo_trained_model_final.pt")
    trainer.train(
        use_saved_data=True,
        data_path=gameplay_data_path,
        save_model_path=final_model_path
    )
    print(f"GRPO training completed. Final model saved to {final_model_path}")
except Exception as e:
    print(f"Error during GRPO training: {str(e)}")
    raise

print("\nProcess completed successfully!") 