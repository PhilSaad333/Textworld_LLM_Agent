import os
import argparse
import torch
from datetime import datetime

# Import your custom modules
from config.config import get_game_config, RewardType, GoalType, GameType
from config.rl_config import RLConfig
from environment.task_env import TaskEnvManager, TaskConfig
from agents.textworld_llm_agent import TextWorldLLMAgent
from training.trainer import TextWorldRLTrainer

def setup_directories():
    """Create necessary directories for saving data and models"""
    # Set up directories for checkpoints and data
    os.makedirs("checkpoints/rl", exist_ok=True)
    
    # If in Colab, set up Google Drive directories
    try:
        from google.colab import drive
        drive_mounted = os.path.exists('/content/drive')
        if not drive_mounted:
            drive.mount('/content/drive')
        
        # Create directories in Google Drive
        os.makedirs('/content/drive/MyDrive/textworld_rl_data', exist_ok=True)
        os.makedirs('/content/drive/MyDrive/textworld_rl_models', exist_ok=True)
        
        print("Google Drive mounted and directories created.")
        return True
    except ImportError:
        print("Not running in Colab or Google Drive not available.")
        return False

def create_configs(args):
    """Create configurations based on command line arguments"""
    # Create main game config
    main_config = get_game_config(
        reward_type=RewardType.DENSE,
        goal_type=GoalType.DETAILED,
        max_history_actions=3
    )
    
    # Create RL config
    rl_config = RLConfig(
        # Environment parameters
        max_steps=args.max_steps,
        scale=args.scale,
        
        # Model parameters
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_output_length=args.max_output_length,
        max_input_length=args.max_input_length,
        num_epochs=args.num_epochs,
        
        # Logging and checkpointing
        log_steps=args.log_steps,
        save_steps=args.save_steps,
        checkpoint_dir=args.checkpoint_dir,
        
        # GRPO specific parameters
        num_generations=args.num_generations,
        beta=args.beta,
        use_vllm=args.use_vllm,
        
        # Reward parameters
        gamma=args.gamma,
        format_failure_penalty=args.format_failure_penalty,
        room_prediction_penalty=args.room_prediction_penalty,
        
        # Data collection parameters
        difficulties=args.difficulties,
        episodes_per_difficulty=args.episodes_per_difficulty,
        
        # Agent parameters
        temperature=args.temperature,
        top_p=args.top_p,
        use_map=args.use_map
    )
    
    return main_config, rl_config

def collect_data(trainer, args):
    """Collect gameplay data and save it"""
    print("\n=== Collecting Gameplay Data ===")
    
    # Determine save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.in_colab:
        save_dir = '/content/drive/MyDrive/textworld_rl_data'
    else:
        save_dir = 'data'
        os.makedirs(save_dir, exist_ok=True)
    
    save_path = f"{save_dir}/gameplay_data_{timestamp}.json"
    
    # Collect and save data
    data_path = trainer.collect_and_save_gameplay_data(
        difficulties=args.difficulties,
        episodes_per_difficulty=args.episodes_per_difficulty,
        save_path=save_path
    )
    
    print(f"Data saved to: {data_path}")
    return data_path

def train_model(trainer, args, data_path=None):
    """Train the model using collected data"""
    print("\n=== Training Model ===")
    
    # Train using saved data if specified
    if args.use_saved_data and data_path:
        print(f"Using saved data from: {data_path}")
        trainer.train(use_saved_data=True, data_path=data_path)
    else:
        print("Collecting new data for training")
        trainer.train(use_saved_data=False)
    
    print("Training complete!")

def evaluate_model(trainer, args):
    """Evaluate the trained model"""
    print("\n=== Evaluating Model ===")
    
    results = trainer.evaluate(
        difficulties=args.eval_difficulties,
        episodes_per_difficulty=args.eval_episodes
    )
    
    # Print results
    print("\nEvaluation Results:")
    for difficulty, metrics in results.items():
        print(f"Difficulty {difficulty}:")
        print(f"  Success Rate: {metrics['success_rate']:.2f}")
        print(f"  Average Reward: {metrics['avg_reward']:.2f}")
        print(f"  Average Steps: {metrics['avg_steps']:.2f}")
    
    return results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="TextWorld RL Training")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="all", 
                        choices=["collect", "train", "evaluate", "all"],
                        help="Operation mode: collect data, train model, evaluate model, or all")
    
    # Model loading
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to pretrained model")
    
    # Data handling
    parser.add_argument("--use_saved_data", action="store_true",
                        help="Use saved gameplay data instead of collecting new data")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to saved gameplay data")
    
    # Environment parameters
    parser.add_argument("--max_steps", type=int, default=50,
                        help="Maximum steps per episode")
    parser.add_argument("--scale", type=int, default=10,
                        help="Scale parameter for environment")
    
    # Model parameters
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_output_length", type=int, default=256,
                        help="Maximum output length")
    parser.add_argument("--max_input_length", type=int, default=512,
                        help="Maximum input length")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    
    # Logging and checkpointing
    parser.add_argument("--log_steps", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every N steps")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/rl",
                        help="Directory to save checkpoints")
    
    # GRPO specific parameters
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Number of completions to generate per prompt")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty coefficient")
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM for faster generation")
    
    # Reward parameters
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--format_failure_penalty", type=float, default=-0.2,
                        help="Penalty for incorrect format")
    parser.add_argument("--room_prediction_penalty", type=float, default=-0.1,
                        help="Penalty for incorrect room prediction")
    
    # Data collection parameters
    parser.add_argument("--difficulties", type=int, nargs="+", default=[1, 5, 10],
                        help="List of difficulties to collect data from")
    parser.add_argument("--episodes_per_difficulty", type=int, default=5,
                        help="Number of episodes to collect per difficulty")
    
    # Agent parameters
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--use_map", action="store_true", default=True,
                        help="Use map tool in agent")
    
    # Evaluation parameters
    parser.add_argument("--eval_difficulties", type=int, nargs="+", default=[1, 5, 10, 15],
                        help="List of difficulties to evaluate on")
    parser.add_argument("--eval_episodes", type=int, default=3,
                        help="Number of episodes per difficulty for evaluation")
    
    args = parser.parse_args()
    
    # Check if running in Colab
    try:
        import google.colab
        args.in_colab = True
    except ImportError:
        args.in_colab = False
    
    return args

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set up directories
    args.in_colab = setup_directories()
    
    # Create configurations
    main_config, rl_config = create_configs(args)
    
    # Print configurations
    print("\n=== Configuration ===")
    print(f"Mode: {args.mode}")
    print(f"Model Path: {args.model_path}")
    print(f"Use Saved Data: {args.use_saved_data}")
    print(f"Data Path: {args.data_path}")
    print(f"Difficulties: {args.difficulties}")
    print(f"Episodes per Difficulty: {args.episodes_per_difficulty}")
    print(f"Use Map: {args.use_map}")
    
    # Initialize trainer
    print("\n=== Initializing Trainer ===")
    trainer = TextWorldRLTrainer(
        rl_config=rl_config,
        main_config=main_config,
        model_path=args.model_path,
        use_map=args.use_map
    )
    
    # Execute requested operations
    data_path = None
    
    if args.mode in ["collect", "all"]:
        data_path = collect_data(trainer, args)
    
    if args.mode in ["train", "all"]:
        # If using saved data but no path provided, use the one we just collected
        if args.use_saved_data and not args.data_path:
            args.data_path = data_path
        
        train_model(trainer, args, args.data_path)
    
    if args.mode in ["evaluate", "all"]:
        evaluate_model(trainer, args)
    
    print("\n=== Done! ===")

if __name__ == "__main__":
    main()