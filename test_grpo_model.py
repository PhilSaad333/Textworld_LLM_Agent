import os
import sys
import torch
import numpy as np
import random
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path to import your modules
if not '/content/Textworld_LLM_Agent' in sys.path:
    sys.path.append('/content/Textworld_LLM_Agent')

# Import necessary modules
from config.config import TextWorldConfig, ModelConfig, GameConfig, RewardType, GoalType, GameType, get_game_config, EvalConfig
from environment.task_env import TaskConfig, TaskEnvManager
from agents.textworld_llm_agent import TextWorldLLMAgent
from agents.llm_game_runner import GameRunner

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # Set random seed
    set_seed()
    
    # Define paths
    model_path = "/content/drive/MyDrive/textworld_rl_models/grpo_trained_model/grpo_trained_model_final.pt"
    log_dir = "/content/drive/MyDrive/textworld_rl_models/evaluation_logs"
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create evaluation config
    eval_config = EvalConfig(
        # Generation parameters
        num_beams=6,
        num_return_sequences=6,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        
        # Evaluation behavior
        print_completions=True,
        print_format_check=True,
        print_action_selection=True,
        
        # Action selection strategy
        action_selection="best_beam",
        
        # Logging
        log_to_file=True,
        log_path=os.path.join(log_dir, f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    )
    
    # Create game config
    config = get_game_config(
        reward_type=RewardType.DENSE,
        goal_type=GoalType.DETAILED,
        max_history_actions=3,
        eval_config=eval_config
    )
    
    # Create environment manager
    env_manager = TaskEnvManager(TaskConfig(max_steps=20))
    
    # Create agent
    agent = TextWorldLLMAgent(
        config=config,
        model_path=model_path,
        use_map=False
    )
    
    # Create game runner
    game_runner = GameRunner(
        agent=agent,
        env_manager=env_manager,
        config=config,
        log_dir=log_dir,
        eval_config=eval_config
    )
    
    # Define difficulties to test
    difficulties = [1, 5, 10, 15]
    
    # Play games at each difficulty
    results = {}
    
    for difficulty in difficulties:
        print(f"\n{'='*50}")
        print(f"Testing difficulty level {difficulty}")
        print(f"{'='*50}\n")
        
        # Play a game at this difficulty
        game_record = game_runner.play_game(difficulty=difficulty, log=True)
        
        # Store results
        results[difficulty] = {
            "success": game_record["success"],
            "steps": len(game_record["actions"]),
            "score": game_record["score"],
            "format_check_passed": sum(1 for step in game_record["steps"] if step.get("format_check_passed", False)),
            "total_steps": len(game_record["steps"])
        }
    
    # Print summary
    print("\n\n" + "="*50)
    print("Evaluation Summary")
    print("="*50)
    
    for difficulty, result in results.items():
        print(f"\nDifficulty {difficulty}:")
        print(f"  Success: {result['success']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Score: {result['score']}")
        print(f"  Format Check Passed: {result['format_check_passed']}/{result['total_steps']} steps ({result['format_check_passed']/result['total_steps']*100:.1f}%)")
    
    # Calculate overall statistics
    success_rate = sum(1 for r in results.values() if r["success"]) / len(results)
    avg_steps = sum(r["steps"] for r in results.values()) / len(results)
    avg_format_check_passed = sum(r["format_check_passed"] for r in results.values()) / sum(r["total_steps"] for r in results.values())
    
    print("\nOverall Statistics:")
    print(f"  Success Rate: {success_rate*100:.1f}%")
    print(f"  Average Steps: {avg_steps:.1f}")
    print(f"  Format Check Passed: {avg_format_check_passed*100:.1f}%")
    
    # Save results to file
    results_path = os.path.join(log_dir, f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(results_path, "w") as f:
        f.write("Evaluation Results\n")
        f.write("="*50 + "\n\n")
        
        for difficulty, result in results.items():
            f.write(f"Difficulty {difficulty}:\n")
            f.write(f"  Success: {result['success']}\n")
            f.write(f"  Steps: {result['steps']}\n")
            f.write(f"  Score: {result['score']}\n")
            f.write(f"  Format Check Passed: {result['format_check_passed']}/{result['total_steps']} steps ({result['format_check_passed']/result['total_steps']*100:.1f}%)\n\n")
        
        f.write("Overall Statistics:\n")
        f.write(f"  Success Rate: {success_rate*100:.1f}%\n")
        f.write(f"  Average Steps: {avg_steps:.1f}\n")
        f.write(f"  Format Check Passed: {avg_format_check_passed*100:.1f}%\n")
    
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main() 