import os
import torch
import random
import numpy as np
from pathlib import Path

# Import project modules
from config.config import get_game_config, RewardType, GoalType, GameType
from environment.task_env import TaskEnvManager, TaskConfig
from agents.textworld_llm_agent import TextWorldLLMAgent, MapTool
from agents.llm_game_runner import GameRunner

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def run_game(difficulty=1, use_map=True, seed=42):
    """
    Run a single game with the GPT-2 Medium model
    
    Args:
        difficulty: Game difficulty level (1-30)
        use_map: Whether to use the map tool
        seed: Random seed for reproducibility
    """
    # Set random seed
    set_seed(seed)
    
    print(f"Running game with difficulty {difficulty}, use_map={use_map}")
    
    # Create game config
    config = get_game_config(
        reward_type=RewardType.DENSE,
        goal_type=GoalType.DETAILED,
        max_history_actions=5
    )
    
    # Update game type and difficulty
    config.game_config.game_type = GameType.TREASURE
    config.game_config.treasure_level = difficulty
    
    # Create task environment
    task_config = TaskConfig(max_steps=100)
    env_manager = TaskEnvManager(task_config)
    env_manager.seed(seed)
    
    # Initialize agent
    agent = TextWorldLLMAgent(config, use_map=use_map)
    
    # Create game runner
    log_dir = Path("logs/games")
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = GameRunner(agent, env_manager, config, log_dir=log_dir)
    
    # Play game
    game_results = runner.play_game(difficulty=difficulty, log=True)
    
    # Print results
    print("\n" + "="*50)
    print(f"Game completed with score: {game_results['score']}/{game_results['max_score']}")
    print(f"Steps taken: {game_results['steps']}")
    print(f"Game completed: {'Yes' if game_results['won'] else 'No'}")
    print("="*50)
    
    return game_results

if __name__ == "__main__":
    # Run a game with difficulty level 1
    run_game(difficulty=1, use_map=True)
    
    # Uncomment to run multiple games with different difficulties
    # for difficulty in [1, 2, 3]:
    #     run_game(difficulty=difficulty, use_map=True) 