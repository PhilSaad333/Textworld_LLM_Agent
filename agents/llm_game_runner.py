import json
from datetime import datetime
from pathlib import Path
import os

class GameRunner:
    def __init__(self, agent, env_manager, config, log_dir="logs/games"):
        """
        Initialize game runner
        
        Args:
            agent: TextWorld agent instance
            env_manager: Environment manager instance
            config: Game configuration
            log_dir: Directory to save game logs
        """
        self.agent = agent
        self.env_manager = env_manager
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def play_game(self, difficulty=1, log=True):
        """
        Play a single game
        
        Args:
            difficulty: Game difficulty level
            log: Whether to log the game
            
        Returns:
            dict: Game results including trajectory and metrics
        """
        # Initialize game record
        game_record = {
            'difficulty': difficulty,
            'timestamp': datetime.now().isoformat(),
            'trajectory': [],
            'metrics': {
                'steps': 0,
                'score': 0,
                'success': False,
                'format_failures': 0
            }
        }
        
        # Get environment
        env = self.env_manager.get_or_create_env(difficulty)
        obs, infos = env.reset()
        
        # Clean initial observation for display
        clean_obs = self.agent._clean_observation(obs)
        print(f"\nStarting game with difficulty {difficulty}")
        print(f"Initial observation (cleaned): {clean_obs[:200]}...")
        print(f"Initial inventory: {infos.get('inventory', 'nothing')}")
        print(f"Valid actions: {infos['admissible_commands']}")
        
        # Extract and display goal
        goal = self.agent.parse_goal(obs)
        print(f"Goal: {goal}")
        
        done = False
        step = 0
        total_format_failures = 0
        
        while not done and step < self.config.max_steps:
            # Get valid actions
            valid_actions = [
                a for a in infos["admissible_commands"]
                if a.lower() not in ['inventory', 'look']
            ]
            
            print(f"\nStep {step}:")
            current_room = self.agent._get_room_name(obs)
            print(f"Room: {current_room}")
            print(f"Valid actions: {valid_actions}")
            
            # Get action from agent
            action, action_info = self.agent.get_action(
                env,
                obs,
                infos,
                valid_actions,
                step
            )
            
            if action is None:  # Agent detected terminal state
                print("Agent detected terminal state")
                break
                
            # Extract format check information
            format_check_passed = self.agent.true_state.get('format_check_passed', True)
            step_format_failures = self.agent.true_state.get('format_failures', 0)
            predicted_room = self.agent.true_state.get('predicted_room', "Unknown")
            
            total_format_failures += step_format_failures
            
            # Take action
            next_obs, reward, done, next_infos = env.step(action)
            
            # Apply format failure penalty if configured
            format_penalty = 0
            if not format_check_passed and hasattr(self.config, 'format_failure_penalty'):
                format_penalty = self.config.format_failure_penalty
                reward += format_penalty
                print(f"Applied format failure penalty: {format_penalty}")
            
            # Update agent's state
            self.agent.update_state_after_action(next_obs, reward, done, next_infos)
            
            # Get the actual next room
            next_room = self.agent._get_room_name(next_obs)
            room_prediction_correct = (predicted_room == next_room)
            
            # Record step
            step_record = {
                'step': step,
                'observation': obs,
                'action': action,
                'reward': reward,
                'done': done,
                'valid_actions': valid_actions,
                'room': current_room,
                'next_room': next_room,
                'predicted_room': predicted_room,
                'room_prediction_correct': room_prediction_correct,
                'inventory': infos.get('inventory', ''),
                'format_check_passed': format_check_passed,
                'format_failures': step_format_failures,
                'format_penalty': format_penalty
            }
            game_record['trajectory'].append(step_record)
            
            print(f"Action taken: {action}")
            print(f"Predicted room: {predicted_room}")
            print(f"Actual next room: {next_room}")
            print(f"Room prediction correct: {room_prediction_correct}")
            print(f"Format check passed: {format_check_passed}")
            print(f"Reward: {reward}")
            print(f"Done: {done}")
            if reward != 0:
                print(f"Score changed! New score: {game_record['metrics']['score'] + reward}")
            
            # Update for next step
            obs, infos = next_obs, next_infos
            step += 1
            game_record['metrics']['score'] += reward
            
        # Update final metrics
        game_record['metrics'].update({
            'steps': step,
            'success': game_record['metrics']['score'] > 0,
            'format_failures': total_format_failures
        })
        
        # Log game if requested
        if log:
            self._save_game_log(game_record)
            
        print(f"\nGame finished:")
        print(f"Steps: {step}")
        print(f"Score: {game_record['metrics']['score']}")
        print(f"Success: {game_record['metrics']['success']}")
        print(f"Total format failures: {total_format_failures}")
        
        return game_record
    
    def _save_game_log(self, game_record):
        """Save game log to file"""
        timestamp = game_record['timestamp'].replace(':', '-')
        filename = f"game_diff{game_record['difficulty']}_{timestamp}.json"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(game_record, f, indent=2)
        print(f"\nGame log saved to: {filepath}")
        
    def play_multiple_games(self, difficulties=None, games_per_difficulty=1):
        """
        Play multiple games across different difficulty levels
        
        Args:
            difficulties: List of difficulty levels to play
            games_per_difficulty: Number of games to play per difficulty level
            
        Returns:
            dict: Summary of results across all games
        """
        if difficulties is None:
            difficulties = [1, 5, 10, 15, 20]  # Default difficulties
            
        results = {
            'games': [],
            'summary': {
                'total_games': 0,
                'success_rate': 0,
                'avg_score': 0,
                'avg_steps': 0,
                'avg_format_failures': 0
            }
        }
        
        total_success = 0
        total_score = 0
        total_steps = 0
        total_format_failures = 0
        
        for difficulty in difficulties:
            for game_num in range(games_per_difficulty):
                print(f"\n{'='*50}")
                print(f"Playing game {game_num+1}/{games_per_difficulty} at difficulty {difficulty}")
                print(f"{'='*50}")
                
                game_result = self.play_game(difficulty=difficulty)
                
                # Add to results
                results['games'].append({
                    'difficulty': difficulty,
                    'game_num': game_num,
                    'success': game_result['metrics']['success'],
                    'score': game_result['metrics']['score'],
                    'steps': game_result['metrics']['steps'],
                    'format_failures': game_result['metrics'].get('format_failures', 0)
                })
                
                # Update totals
                total_success += 1 if game_result['metrics']['success'] else 0
                total_score += game_result['metrics']['score']
                total_steps += game_result['metrics']['steps']
                total_format_failures += game_result['metrics'].get('format_failures', 0)
        
        # Calculate summary
        total_games = len(difficulties) * games_per_difficulty
        results['summary'] = {
            'total_games': total_games,
            'success_rate': total_success / total_games if total_games > 0 else 0,
            'avg_score': total_score / total_games if total_games > 0 else 0,
            'avg_steps': total_steps / total_games if total_games > 0 else 0,
            'avg_format_failures': total_format_failures / total_games if total_games > 0 else 0
        }
        
        # Save summary
        summary_path = self.log_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nMultiple games summary:")
        print(f"Total games: {total_games}")
        print(f"Success rate: {results['summary']['success_rate']:.2f}")
        print(f"Average score: {results['summary']['avg_score']:.2f}")
        print(f"Average steps: {results['summary']['avg_steps']:.2f}")
        print(f"Average format failures: {results['summary']['avg_format_failures']:.2f}")
        
        return results


# Function to set up and run games
def setup_and_run_games(model_path, difficulties=None, games_per_difficulty=1, use_map=False):
    """
    Set up the environment, agent, and run games
    
    Args:
        model_path: Path to the fine-tuned model
        difficulties: List of difficulty levels to play
        games_per_difficulty: Number of games to play per difficulty level
        use_map: If True, use the map tool to track room connections
    """
    from environment.task_env import TaskEnvManager, TaskConfig
    from agents.textworld_llm_agent import TextWorldLLMAgent
    from config.config import get_game_config, RewardType, GoalType
    
    # Create configuration
    config = get_game_config(
        RewardType.DENSE,
        GoalType.BRIEF,
        max_history_actions=3
    )
    
    # Add format failure penalty to config
    config.format_failure_penalty = -0.1  # Small penalty for format failures
    
    # Create environment manager
    task_config = TaskConfig(
        max_steps=100,
        scale=10
    )
    env_manager = TaskEnvManager(task_config)
    
    # Create agent with fine-tuned model
    agent = TextWorldLLMAgent(config, model_path=model_path, use_map=use_map)
    
    # Create game runner
    log_dir = "logs/games"  # Use local path instead of Drive
    runner = GameRunner(agent, env_manager, config, log_dir=log_dir)
    
    # Run games
    if difficulties is None:
        difficulties = [1, 5, 10, 15, 20]
        
    results = runner.play_multiple_games(
        difficulties=difficulties,
        games_per_difficulty=games_per_difficulty
    )
    
    return results

# Use local path
log_dir = "logs/games"
