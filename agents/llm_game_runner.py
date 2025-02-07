import json
from datetime import datetime
from pathlib import Path
from google.colab import drive

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
                'success': False
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
        
        done = False
        step = 0
        
        while not done and step < self.config.max_steps:
            # Get valid actions
            valid_actions = [
                a for a in infos["admissible_commands"]
                if a.lower() not in ['inventory', 'look']
            ]
            
            print(f"\nStep {step}:")
            print(f"Room: {self.agent._get_room_name(obs)}")
            print(f"Valid actions: {valid_actions}")
            
            # Get action from agent
            action, _ = self.agent.get_action(
                env,
                obs,
                infos,
                valid_actions,
                step
            )
            
            if action is None:  # Agent detected terminal state
                print("Agent detected terminal state")
                break
                
            # Take action
            next_obs, reward, done, next_infos = env.step(action)
            
            # Update agent's state
            self.agent.update_state_after_action(next_obs, reward, done, next_infos)
            
            # Record step
            step_record = {
                'step': step,
                'observation': obs,
                'action': action,
                'reward': reward,
                'done': done,
                'valid_actions': valid_actions,
                'room': self.agent._get_room_name(obs),
                'inventory': infos.get('inventory', '')
            }
            game_record['trajectory'].append(step_record)
            
            print(f"Action taken: {action}")
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
            'success': game_record['metrics']['score'] > 0
        })
        
        # Log game if requested
        if log:
            self._save_game_log(game_record)
            
        print(f"\nGame finished:")
        print(f"Steps: {step}")
        print(f"Score: {game_record['metrics']['score']}")
        print(f"Success: {game_record['metrics']['success']}")
        
        return game_record
    
    def _save_game_log(self, game_record):
        """Save game log to file"""
        timestamp = game_record['timestamp'].replace(':', '-')
        filename = f"game_diff{game_record['difficulty']}_{timestamp}.json"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(game_record, f, indent=2)
        print(f"\nGame log saved to: {filepath}")

# Mount Google Drive
drive.mount('/content/drive')

# Use Drive path
log_dir = "/content/drive/MyDrive/textworld_logs/games"
