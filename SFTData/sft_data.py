from agents.textworld_llm_agent import TextWorldLLMAgent
from SFTData.utils import get_llm_response
import re
import json
import os
from typing import List, Dict, Any
import random
from torch.utils.data import Dataset
import torch
from datetime import datetime





class SFTData(Dataset):
    def __init__(self, data_dir: str = "/content/drive/MyDrive/textworld_data/sft"):
        """
        Initialize SFTData manager
        
        Args:
            data_dir: Directory to store/load data files (default for Google Drive)
        """
        self.data_dir = data_dir
        self.examples = []
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
    def generate_training_examples(self, client, agent, env_manager, num_games=100, max_difficulty=10) -> List[Dict[str, str]]:
        """Generate training examples using GPT-4
        
        Args:
            client: OpenAI API client
            agent: TextWorld agent instance
            env_manager: TextWorld environment manager
            num_games: Total number of games to generate examples from
            max_difficulty: Maximum difficulty level to use
            
        Returns:
            List of training examples
        """
        print(f"Generating {num_games} training examples across {max_difficulty} difficulty levels...")
        
        training_data = []
        
        for difficulty in range(1, max_difficulty + 1):
            for game_num in range(num_games // max_difficulty):
                print(f"\nGenerating game {game_num + 1} at difficulty {difficulty}")
                
                # Create new game and reset agent state
                env = env_manager.get_or_create_env(difficulty)
                obs, info = env.reset()
                agent.reset()  # Reset agent state including step count
                done = False
                
                # Parse goal and clean first observation
                print("\nDEBUG - Before parse_goal:")
                print(f"DEBUG - Agent goal: {agent.goal}")
                agent.parse_goal(obs)
                print(f"DEBUG - After parse_goal:")
                print(f"DEBUG - Agent goal: {agent.goal}")
                clean_obs = agent._clean_first_observation(agent._clean_observation(obs))
                print(f"DEBUG - Cleaned observation: {clean_obs[:200]}...")
                
                while not done:
                    valid_actions = info['admissible_commands']
                    filtered_actions = self.filter_valid_actions(valid_actions, agent.true_state['step_count'])
                    room = agent._get_room_name(obs)
                    
                    # Only clean subsequent observations normally
                    if agent.true_state['step_count'] > 0:
                        clean_obs = agent._clean_observation(obs)
                        print(f"DEBUG: New Cleaned Observation: {clean_obs}")
                    
                    prompt = f"""You are an expert at text adventure games. Analyze this game state and generate a training example:

Game State:
Goal: Take the latchkey from the floor. You may have to explore other rooms to find it.
Location: {room}
Observation: {clean_obs}
Previous action history: (Hidden - you don't know your previous actions)
Currently available actions: {filtered_actions}

Generate a *concise* response that analyzes each available action and makes the optimal choice:

A) First, analyze each possible action:
[Action Number]) [Action]
[Analysis of what this action would do]

B) Then, state your chosen action - Make sure it is in the available actions list:
Therefore, I choose: [exact action]

Your response:"""

                    # Get analysis from GPT-4
                    response = get_llm_response(client, prompt)
                    
                    if response:
                        # Save as training example
                        training_data.append({
                            "input": prompt,
                            "output": response,
                            "difficulty": difficulty,
                            "step": agent.true_state['step_count']
                        })
                        
                        # Use filtered_actions for validation
                        action = self.extract_action(response, filtered_actions)
                        if action:
                            obs, reward, done, info = env.step(action)
                            # Update agent's step count
                            agent.true_state['step_count'] += 1
                            print(f'DEBUG: Step taken: {action}')
                            print(f"DEBUG: New True Observation: {obs}")
                        else:
                            print(f"Warning: Could not extract valid action from response")
                            break
                    else:
                        print(f"Warning: No response received from API")
                        break
                
                env.close()
        
        self.examples.extend(training_data)
        return training_data
    
    def extract_action(self,response, valid_actions):
        """Extract chosen action from model response
        
        Args:
            response: Full response from GPT-4o
            valid_actions: List of valid actions for validation
        
        Returns:
            Extracted action if found and valid, else None
        """
        # Look for "Therefore, I choose: " pattern
        choice_match = re.search(r"Therefore,\s*I\s*choose:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if choice_match:
            action = choice_match.group(1).strip()
            if action in valid_actions:
                return action
                
        # Fallback: look for any valid action in the response
        found_actions = [action for action in valid_actions if action in response]
        if len(found_actions) == 1:
            return found_actions[0]
            
        return None
    
    def save_data(self, filename: str = None):
        """Save training data to JSON file"""
        if not filename:
            # Generate filename based on number of examples and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sft_data_{len(self.examples)}_{timestamp}.json"
        
        filepath = os.path.join(self.data_dir, filename)
        
        print(f"Saving {len(self.examples)} examples to {filepath}")
        with open(filepath, 'w') as f:
            json.dump(self.examples, f)
    
    def load_data(self, filename: str = None):
        """Load training data from JSON file(s)"""
        if filename:
            # Load specific file
            filepath = os.path.join(self.data_dir, filename)
            with open(filepath, 'r') as f:
                self.examples.extend(json.load(f))
        else:
            # Load all JSON files in directory
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.data_dir, filename)
                    with open(filepath, 'r') as f:
                        self.examples.extend(json.load(f))
        
        print(f"Loaded {len(self.examples)} examples")
    
    def prepare_batch(self, batch_size: int = 32, shuffle: bool = True) -> List[Dict[str, str]]:
        """Prepare a batch of examples for training"""
        if shuffle:
            batch_indices = random.sample(range(len(self.examples)), min(batch_size, len(self.examples)))
        else:
            batch_indices = list(range(min(batch_size, len(self.examples))))
        
        return [self.examples[i] for i in batch_indices]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def get_stats(self):
        """Get statistics about the dataset"""
        if not self.examples:
            return "No examples loaded"
            
        stats = {
            "total_examples": len(self.examples),
            "examples_per_difficulty": {},
            "avg_steps_per_game": sum(ex["step"] for ex in self.examples) / len(self.examples),
        }
        
        for ex in self.examples:
            diff = ex["difficulty"]
            stats["examples_per_difficulty"][diff] = stats["examples_per_difficulty"].get(diff, 0) + 1
            
        return stats

    def filter_valid_actions(self, valid_actions, step_count):
        """Strategically filter valid actions to encourage diversity"""
        print(f"\nDEBUG - Before filtering: {valid_actions}")
        
        ALWAYS_AVAILABLE = ['go north', 'go south', 'go east', 'go west', 'take', 'open', 'unlock']
        INFO_ACTIONS = ['look', 'inventory', 'examine']
        
        # Keep all navigation/interaction actions
        filtered_actions = [action for action in valid_actions 
                           if any(action.startswith(act) for act in ALWAYS_AVAILABLE)]
        
        # Randomly include info actions with decreasing probability
        for action in valid_actions:
            if any(action.startswith(act) for act in INFO_ACTIONS):
                prob = max(0.1, 1.0 - (step_count * 0.2))
                print(f"DEBUG - Info action: {action}, probability: {prob}")
                if random.random() < prob:
                    filtered_actions.append(action)
        
        print(f"DEBUG - After filtering: {filtered_actions}")
        return filtered_actions if filtered_actions else valid_actions
