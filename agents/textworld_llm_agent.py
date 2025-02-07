import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re

class TextWorldLLMAgent:
    def __init__(self, config, training_mode=False):
        """
        Initialize LLM-based TextWorld agent
        
        Args:
            config: Configuration object containing model settings
            training_mode: If True, enables batch processing and disables debug output
        """
        self.config = config
        self.training_mode = training_mode
        
        # Verify config has required attributes
        if not hasattr(config, 'game_config'):
            raise ValueError("Config must have game_config attribute")
        if not hasattr(config.game_config, 'treasure_level'):
            raise ValueError("game_config must have treasure_level attribute")
        if not hasattr(config, 'max_steps'):
            raise ValueError("Config must have max_steps attribute")
        
        # Initialize model and tokenizer only if not in training mode
        if not training_mode:
            print("Loading FLAN-T5-BASE model and tokenizer...")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
                self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
                
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Using device: {self.device}")
                self.model.to(self.device)
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {str(e)}")
        
        # Initialize state tracking
        self.reset()
        
    def __del__(self):
        """Cleanup when agent is deleted"""
        try:
            # Clear GPU memory
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Error during cleanup: {str(e)}")
        
    def reset(self):
        """Reset agent state"""
        self.goal = None
        self.goals = [] if self.training_mode else None
        self.known_rooms = set()
        self.last_known_room = None
        self.action_room_history = [] if not self.training_mode else None
        self.true_state = {'step_count': 0}  # Initialize step count

    def parse_goal(self, initial_obs):
        """Extract goal from initial observation"""
        print("\nDEBUG - parse_goal called")
        print(f"DEBUG - Initial observation: {initial_obs[:200]}...")  # First 200 chars to keep output readable
        
        # Split into lines and clean
        lines = [line.strip() for line in initial_obs.split('\n') 
                if line.strip() and sum(c in r'\_|/$[]{}=+-' for c in line) <= len(line) * 0.1]
        
        print(f"DEBUG - Cleaned lines: {lines[:3]}...")  # First few lines
        
        if not lines:
            print("WARNING - Could not find any valid lines in observation")
            return "Unknown"
        
        # Find the line containing the goal
        goal_line = None
        goal_markers = [
            "Your task",
            "First stop",
            "First thing",
            "First off",
            "First of all",
            "First step",
            "Your first objective",
            "Here is how to play",
            "Here is your task",
            "there is something I need you to do",
            "Who's got a virtual machine",
            "Welcome to"
        ]
        
        # Try to find the goal line
        for line in lines:
            # Check for any goal marker
            if any(marker.lower() in line.lower() for marker in goal_markers):
                goal_line = line
                self.raw_goal_line = line
                print(f"DEBUG - Found goal line with marker: {line}")
                break
                
        # If no goal line found with markers, look for the pattern that often appears in these games
        if not goal_line:
            for line in lines:
                if ("TextWorld" in line and 
                    any(x in line.lower() for x in ["playing", "entered", "welcome", "ready"])):
                    goal_line = line
                    self.raw_goal_line = line
                    print(f"DEBUG - Found goal line with TextWorld pattern: {line}")
                    break
        
        if not goal_line:
            print("WARNING - Could not find goal line")
            return lines[0]
        
        self.raw_goal_line = goal_line
            
        # Remove intro text with more conservative patterns
        intro_patterns = [
            # Remove standard TextWorld intros but preserve the actual task description
            r"^.*?TextWorld!\s*",
            r"^Get ready to pick stuff up and put it in places, because you've just entered TextWorld!\s*",
            r"^Hey, thanks for coming over to the TextWorld today, there is something I need you to do for me\.\s*",
            r"^I hope you're ready to go into rooms and interact with objects, because you've just entered TextWorld!\s*",
            r"^Welcome to (?:another )?(?:profound|exciting|fast paced|life changing) (?:game|episode|round|session) of TextWorld!\s*",
            r"^It's time to explore the amazing world of TextWorld!\s*",
            r"^You are now playing an? (?:profound|exciting|fast paced|life changing) (?:game|episode|round|session) of TextWorld!\s*",
            r"^Who's got a virtual machine and is about to play through an? (?:profound|exciting|fast paced) round of TextWorld\? You do!\s*",
            
            # Clean up transitions while preserving task descriptions
            r"^Here is how to play!\s*",
            r"^Here is your task for today\.\s*",
            r"to play!\s*",
        ]
        
        cleaned_goal = goal_line
        for pattern in intro_patterns:
            cleaned_goal = re.sub(pattern, "", cleaned_goal, flags=re.IGNORECASE).strip()
            
        # Final cleanup
        cleaned_goal = cleaned_goal.strip('., ')
        if cleaned_goal.lower().startswith("i need you to "):
            cleaned_goal = cleaned_goal[len("i need you to "):].strip()
        
        print(f"DEBUG - Final cleaned goal: {cleaned_goal}")
        print(f"DEBUG - Setting self.goal to: {cleaned_goal}")
        
        self.goal = cleaned_goal
        return cleaned_goal
    
    def _clean_first_observation(self, obs):
        """Remove ascii art and goal line from first observation, keeping only what comes after"""
        if hasattr(self, 'raw_goal_line'):
            # Split on the raw goal line and take everything after it
            parts = obs.split(self.raw_goal_line)
            if len(parts) > 1:
                # Take everything after the goal line
                clean_obs = parts[1].strip()
                # If clean_obs is empty, try the next non-empty part
                if not clean_obs and len(parts) > 2:
                    clean_obs = parts[2].strip()
                return clean_obs
        return obs

    def _get_room_name(self, obs):
        """Extract room name from observation"""
        # Look for room name between -= and =- on its own line
        room_match = re.search(r'^-=\s*([^=]+?)\s*=-\s*$', obs, re.MULTILINE)
        if room_match:
            room_name = room_match.group(1).strip()
            self.last_known_room = room_name
            return room_name
            
        # If we can't find a new room name, return the last known room
        if self.last_known_room:
            return self.last_known_room
        
        # If we have no room information yet, log it
        print(f"DEBUG - Could not find room name in observation:\n{obs[:200]}...")
        return None
        
    def _clean_observation(self, obs):
        """Remove ASCII art and clean up observation text"""
        # First, split by double newlines to separate sections
        sections = obs.split('\n\n')
        
        # Find the first section that doesn't look like ASCII art
        clean_sections = []
        for section in sections:
            # Skip sections that are mostly special characters (ASCII art)
            if not section.strip() or sum(c in r'\_|/$[]{}=+-' for c in section) > len(section) * 0.1:
                continue
            clean_sections.append(section)
        
        # Join the clean sections
        clean_obs = '\n'.join(clean_sections)
        
        # Remove any remaining ASCII art patterns
        clean_obs = re.sub(r'[|_=\-+\\/${}[\]]+', '', clean_obs)
        
        # Clean up multiple spaces and newlines
        clean_obs = re.sub(r'\s+', ' ', clean_obs).strip()
        
        return clean_obs

    def format_prompt(self, obs, valid_actions, infos, batch_mode=False):
        """
        Format input for LLM. Can handle both single inputs and batched inputs.
        
        Args:
            obs: Single observation string or list of observation strings
            valid_actions: Single list of valid actions or list of lists
            infos: Single info dict or list of info dicts
            batch_mode: If True, process inputs as batch
            
        Returns:
            Single formatted prompt string or list of prompt strings
        """
        if not batch_mode:
            # Single input processing - original logic
            inventory = infos.get('inventory', 'nothing')
            room = self._get_room_name(obs)
            clean_obs = self._clean_observation(obs)
            
            history_pairs = [f"({r}: {a})" for r, a in self.action_room_history[-self.config.game_config.max_history_actions:]]
            history_str = " → ".join(history_pairs) if history_pairs else "None"
            
            numbered_actions = [f"{i+1}) {action}" for i, action in enumerate(valid_actions)]
            
            prompt = f"""You are playing a text adventure game. Here's your situation:

GOAL: {self.goal if self.goal else 'Not set'}

Current Location: {room if room else 'Unknown'}
Inventory: {inventory}
Recent Actions: {history_str}

What you see: {clean_obs}

IMPORTANT: Object names must match exactly! For example:
- 'key' is NOT the same as 'latchkey'
- 'door' is NOT the same as 'wooden door'

Available actions:
{chr(10).join(numbered_actions)}

Instructions:
1. Analyze each action and predict its outcome
2. Choose the best action to achieve the goal
3. End your response with "Therefore, I choose: [exact action]"

Your response:"""
            
            if not self.training_mode:
                print("\nDEBUG - Formatted prompt:")
                print(prompt)
                
            return prompt
            
        else:
            # Batch processing - shorter prompts for training
            prompts = []
            for i in range(len(obs)):
                inventory = infos[i].get('inventory', 'nothing')
                room = self._get_room_name(obs[i])
                clean_obs = self._clean_observation(obs[i])
                
                # Note: In batch mode, we don't maintain action history
                numbered_actions = [f"{j+1}) {action}" for j, action in enumerate(valid_actions[i])]
                
                prompt = f"""You are playing a text adventure game. Here's your situation:

GOAL: {self.goals[i] if hasattr(self, 'goals') else 'Not set'}

Current Location: {room if room else 'Unknown'}
Inventory: {inventory}

What you see: {clean_obs}

Available actions:
{chr(10).join(numbered_actions)}

Therefore, I choose:"""  # Shorter prompt for training
                
                prompts.append(prompt)
                
            return prompts
    
    def get_action(self, env, obs, infos, valid_actions, step=0, batch_mode=False):
        """Get next action using LLM. Can handle both single and batched inputs."""
        if not batch_mode:
            # Single input processing - original logic
            clean_obs = self._clean_observation(obs)
            
            if not self.training_mode:
                print("\nDEBUG - get_action called")
                print(f"DEBUG - Current observation (cleaned): {clean_obs[:100]}...")
            
            if self.goal is None or self.goal == "Not set":
                self.goal = self.parse_goal(clean_obs)
                
            if step == 0:
                self.reset()
                
            # Get action with up to 3 attempts
            max_attempts = 3 if not self.training_mode else 1
            
            for attempt in range(max_attempts):
                prompt = self.format_prompt(obs, valid_actions, infos)
                
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    min_length=20,
                    num_beams=5,
                    temperature=0.7,
                    do_sample=True,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                if not self.training_mode:
                    print(f"\nDEBUG - Model full response:\n{full_response}")
                
                # Extract action choice
                choice_match = re.search(r"Therefore,\s*I\s*choose:\s*(.+?)(?:\n|$)", full_response, re.IGNORECASE)
                if choice_match:
                    action = choice_match.group(1).strip()
                    if action in valid_actions:
                        if not self.training_mode:
                            current_room = self._get_room_name(obs)
                            self.action_room_history.append((current_room, action))
                        return action, {}
                
                if not self.training_mode:
                    print(f"DEBUG - Invalid action in attempt {attempt + 1}")
            
            # Fallback
            if not self.training_mode:
                print("WARNING - Failed to get valid action after max attempts")
            return valid_actions[0], {}
            
        else:
            # Batch processing - simplified logic for training
            prompts = self.format_prompt(obs, valid_actions, infos, batch_mode=True)
            
            # Process all prompts in batch
            inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,  # Shorter for training
                num_beams=3,
                temperature=0.7,
                do_sample=True,
                early_stopping=True
            )
            
            responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            actions = []
            
            for i, response in enumerate(responses):
                valid_acts = valid_actions[i]
                # Take first valid action mentioned in response
                action = next((act for act in valid_acts if act in response), valid_acts[0])
                actions.append(action)
            
            return actions, {}

    def update_state_after_action(self, obs, reward, done, infos):
        """Update agent's state after an action is taken"""
        print("DEBUG - Updated agent state:")
        print(f"  Step: {self.true_state.get('step_count', 0)}")
        print(f"  Current room: {self._get_room_name(obs)}")
        print(f"  History: {self.true_state.get('history', [])}")
        print(f"  Done: {done}")
        
        self.true_state.update({
            'observation': obs,
            'infos': infos,
            'done': done,
            'last_reward': reward
        })

    def analyze_goals_by_difficulty(self, env_manager, max_difficulty=30):
        """Analyze goals for different difficulty levels"""
        print("\nAnalyzing goals across difficulty levels:")
        print("----------------------------------------")
        
        for difficulty in range(1, max_difficulty + 1):
            try:
                # Create environment
                env = env_manager.get_or_create_env(difficulty)
                obs, _ = env.reset()
                
                # Split observation into lines
                lines = obs.split('\n')
                goal_line = None
                for line in lines:
                    # Skip empty or whitespace-only lines
                    if not line.strip():
                        continue
                    # Skip ASCII art (lines with high proportion of special characters)
                    if sum(c in r'\_|/$[]{}=+-' for c in line) > len(line) * 0.1:
                        continue
                    # First non-ASCII line is our goal
                    goal_line = line.strip()
                    break
                
                print(f"\nDifficulty {difficulty}:")
                print(f"Goal: {goal_line}")
                
                # Clean up
                env.close()
                
            except Exception as e:
                print(f"\nDifficulty {difficulty}:")
                print(f"Error: {str(e)}")
                continue

    def analyze_cleaned_goals_by_difficulty(self, env_manager, max_difficulty=30):
        """Analyze cleaned goals for different difficulty levels
        
        Args:
            env_manager: Environment manager to create games
            max_difficulty: Maximum difficulty level to analyze
        """
        print("\nAnalyzing cleaned goals across difficulty levels:")
        print("----------------------------------------------")
        
        for difficulty in range(1, max_difficulty + 1):
            try:
                # Create environment
                env = env_manager.get_or_create_env(difficulty)
                obs, _ = env.reset()
                
                # Parse and clean goal
                parsed_goal = self.parse_goal(obs)
                #cleaned_goal = self._clean_goal_text(original_goal)
                
                print(f"\nDifficulty {difficulty}:")
                print(f"Parsed Goal: {parsed_goal}")
                #print(f"Cleaned:  {cleaned_goal}")
                
                # Clean up
                env.close()
                
            except Exception as e:
                print(f"\nDifficulty {difficulty}:")
                print(f"Error: {str(e)}")
                continue

    def prepare_for_training(self):
        """Prepare agent for training mode"""
        self.training_mode = True
        # Clear model and tokenizer to free memory
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def update_state(self, step_count):
        """Update agent's state tracking"""
        if not hasattr(self, 'true_state'):
            self.true_state = {}
        self.true_state['step_count'] = step_count
