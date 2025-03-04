import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re

class TextWorldLLMAgent:
    def __init__(self, config, training_mode=False, model_path=None):
        """
        Initialize LLM-based TextWorld agent
        
        Args:
            config: Configuration object containing model settings
            training_mode: If True, enables batch processing and disables debug output
            model_path: Optional path to a fine-tuned model checkpoint
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
            if model_path:
                print(f"Loading fine-tuned model from {model_path}...")
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Load the base model and tokenizer
                    self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
                    self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
                    
                    # Add special tokens for command and room tags
                    special_tokens = {
                        'additional_special_tokens': ['<command>', '</command>', '<room>', '</room>']
                    }
                    self.tokenizer.add_special_tokens(special_tokens)
                    
                    # Resize the model's token embeddings to account for the new tokens
                    self.model.resize_token_embeddings(len(self.tokenizer))
                    
                    # Load the fine-tuned weights
                    checkpoint = torch.load(model_path, map_location='cpu')
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    print(f"Using device: {self.device}")
                    self.model.to(self.device)
                    self.model.eval()  # Set to evaluation mode
                    
                except Exception as e:
                    raise RuntimeError(f"Failed to load fine-tuned model: {str(e)}")
            else:
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
        # Split into lines and clean
        lines = [line.strip() for line in initial_obs.split('\n') 
                if line.strip() and sum(c in r'\_|/$[]{}=+-' for c in line) <= len(line) * 0.1]
        
        if not lines:
            return "Unknown"
        
        # Get the first non-ASCII line as the raw goal line
        goal_line = None
        for line in lines:
            if line.strip():
                goal_line = line.strip()
                self.raw_goal_line = line
                break
        
        if not goal_line:
            return "Unknown"
        
        # Split goal line into sentences using multiple delimiters
        sentences = re.split(r'[.!?]+\s*', goal_line.strip())
        sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty sentences
        
        if len(sentences) > 1:
            # If multiple sentences, remove the first one and join with periods
            cleaned_goal = '. '.join(sentences[1:]) + '.'
        else:
            # If only one sentence, keep it and add period if needed
            cleaned_goal = sentences[0] + ('.' if not sentences[0].endswith('.') else '')
        
        # Final cleanup (only removing leading/trailing spaces and common phrases)
        cleaned_goal = cleaned_goal.strip()
        if cleaned_goal.lower().startswith("i need you to "):
            cleaned_goal = cleaned_goal[len("i need you to "):].strip()
            # Add period if it was removed
            if not cleaned_goal.endswith('.'):
                cleaned_goal += '.'
        
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

    def filter_valid_actions(self, valid_actions):
        """Filter out 'look' and 'inventory' actions"""
        return [action for action in valid_actions if action not in ['look', 'inventory']]

    def format_prompt(self, obs, valid_actions, room):
        """Format prompt for LLM using the same format as SFT training"""
        # Clean observation
        clean_obs = self._clean_observation(obs)
        
        # Format action history
        history_str = "None"
        if self.action_room_history:
            history_items = []
            for i, (r, a) in enumerate(self.action_room_history[-5:], 1):  # Show last 5 actions
                history_items.append(f"{i}. In {r}, I chose {a}")
            history_str = "\n".join(history_items)
        
        # Filter actions
        filtered_actions = self.filter_valid_actions(valid_actions)
        
        # Get inventory if available
        inventory_str = self.true_state.get('inventory', "No inventory information available")
        
        prompt = f"""You are playing a text adventure game. Analyze this game state and give a response formatted as requested:

Game State:
Goal: {self.goal if self.goal else "Unknown"}
Location: {room}
Observation: {clean_obs}
Inventory: {inventory_str}
Previous actions:
{history_str}
Currently available actions: {filtered_actions}

Generate a *concise* response in the following format:

A) One sentence reasoning about the game state, which actions seem relevant, and what those actions might achieve.

B) Then, state your chosen action - Make sure it is in the available actions list:
Therefore, I choose: <command>[exact action]</command>

C) Then, state your prediction for the room you will be in after taking this action (say "New Room" if you think it will be a room you haven't been in yet):
I predict that I will be in room: <room>[room name]</room>

Your response:"""

        return prompt

    def get_action(self, env, obs, infos, valid_actions, step=0, batch_mode=False):
        """Get next action using LLM. Can handle both single and batched inputs."""
        if not batch_mode:
            # Single input processing - original logic
            clean_obs = self._clean_observation(obs)
            
            #if not self.training_mode:
            #    print("\nDEBUG - get_action called")
            #    print(f"DEBUG - Current observation (cleaned): {clean_obs[:100]}...")
            
            if self.goal is None or self.goal == "Not set":
                self.goal = self.parse_goal(clean_obs)
                
            if step == 0:
                self.reset()
                
            # Get action with up to 3 attempts
            max_attempts = 3 if not self.training_mode else 1
            format_failures = 0
            
            for attempt in range(max_attempts):
                prompt = self.format_prompt(obs, valid_actions, self._get_room_name(obs))
                
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
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                # Remove <pad> token if present
                full_response = full_response.replace("<pad>", "").strip()
                
                if not self.training_mode:
                    print(f"\nDEBUG - Model full response:\n{full_response}")
                
                # Check format using the check_format function
                format_check = check_format(full_response)
                
                if format_check["has_command_tags"] and format_check["has_room_tags"]:
                    action = format_check["command"]
                    predicted_room = format_check["room"]
                    
                    # Verify action is valid
                    if action in valid_actions:
                        if not self.training_mode:
                            current_room = self._get_room_name(obs)
                            self.action_room_history.append((current_room, action))
                        
                        # Track format success in true_state
                        self.true_state.update({
                            'format_check_passed': True,
                            'format_failures': format_failures,
                            'predicted_room': predicted_room
                        })
                        
                        return action, {"predicted_room": predicted_room, "format_check": format_check}
                    else:
                        if not self.training_mode:
                            print(f"DEBUG - Action '{action}' not in valid actions")
                        format_failures += 1
                else:
                    if not self.training_mode:
                        print(f"DEBUG - Format check failed: {format_check}")
                    format_failures += 1
                    
                    # Try fallback extraction methods
                    # 1. Try old pattern with "Therefore, I choose:"
                    choice_match = re.search(r"Therefore,\s*I\s*choose:\s*(.+?)(?:\n|$)", full_response, re.IGNORECASE)
                    if choice_match:
                        action = choice_match.group(1).strip()
                        if action in valid_actions:
                            if not self.training_mode:
                                current_room = self._get_room_name(obs)
                                self.action_room_history.append((current_room, action))
                            
                            # Track format failure in true_state
                            self.true_state.update({
                                'format_check_passed': False,
                                'format_failures': format_failures,
                                'predicted_room': "Unknown"
                            })
                            
                            return action, {"format_check_passed": False}
            
            # All attempts failed, use fallback
            if not self.training_mode:
                print("WARNING - Failed to get valid action after max attempts")
            
            # Track format failure in true_state
            self.true_state.update({
                'format_check_passed': False,
                'format_failures': max_attempts,
                'predicted_room': "Unknown"
            })
            
            return valid_actions[0], {"format_check_passed": False}
            
        else:
            # Batch processing - simplified logic for training
            prompts = []
            for i in range(len(obs)):
                prompt = self.format_prompt(obs[i], valid_actions[i], self._get_room_name(obs[i]))
                prompts.append(prompt)
            
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
            
            responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
            actions = []
            format_checks = []
            
            for i, response in enumerate(responses):
                valid_acts = valid_actions[i]
                response = response.replace("<pad>", "").strip()
                
                # Check format
                format_check = check_format(response)
                format_checks.append(format_check)
                
                if format_check["has_command_tags"] and format_check["command"] in valid_acts:
                    actions.append(format_check["command"])
                else:
                    # Fallback: take first valid action mentioned in response
                    action = next((act for act in valid_acts if act in response), valid_acts[0])
                    actions.append(action)
            
            return actions, {"format_checks": format_checks}

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





def check_format(text):
    """
    Check if the text follows the expected format with command and room tags.
    
    Args:
        text: Text to check
        
    Returns:
        dict: Format check results including:
            - has_command_tags: Whether text has <command> tags
            - has_room_tags: Whether text has <room> tags
            - command: Extracted command (or "None" if not found)
            - room: Extracted room (or "None" if not found)
    """
    # Check for A/B/C format (not strict about line starts)
    has_section_a = bool(re.search(r'A\)', text, re.IGNORECASE))
    has_section_b = bool(re.search(r'B\)', text, re.IGNORECASE))
    has_section_c = bool(re.search(r'C\)', text, re.IGNORECASE))
    
    # Check for command and room tags
    has_command_tags = '<command>' in text and '</command>' in text
    has_room_tags = '<room>' in text and '</room>' in text
    
    # Extract command and room, handling the extra spaces
    command_match = re.search(r"<command>\s*(.+?)\s*</command>", text)
    command = command_match.group(1).strip() if command_match else "None"
    
    room_match = re.search(r"<room>\s*(.+?)\s*</room>", text)
    room = room_match.group(1).strip() if room_match else "None"
    
    return {
        "format_correct": has_section_a and has_section_b and has_section_c,
        "has_command_tags": has_command_tags,
        "has_room_tags": has_room_tags,
        "command": command,
        "room": room
    }