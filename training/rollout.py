import re
import copy
import torch
import random
import numpy as np

class Rollout:
    """Performs a rollout from a given state using a specific action"""
    
    def __init__(self, model, tokenizer, device, env, agent, action_history, completion=None, action=None):
        """Initialize a rollout
        
        Args:
            model: The language model to use for generating actions
            tokenizer: The tokenizer for the model
            device: The device to run the model on
            env: The environment to run the rollout in
            agent: The agent to use for the rollout
            action_history: List of actions taken so far in the episode
            completion: The completion to extract the action from (optional)
            action: The action to take (optional, if not provided will be extracted from completion)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.env = env
        self.agent = self._clone_agent(agent)
        self.action_history = action_history.copy()
        self.completion = completion
        self.action = action
        
        # Rollout results
        self.reward = 0
        self.steps = 0
        self.done = False
        self.format_check_passed = False
        self.room_prediction_correct = None
        self.success = False
        
    def _clone_agent(self, agent):
        """Create a copy of the agent for rollout without reinitializing the model"""
        # Create a new agent with training_mode=True to avoid loading a new model
        cloned_agent = type(agent)(agent.config, training_mode=True, use_map=getattr(agent, 'use_map', False))
        
        # Directly set the model, tokenizer, and device from the existing agent
        cloned_agent.model = self.model  # Use the model passed to the Rollout
        cloned_agent.tokenizer = self.tokenizer  # Use the tokenizer passed to the Rollout
        cloned_agent.device = self.device
        
        # Copy the agent's state
        if hasattr(agent, 'goal') and agent.goal is not None:
            cloned_agent.goal = agent.goal
        else:
            # If the original agent doesn't have a goal set, we'll set it later in run()
            cloned_agent.goal = None
            
        if hasattr(agent, 'last_known_room'):
            cloned_agent.last_known_room = agent.last_known_room or "Unknown Room"
        if hasattr(agent, 'known_rooms'):
            cloned_agent.known_rooms = copy.deepcopy(agent.known_rooms)
        if hasattr(agent, 'true_state'):
            cloned_agent.true_state = copy.deepcopy(agent.true_state)
        
        # Copy the map tool if it exists
        if hasattr(agent, 'map_tool') and agent.map_tool:
            cloned_agent.map_tool = copy.deepcopy(agent.map_tool)
        
        # Set training_mode to False after initialization to ensure normal behavior
        cloned_agent.training_mode = False
        
        return cloned_agent
    
    def extract_action_from_completion(self, completion, valid_actions):
        """Extract action and format check from a completion"""
        # Check if completion is None
        if completion is None:
            print("Warning: Received None completion")
            return {
                "action": valid_actions[0] if valid_actions else "look",
                "format_check_passed": False,
                "command": None,
                "room_prediction": None,
                "completion_token_count": 0
            }
        
        # Use the agent's method to extract action
        try:
            action_info = self.agent.extract_action_from_completion(completion, valid_actions)
        except Exception as e:
            print(f"Error in agent's extract_action_from_completion: {e}")
            action_info = {
                "action": None,
                "format_check_passed": False,
                "command": None
            }
        
        # Get format check result directly from agent
        try:
            format_check_result = self.agent.check_format(completion)
            # Store the full format check result for more detailed reward calculation
            self.format_check_result = format_check_result
            
            # Extract room prediction directly from format check result
            room_prediction = format_check_result["room"]
        except Exception as e:
            print(f"Error extracting room prediction: {e}")
            room_prediction = None
            self.format_check_result = {
                "has_command_tags": False,
                "has_room_tags": False,
                "command": None,
                "room": None
            }
        
        # Add room prediction to the action info
        action_info["room_prediction"] = room_prediction
        
        # Update format check status
        self.format_check_passed = action_info.get("format_check_passed", False)
        
        # Count tokens in the completion (safely)
        try:
            completion_token_count = len(self.tokenizer.encode(completion))
            action_info["completion_token_count"] = completion_token_count
        except Exception as e:
            print(f"Error counting tokens: {e}")
            action_info["completion_token_count"] = 0
        
        # Ensure we have a valid action
        if not action_info.get('action') and valid_actions:
            # Try to find any valid action mentioned in the completion
            for action in valid_actions:
                if action.lower() in completion.lower():
                    action_info['action'] = action
                    print(f"Found valid action in completion: {action}")
                    break
            
            # If still no action found, use a random valid action
            if not action_info.get('action') and valid_actions:
                action_info['action'] = np.random.choice(valid_actions)
                print(f"No valid action found in completion, using random action: {action_info['action']}")
        
        return action_info
    
    def run(self, max_steps=10, gamma=0.99):
        """Run the rollout and return the reward"""
        try:
            # Reset environment
            obs, infos = self.env.reset()
            
            # Replay action history to get to current state
            for past_action in self.action_history:
                obs, _, _, infos = self.env.step(past_action)
            
            # Ensure the agent has a goal set
            if self.agent.goal is None or self.agent.goal == "Not set":
                self.agent.goal = self.agent.parse_goal(obs)
                print(f"Rollout: Set goal: {self.agent.goal}")
            
            # Get valid actions
            valid_actions = [
                a for a in infos["admissible_commands"]
                if a.lower() not in ['inventory', 'look']
            ]
            
            # If no valid actions, add a default one
            if not valid_actions:
                valid_actions = ["look"]
            
            # If we have a completion but no action, extract the action
            if self.completion and not self.action:
                try:
                    action_info = self.extract_action_from_completion(self.completion, valid_actions)
                    self.action = action_info.get('action', None)
                    
                    # Store the completion token count (safely)
                    self.completion_token_count = action_info.get('completion_token_count', 0)
                    
                    # Store the predicted room but don't verify it yet (safely)
                    self.predicted_room = action_info.get('room_prediction', '').lower() if action_info.get('room_prediction') else ''
                    
                    # We'll verify room prediction after taking the action
                    self.room_prediction_correct = None
                    
                except Exception as e:
                    print(f"Error extracting action from completion: {e}")
                    self.action = None
            
            # If we still don't have an action, choose a random valid action
            if not self.action or self.action not in valid_actions:
                if valid_actions:
                    self.action = random.choice(valid_actions)
                    print(f"Using random action: {self.action}")
                else:
                    # If there are no valid actions, use a default action
                    self.action = "look"
                    print("No valid actions available, using 'look'")
                
                # Apply penalty for invalid action
                self.reward = -1.0
            
            # Take the first action
            next_obs, reward, done, next_infos = self.env.step(self.action)
            self.reward += reward
            self.steps += 1
            
            # Extract room prediction from completion
            self.predicted_room = None
            if hasattr(self, 'action_info') and self.action_info:
                self.predicted_room = self.action_info.get('room_prediction')
            
            # If room prediction is missing or not correctly formatted, set to None
            if not self.predicted_room or not hasattr(self, 'format_check_result') or not self.format_check_result.get("has_room_tags", False):
                self.predicted_room = None
                
            # Verify room prediction if possible
            if self.predicted_room is not None:
                try:
                    # Get the room name from the next observation
                    next_room = self.agent._get_room_name(next_obs)
                    
                    if next_room is None:
                        # If room name not found in observation, assume we're still in the same room
                        next_room = self.agent.last_known_room
                    
                    # Special handling for "new room" prediction
                    if self.predicted_room == "new room":
                        # Check if this room has been seen before in the agent's known_rooms
                        if hasattr(self.agent, 'known_rooms') and next_room is not None:
                            # If the room is not in known_rooms, then "new room" is correct
                            self.room_prediction_correct = next_room not in self.agent.known_rooms
                        else:
                            # If we can't verify, default to False
                            self.room_prediction_correct = False
                    else:
                        # For specific room predictions, check exact match
                        if next_room is not None:
                            self.room_prediction_correct = (next_room.lower() == self.predicted_room)
                        else:
                            # If we can't verify, default to False
                            self.room_prediction_correct = False
                except Exception as e:
                    print(f"Error verifying room prediction: {e}")
                    self.room_prediction_correct = False
            
            # If done after first action, return the reward without room prediction penalty
            if done:
                self.done = True
                self.success = (reward > 0)
                # Skip room prediction verification for terminal states
                self.room_prediction_correct = None
                return self.reward
            
            # Update observation and infos for the next step
            obs, infos = next_obs, next_infos
            
            # Continue the rollout
            while not done and self.steps < max_steps:
                # Get valid actions
                valid_actions = [
                    a for a in infos["admissible_commands"]
                    if a.lower() not in ['inventory', 'look']
                ]
                
                # If no valid actions, add a default one
                if not valid_actions:
                    valid_actions = ["look"]
                
                # Get action from agent
                try:
                    action, _ = self.agent.get_action_fast(
                        self.env, obs, infos, valid_actions, self.steps
                    )
                except Exception as e:
                    print(f"Error getting action: {e}")
                    action = valid_actions[0]
                
                # Take action in environment
                obs, step_reward, done, infos = self.env.step(action)
                
                # Update reward (with discount)
                self.reward += (gamma ** self.steps) * step_reward
                
                # Update agent state
                self.agent.update_state_after_action(obs, step_reward, done, infos)
                
                self.steps += 1
            
            self.done = done
            self.success = (done and self.reward > 0)
            
        except Exception as e:
            print(f"Error during rollout: {e}")
            # If there's an error during the rollout, return a penalty
            self.reward = -1.0
            self.done = True
            self.success = False
        
        return self.reward
    
    def compute_total_reward(self, config):
        """Compute the total reward including format and room prediction penalties"""
        total_reward = self.reward  # Base reward from the environment
        
        # Format penalty - only apply for missing command or room tags
        if hasattr(self, 'format_check_result'):
            format_check_result = self.format_check_result
            
            # Apply partial penalties based on what's missing
            if not format_check_result.get("has_command_tags", False):
                total_reward += config.format_penalty / 2
            
            if not format_check_result.get("has_room_tags", False):
                total_reward += config.format_penalty / 2
        elif hasattr(self, 'format_check_passed') and not self.format_check_passed:
            # Fallback if we don't have detailed format check result
            total_reward += config.format_penalty
        
        # Room prediction penalty - apply if prediction is incorrect or missing
        if hasattr(config, 'room_prediction_penalty') and config.room_prediction_penalty < 0:
            # If room prediction is None (missing tags) or explicitly incorrect, apply penalty
            if (hasattr(self, 'format_check_result') and not self.format_check_result.get("has_room_tags", False)) or \
               (hasattr(self, 'room_prediction_correct') and (self.room_prediction_correct is False)):
                total_reward += config.room_prediction_penalty
        
        return total_reward




