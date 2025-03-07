import re
import copy
import torch
import random

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
        self.room_prediction_correct = False
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
        if hasattr(agent, 'goal'):
            cloned_agent.goal = agent.goal
        if hasattr(agent, 'last_known_room'):
            cloned_agent.last_known_room = agent.last_known_room
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
        
        # Use the agent's method 
        action_info = self.agent.extract_action_from_completion(completion, valid_actions)
        
        # Extract room prediction (this part is not in the agent's method)
        room_prediction = None
        format_check_result = self.agent.check_format(completion)
        if format_check_result["has_room_tags"]:
            room_match = re.search(r'<room>(.*?)</room>', completion, re.DOTALL)
            if room_match:
                room_prediction = room_match.group(1).strip()
        
        # Add room prediction to the action info
        action_info["room_prediction"] = room_prediction
        
        # Update format check status
        self.format_check_passed = action_info["format_check_passed"]
        
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
            
            # If still no action found, use the first valid action
            if not action_info.get('action') and valid_actions:
                action_info['action'] = valid_actions[0]
                print(f"No valid action found in completion, using: {valid_actions[0]}")
        
        return action_info
    
    def run(self, max_steps=10, gamma=0.99):
        """Run the rollout and return the reward"""
        try:
            # Reset environment
            obs, infos = self.env.reset()
            
            # Replay action history to get to current state
            for past_action in self.action_history:
                obs, _, _, infos = self.env.step(past_action)
            
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
                    
                    # Store the completion token count
                    self.completion_token_count = action_info.get('completion_token_count', 0)
                    
                    # Check if room prediction is correct
                    if action_info.get('room_prediction') and self.action is not None:
                        try:
                            # Take the action to see what room we end up in
                            temp_obs, _, temp_done, _ = self.env.step(self.action)
                            if not temp_done:
                                next_room = self.agent._get_room_name(temp_obs)
                                
                                # Handle the case where next_room is None (room didn't change)
                                if next_room is None:
                                    # If room name not found in observation, assume we're still in the same room
                                    next_room = self.agent.last_known_room
                                
                                # Now check if the prediction is correct (safely)
                                if next_room is not None and action_info.get('room_prediction') is not None:
                                    self.room_prediction_correct = (next_room.lower() == action_info.get('room_prediction').lower())
                                else:
                                    # If either is None, we can't verify the prediction
                                    self.room_prediction_correct = False
                            
                            # Reset back to before taking the action
                            obs, infos = self.env.reset()
                            for past_action in self.action_history:
                                obs, _, _, infos = self.env.step(past_action)
                        except Exception as e:
                            print(f"Error checking room prediction: {e}")
                            # If there's an error, just continue without checking room prediction
                            self.room_prediction_correct = False
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
            obs, reward, done, infos = self.env.step(self.action)
            self.reward += reward
            self.steps += 1
            
            # If done after first action, return the reward
            if done:
                self.done = True
                self.success = (reward > 0)
                return self.reward
            
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
        
        # Format penalty (only apply penalty, no reward for correct format)
        if not self.format_check_passed:
            total_reward += config.format_failure_penalty
        
        # Room prediction penalty (only apply penalty, no reward for correct prediction)
        if hasattr(config, 'room_prediction_penalty') and config.room_prediction_penalty < 0:
            # Check if we have a room prediction and it's incorrect
            if hasattr(self, 'room_prediction_correct') and not self.room_prediction_correct:
                total_reward += config.room_prediction_penalty
        
        return total_reward




