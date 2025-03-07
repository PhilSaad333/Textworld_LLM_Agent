import re
import copy
import torch

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
        """Create a copy of the agent for rollout"""
        # Create a new agent with the same configuration
        cloned_agent = type(agent)(agent.config, use_map=getattr(agent, 'use_map', False))
        
        # Copy over the model, tokenizer, and device
        cloned_agent.model = self.model
        cloned_agent.tokenizer = self.tokenizer
        cloned_agent.device = self.device
        
        # Copy the agent's state
        if hasattr(agent, 'true_state'):
            cloned_agent.true_state = copy.deepcopy(agent.true_state)
        
        # Copy the map tool if it exists
        if hasattr(agent, 'map_tool') and agent.map_tool:
            cloned_agent.map_tool = copy.deepcopy(agent.map_tool)
            
        return cloned_agent
    
    def extract_action_from_completion(self, completion, valid_actions):
        """Extract action and format check from a completion"""
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
        
        # Count tokens in the completion
        completion_token_count = len(self.tokenizer.encode(completion))
        action_info["completion_token_count"] = completion_token_count
        
        return action_info
    
    def run(self, max_steps=10, gamma=0.99):
        """Run the rollout and return the reward
        
        Args:
            max_steps: Maximum number of steps to run the rollout for
            gamma: Discount factor for future rewards
            
        Returns:
            The total discounted reward from the rollout
        """
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
        
        # If we have a completion but no action, extract the action
        if self.completion and not self.action:
            action_info = self.extract_action_from_completion(self.completion, valid_actions)
            self.action = action_info.get('action', None)
            
            # Store the completion token count
            self.completion_token_count = action_info.get('completion_token_count', 0)
            
            # Check if room prediction is correct
            if action_info.get('room_prediction'):
                # Take the action to see what room we end up in
                temp_obs, _, temp_done, _ = self.env.step(self.action)
                if not temp_done:
                    next_room = self.agent._get_room_name(temp_obs)
                    self.room_prediction_correct = (next_room.lower() == action_info.get('room_prediction').lower())
                
                # Reset back to before taking the action
                obs, infos = self.env.reset()
                for past_action in self.action_history:
                    obs, _, _, infos = self.env.step(past_action)
        
        # If we still don't have an action, return a penalty
        if not self.action or self.action not in valid_actions:
            self.reward = -1.0  # Penalty for invalid action
            return self.reward
        
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
            
            # Get action from agent
            action, _ = self.agent.get_action_fast(
                self.env, obs, infos, valid_actions, self.steps
            )
            
            # Take action in environment
            obs, step_reward, done, infos = self.env.step(action)
            
            # Update reward (with discount)
            self.reward += (gamma ** self.steps) * step_reward
            
            # Update agent state
            self.agent.update_state_after_action(obs, step_reward, done, infos)
            
            self.steps += 1
        
        self.done = done
        self.success = (done and self.reward > 0)
        
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




