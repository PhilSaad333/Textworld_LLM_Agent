"""
Script to fix rollout issues and the get_action_fast method.
Run this in Colab to apply the fixes without restarting the session.
"""

import os
import sys
import importlib
import inspect
import types

# Add the project root to the Python path to import your modules
if not '/content/Textworld_LLM_Agent' in sys.path:
    sys.path.append('/content/Textworld_LLM_Agent')

# Import necessary modules
from agents.textworld_llm_agent import TextWorldLLMAgent
from training.trainer import TextWorldRLTrainer
from training.rollout import Rollout

def fix_get_action_fast():
    """Fix the get_action_fast method to properly return a tuple"""
    original_get_action_fast = TextWorldLLMAgent.get_action_fast
    
    def patched_get_action_fast(self, env, obs, infos, valid_actions, step=0):
        """Patched version of get_action_fast that returns a tuple of (action, {})"""
        try:
            # Call the original method
            action = original_get_action_fast(self, env, obs, infos, valid_actions, step)
            
            # If the action is already a tuple, return it as is
            if isinstance(action, tuple):
                return action
            
            # Otherwise, return a tuple of (action, {})
            return action, {}
        except Exception as e:
            print(f"Error in patched_get_action_fast: {e}")
            # Fallback to a valid action
            if valid_actions:
                return valid_actions[0], {}
            else:
                return "look", {}
    
    # Apply the patch
    TextWorldLLMAgent.get_action_fast = patched_get_action_fast
    print("✓ Fixed get_action_fast method")

def fix_collect_gameplay_data():
    """Fix the collect_gameplay_data method to simplify the rollout process"""
    original_collect_gameplay_data = TextWorldRLTrainer.collect_gameplay_data
    
    def patched_collect_gameplay_data(self, difficulties=None, episodes_per_difficulty=5):
        """Patched version of collect_gameplay_data with simplified rollout process"""
        if difficulties is None:
            difficulties = self.config.difficulties if hasattr(self.config, 'difficulties') and self.config.difficulties else [1, 5, 10]
            
        episodes_per_difficulty = self.config.episodes_per_difficulty if hasattr(self.config, 'episodes_per_difficulty') else episodes_per_difficulty

        # Store episode data as a list of dictionaries
        all_episode_data = []
        
        # Number of completions to generate per prompt
        num_generations = self.grpo_config.num_generations
        
        # Ensure model is on the correct device before starting
        self.model = self.model.to(self.device)

        # Track token statistics
        total_input_tokens = 0
        total_output_tokens = 0
        max_input_tokens = 0
        max_output_tokens = 0
        
        # Process each difficulty level
        for difficulty in difficulties:
            print(f"Collecting data for difficulty {difficulty}")
            # Create environment for this difficulty
            env = self.env_manager.get_or_create_env(difficulty)

            for episode in range(episodes_per_difficulty):
                print(f"Episode {episode+1}/{episodes_per_difficulty}")
                obs, infos = env.reset()
                done = False
                episode_success = False
                episode_data = []
                action_history = []

                # Reset the agent and initialize known_rooms
                self.agent.reset()
                self.agent.known_rooms = set()

                while not done and len(episode_data) < self.config.max_steps:
                    # Get valid actions
                    valid_actions = [
                        a for a in infos["admissible_commands"]
                        if a.lower() not in ['inventory', 'look']
                    ]

                    # Format prompt
                    current_room = self.agent._get_room_name(obs)
                    prompt = self.agent.format_prompt(obs, valid_actions, current_room)
                    
                    # Count input tokens
                    input_tokens = self.tokenizer.encode(prompt)
                    input_token_count = len(input_tokens)
                    total_input_tokens += input_token_count
                    max_input_tokens = max(max_input_tokens, input_token_count)

                    # Log warning if input is close to or exceeds max length
                    if hasattr(self.grpo_config, 'max_prompt_length') and input_token_count >= self.grpo_config.max_prompt_length - 10:
                        print(f"⚠️ Input length ({input_token_count} tokens) close to max ({self.grpo_config.max_prompt_length})")

                    # Generate G completions
                    completions = []
                    completion_token_counts = []
                    
                    # Prepare inputs for generation
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    
                    # Generate G completions
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=self.grpo_config.max_completion_length if hasattr(self.grpo_config, 'max_completion_length') else 128,
                            min_length=20,
                            num_return_sequences=num_generations,
                            num_beams=num_generations,  # Use beam search to get diverse completions
                            do_sample=True,  # Set to True to use temperature
                            temperature=0.7,
                            early_stopping=True,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                    
                    # Process completions
                    for i in range(num_generations):
                        completion = self.tokenizer.decode(outputs.sequences[i], skip_special_tokens=False)
                        completion = completion.replace("<pad>", "").strip()
                        
                        # Ensure completion ends with EOS token
                        if not completion.endswith(self.tokenizer.eos_token):
                            completion += self.tokenizer.eos_token
                        
                        completions.append(completion)
                        print(f"Completion {i+1}: {completion}")
                        
                        # Count output tokens
                        output_tokens = self.tokenizer.encode(completion)
                        output_token_count = len(output_tokens)
                        completion_token_counts.append(output_token_count)
                        total_output_tokens += output_token_count
                        max_output_tokens = max(max_output_tokens, output_token_count)
                        
                        # Log warning if output is close to max length
                        if hasattr(self.grpo_config, 'max_completion_length') and output_token_count >= self.grpo_config.max_completion_length - 10:
                            print(f"⚠️ Output length ({output_token_count} tokens) close to max ({self.grpo_config.max_completion_length})")
                    
                    # Do rollouts for each completion to get rewards
                    rollout_rewards = []
                    
                    for completion_idx, completion in enumerate(completions):
                        # Create a rollout for this completion
                        rollout = Rollout(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            device=self.device,
                            env=env,
                            agent=self.agent,
                            action_history=action_history,
                            completion=completion
                        )
                        
                        # Run the rollout
                        max_rollout_steps = self.config.max_steps - len(action_history) if hasattr(self.config, 'max_steps') else 10
                        
                        # Run the rollout - no special case handling needed
                        rollout.run(max_steps=max_rollout_steps, gamma=self.config.gamma if hasattr(self.config, 'gamma') else 0.99)
                        
                        # Compute total reward including format and room prediction bonuses
                        total_reward = rollout.compute_total_reward(self.config)
                        rollout_rewards.append(total_reward)
                    
                    # Store step data
                    step_data = {
                        "prompt": prompt,
                        "completions": completions,
                        "rewards": rollout_rewards,
                        "completion_token_counts": completion_token_counts
                    }
                    
                    episode_data.append(step_data)
                    
                    # SIMPLIFIED: Directly check format without creating temporary rollouts
                    valid_command_indices = []
                    for i, completion in enumerate(completions):
                        format_check = self.agent.check_format(completion)
                        if format_check["has_command_tags"]:
                            valid_command_indices.append(i)
                    
                    # Strategy 1: If some completions have valid command tags, only sample from those
                    if valid_command_indices:
                        # Choose the completion with the highest reward among those with valid command tags
                        valid_rewards = [rollout_rewards[i] for i in valid_command_indices]
                        best_valid_idx = valid_command_indices[valid_rewards.index(max(valid_rewards))]
                        chosen_completion = completions[best_valid_idx]
                        print(f"Choosing from {len(valid_command_indices)} completions with valid command tags")
                    else:
                        # Strategy 2: If no completions have valid command tags, choose the one with highest reward
                        best_completion_idx = rollout_rewards.index(max(rollout_rewards))
                        chosen_completion = completions[best_completion_idx]
                        print("No completions with valid command tags, choosing based on reward")
                    
                    # SIMPLIFIED: Extract action directly without creating another rollout
                    action_info = self.agent.extract_action_from_completion(chosen_completion, valid_actions)
                    action = action_info.get('action', None)
                    
                    # If action is invalid, choose a random valid action
                    if action is None or action not in valid_actions:
                        action = np.random.choice(valid_actions)
                        print(f"Invalid action, using random action: {action}")
                    
                    # Reset environment to current state
                    obs, infos = env.reset()
                    
                    # Replay action history to get to current state
                    for past_action in action_history:
                        obs, _, _, infos = env.step(past_action)
                    
                    # Take the chosen action
                    next_obs, reward, done, next_infos = env.step(action)
                    
                    # Add action to history
                    action_history.append(action)
                    
                    # Check if episode was completed successfully
                    if done:
                        episode_success = (reward > 0)
                    
                    # Update for next step
                    obs, infos = next_obs, next_infos
                    self.agent.update_state_after_action(obs, reward, done, next_infos)

                    # Explicitly update known_rooms with the current room
                    current_room = self.agent._get_room_name(next_obs)
                    if current_room:
                        self.agent.known_rooms.add(current_room)
                
                # Add episode data to all episodes
                all_episode_data.append({
                    "steps": episode_data,
                    "success": episode_success,
                    "difficulty": difficulty,
                    "num_steps": len(episode_data)
                })
        
        # Prepare dataset for training
        dataset = {
            "prompt": [],
            "completion": [],
            "reward": []
        }
        
        # Flatten episode data into prompt-completion-reward triples
        for episode in all_episode_data:
            for step in episode["steps"]:
                prompt = step["prompt"]
                for i, completion in enumerate(step["completions"]):
                    dataset["prompt"].append(prompt)
                    dataset["completion"].append(completion)
                    dataset["reward"].append(step["rewards"][i])
        
        # Print statistics
        print(f"\nCollected {len(all_episode_data)} episodes with {len(dataset['prompt'])} prompt-completion pairs")
        print(f"Average input tokens: {total_input_tokens / len(dataset['prompt']):.1f}")
        print(f"Average output tokens: {total_output_tokens / len(dataset['completion']):.1f}")
        print(f"Max input tokens: {max_input_tokens}")
        print(f"Max output tokens: {max_output_tokens}")
        
        # Store episode data for later use
        self.episode_data = all_episode_data
        
        return dataset
    
    # Apply the patch
    TextWorldRLTrainer.collect_gameplay_data = patched_collect_gameplay_data
    print("✓ Fixed collect_gameplay_data method")

def fix_rollout_run():
    """Fix the Rollout.run method to handle the get_action_fast return value properly"""
    original_run = Rollout.run
    
    def patched_run(self, max_steps=10, gamma=0.99):
        """Patched version of run that handles the get_action_fast return value properly"""
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
            
            # If we have a completion, extract action from it
            if self.completion:
                action_info = self.extract_action_from_completion(self.completion, valid_actions)
                self.action = action_info.get('action')
                
                # Check if we have a room prediction
                if 'room_prediction' in action_info and action_info['room_prediction']:
                    self.room_prediction = action_info['room_prediction']
                    
                    # Verify if the room prediction is correct
                    try:
                        current_room = self.agent._get_room_name(obs)
                        if current_room:
                            self.room_prediction_correct = (self.room_prediction.lower() == current_room.lower())
                        else:
                            # If we can't verify, default to False
                            self.room_prediction_correct = False
                    except Exception as e:
                        print(f"Error verifying room prediction: {e}")
                        self.room_prediction_correct = False
            
            # If done after first action, return the reward without room prediction penalty
            if self.done:
                return self.reward
            
            # Take the first action if provided
            if self.action and self.action in valid_actions:
                next_obs, reward, done, next_infos = self.env.step(self.action)
                
                # Update reward
                self.reward += reward
                self.steps += 1
                
                # Update agent state
                self.agent.update_state_after_action(next_obs, reward, done, next_infos)
                
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
                    # FIXED: Handle both tuple and non-tuple returns
                    action_result = self.agent.get_action_fast(
                        self.env, obs, infos, valid_actions, self.steps
                    )
                    
                    # Check if action_result is a tuple
                    if isinstance(action_result, tuple):
                        action = action_result[0]
                    else:
                        action = action_result
                        
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
    
    # Apply the patch
    Rollout.run = patched_run
    print("✓ Fixed Rollout.run method")

def apply_all_fixes():
    """Apply all fixes"""
    print("Applying fixes to TextWorld_LLM_Agent...")
    
    # Import necessary modules
    import torch
    import numpy as np
    
    # Fix the get_action_fast method
    fix_get_action_fast()
    
    # Fix the collect_gameplay_data method
    fix_collect_gameplay_data()
    
    # Fix the Rollout.run method
    fix_rollout_run()
    
    print("\nAll fixes applied successfully!")

if __name__ == "__main__":
    apply_all_fixes() 