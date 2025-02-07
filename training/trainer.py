import torch
import numpy as np
from collections import defaultdict
from environment.environment_manager import DifficultyEnvironmentManager


class ExperienceBuffer:
    """Stores and manages training experiences"""
    def __init__(self, config):
        self.config = config
        
        # Main buffers
        self.regular_buffer = []  # All episodes
        self.success_buffer = []  # Only successful episodes
        
        # Priority tracking
        self.priorities = []  # Priority weights for regular buffer
        self.success_priorities = []  # Priority weights for success buffer
        
        # Difficulty tracking
        self.difficulty_counts = defaultdict(int)  # Count episodes per difficulty
        self.success_difficulty_counts = defaultdict(int)
        
        # Buffer limits
        self.max_size = config.training_config.buffer_size
        self.max_success_size = config.training_config.success_buffer_size
    
    def add_episode(self, episode, priority=1.0):
        """Add episode to appropriate buffer"""
        if episode.success:
            # Add to success buffer
            if len(self.success_buffer) >= self.max_success_size:
                # Remove oldest episode of same difficulty if buffer full
                for i, old_ep in enumerate(self.success_buffer):
                    if old_ep.difficulty == episode.difficulty:
                        self.success_buffer.pop(i)
                        self.success_priorities.pop(i)
                        self.success_difficulty_counts[episode.difficulty] -= 1
                        break
            
            self.success_buffer.append(episode)
            self.success_priorities.append(priority)
            self.success_difficulty_counts[episode.difficulty] += 1
            
        # Always add to regular buffer
        if len(self.regular_buffer) >= self.max_size:
            # Remove oldest episode of same difficulty
            for i, old_ep in enumerate(self.regular_buffer):
                if old_ep.difficulty == episode.difficulty:
                    self.regular_buffer.pop(i)
                    self.priorities.pop(i)
                    self.difficulty_counts[episode.difficulty] -= 1
                    break
        
        self.regular_buffer.append(episode)
        self.priorities.append(priority)
        self.difficulty_counts[episode.difficulty] += 1
    
    def sample_batch(self, batch_size):
        """Sample a batch of episodes using priorities and difficulty window"""
        if not self.regular_buffer:
            return None
            
        # Get current difficulty window
        difficulties = sorted(self.difficulty_counts.keys())
        min_diff = max(min(difficulties), self.config.training_config.min_difficulty)
        max_diff = min(max(difficulties), self.config.training_config.max_difficulty)
        window_size = min(self.config.training_config.window_size, max_diff - min_diff + 1)
        
        # Filter episodes in current window
        regular_indices = [
            i for i, ep in enumerate(self.regular_buffer)
            if min_diff <= ep.difficulty <= min_diff + window_size - 1
        ]
        success_indices = [
            i for i, ep in enumerate(self.success_buffer)
            if min_diff <= ep.difficulty <= min_diff + window_size - 1
        ]
        
        if not regular_indices:
            return None
            
        # Calculate sampling weights
        regular_weights = np.array([self.priorities[i] for i in regular_indices])
        regular_weights = regular_weights ** self.config.training_config.priority_alpha
        regular_weights = regular_weights / regular_weights.sum()
        
        if success_indices:
            success_weights = np.array([self.success_priorities[i] for i in success_indices])
            success_weights = success_weights ** self.config.training_config.priority_alpha
            success_weights = success_weights / success_weights.sum()
        
        # Sample episodes
        n_success = min(len(success_indices), batch_size // 4)  # 25% from success buffer
        n_regular = batch_size - n_success
        
        regular_samples = []
        if n_regular > 0:
            sampled_indices = np.random.choice(
                regular_indices, 
                size=n_regular, 
                p=regular_weights,
                replace=True
            )
            regular_samples = [self.regular_buffer[i] for i in sampled_indices]
        
        success_samples = []
        if n_success > 0 and success_indices:
            sampled_indices = np.random.choice(
                success_indices,
                size=n_success,
                p=success_weights,
                replace=True
            )
            success_samples = [self.success_buffer[i] for i in sampled_indices]
        
        return regular_samples + success_samples
    
    def update_priorities(self, indices, priorities):
        """Update priority weights for episodes"""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
    
    def get_stats(self):
        """Return buffer statistics"""
        return {
            'regular_size': len(self.regular_buffer),
            'success_size': len(self.success_buffer),
            'difficulty_counts': dict(self.difficulty_counts),
            'success_difficulty_counts': dict(self.success_difficulty_counts),
            'mean_priority': np.mean(self.priorities) if self.priorities else 0,
            'mean_success_priority': np.mean(self.success_priorities) if self.success_priorities else 0
        }

class Episode:
    """Container for episode data"""
    def __init__(self, difficulty, max_steps):
        # Episode metadata
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.success = False
        
        # Trajectory data
        self.states = []  # List of observations/states
        self.actions = []  # Actions taken
        self.rewards = []  # Rewards received
        self.valid_actions = []  # Valid actions at each step
        self.rooms = []  # Rooms visited
        
        # MCTS statistics
        self.mcts_policies = []  # Action probabilities from MCTS
        self.visit_counts = []  # Visit counts for each action
        self.q_values = []  # Q-values from search
        self.room_predictions = []  # Room predictions at each step
        
        # Training targets (computed later)
        self.returns = None
        self.values = None
    
    def add_step(self, state, action, reward, valid_actions, mcts_stats):
        """Add a step to the episode"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.valid_actions.append(valid_actions)
        self.rooms.append(state['current_room'])
        
        # Store MCTS statistics
        self.mcts_policies.append(mcts_stats['prior_probs'])
        self.visit_counts.append(mcts_stats['visit_counts'])
        self.q_values.append(mcts_stats['q_values'])
        self.room_predictions.append(mcts_stats['room_predictions'])
    
    def compute_returns(self, discount_factor=1.0):
        """Calculate Monte Carlo returns for each step"""
        self.returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + discount_factor * R
            self.returns.insert(0, R)
    
    def to_training_format(self, text_processor):
        """Convert episode data to training tensors"""
        # Process text data
        processed_states = [
            text_processor.process_state(state['observation'])
            for state in self.states
        ]
        
        processed_actions = [
            text_processor.process_actions(actions)
            for actions in self.valid_actions
        ]
        
        # Convert MCTS policies to tensors
        policy_targets = []
        for valid_acts, visit_counts in zip(self.valid_actions, self.visit_counts):
            # Normalize visit counts to get policy target
            total_visits = sum(visit_counts.values())
            policy = torch.zeros(len(valid_acts))
            for i, act in enumerate(valid_acts):
                policy[i] = visit_counts.get(act, 0) / max(total_visits, 1)
            policy_targets.append(policy)
        
        return {
            'states': processed_states,
            'actions': processed_actions,
            'policy_targets': policy_targets,
            'value_targets': torch.tensor(self.returns),
            'room_targets': self.rooms[1:],  # Next room for each state
            'metadata': {
                'difficulty': self.difficulty,
                'success': self.success,
                'length': len(self.states)
            }
        }
    
    def save_raw_text(self, path):
        """Save raw text of episode for analysis"""
        if not self.success:
            return
            
        with open(path, 'w') as f:
            f.write(f"Difficulty: {self.difficulty}\n")
            f.write(f"Steps: {len(self.states)}\n\n")
            
            for i, (state, action) in enumerate(zip(self.states, self.actions)):
                f.write(f"Step {i}:\n")
                f.write(f"Room: {state['current_room']}\n")
                f.write(f"Observation: {state['observation']}\n")
                f.write(f"Action: {action}\n")
                f.write(f"Reward: {self.rewards[i]}\n\n")

class TextWorldTrainer:
    """Main training orchestrator"""
    def __init__(self, config, agent):
        self.config = config
        self.agent = agent
        self.buffer = ExperienceBuffer(config)
        self.env_manager = DifficultyEnvironmentManager(config)
        
        # Track current difficulty window
        self.current_difficulties = range(
            config.training_config.min_difficulty,
            config.training_config.min_difficulty + config.training_config.window_size
        )
        
        # Training metrics
        self.metrics = defaultdict(list)
        
    def initialize_environments(self):
        """Create environments for current difficulty window"""
        print("Initializing environments for difficulties:", self.current_difficulties)
        for diff in self.current_difficulties:
            self.env_manager.get_or_create_env(diff)
    
    def collect_experience(self, episodes_per_difficulty=None):
        """Collect experience by playing episodes"""
        if episodes_per_difficulty is None:
            episodes_per_difficulty = self.config.training_config.episodes_per_difficulty
        
        success_count = 0
        
        # Collect episodes for each difficulty in the current window
        for difficulty in self.current_difficulties:
            print(f"\nCollecting {episodes_per_difficulty} episodes for difficulty {difficulty}")
            
            for ep_num in range(episodes_per_difficulty):
                print(f"\nStarting episode {ep_num + 1} for difficulty {difficulty}")
                
                # Initialize episode
                episode = Episode(difficulty, self.config.max_steps)
                env = self.env_manager.get_or_create_env(difficulty)
                obs, infos = env.reset()
                done = False
                
                # Play episode
                for step in range(self.config.max_steps):
                    if done:
                        break
                        
                    valid_actions = [
                        a for a in infos["admissible_commands"]
                        if a.lower() not in ['inventory', 'look']
                    ]
                    
                    # Get action from agent
                    action, mcts_stats = self.agent.get_action(
                        env,
                        obs,
                        infos,
                        valid_actions,
                        step
                    )
                    
                    if action is None:
                        print("DEBUG - Agent returned None action (terminal state)")
                        break
                    
                    # Take action and update agent's state
                    next_obs, reward, done, next_infos = env.step(action)
                    self.agent.update_state_after_action(next_obs, reward, done, next_infos)
                    
                    # Store step
                    episode.add_step(
                        state={
                            'observation': obs,
                            'current_room': self.agent._get_room_name(obs),
                            'history': episode.actions.copy(),
                            'done': done
                        },
                        action=action,
                        reward=reward,
                        valid_actions=valid_actions,
                        mcts_stats=mcts_stats
                    )
                    
                    if done:
                        episode.success = (reward > 0)
                        if episode.success:
                            success_count += 1
                        print(f"DEBUG - Episode finished: success={episode.success}")
                        break
                        
                    obs, infos = next_obs, next_infos
                
                # Compute returns and add to buffer
                episode.compute_returns()
                self.buffer.add_episode(episode)
    
    def train_networks(self, num_batches):
        """Train networks on collected experience"""
        print("\nTraining networks...")
        self.agent.policy_network.train()
        self.agent.value_network.train()
        self.agent.room_prediction_network.train()
        
        total_policy_loss = 0
        total_value_loss = 0
        total_room_loss = 0
        
        for batch_num in range(num_batches):
            # Sample batch of episodes
            episodes = self.buffer.sample_batch(self.config.training_config.batch_size)
            if not episodes:
                continue
                
            # Convert episodes to training format
            batch_data = [
                ep.to_training_format(self.agent.text_processor)
                for ep in episodes
            ]
            
            # Zero gradients
            self.agent.optimizer.zero_grad()
            
            # Compute losses
            policy_loss = self._compute_policy_loss(batch_data)
            value_loss = self._compute_value_loss(batch_data)
            room_loss = self._compute_room_prediction_loss(batch_data)
            
            # Combined loss
            total_loss = (
                policy_loss +
                self.config.training_config.value_loss_weight * value_loss +
                self.config.training_config.room_loss_weight * room_loss
            )
            
            # Backward pass and optimize
            total_loss.backward()
            self.agent.optimizer.step()
            
            # Track metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_room_loss += room_loss.item()
            
            if (batch_num + 1) % 10 == 0:
                print(f"Batch {batch_num + 1}/{num_batches}")
                print(f"Policy loss: {total_policy_loss / (batch_num + 1):.4f}")
                print(f"Value loss: {total_value_loss / (batch_num + 1):.4f}")
                print(f"Room loss: {total_room_loss / (batch_num + 1):.4f}")
        
        # Store metrics
        self.metrics['policy_loss'].append(total_policy_loss / num_batches)
        self.metrics['value_loss'].append(total_value_loss / num_batches)
        self.metrics['room_loss'].append(total_room_loss / num_batches)
        
        # Set networks back to eval mode
        self.agent.policy_network.eval()
        self.agent.value_network.eval()
        self.agent.room_prediction_network.eval()
    
    def evaluate(self, num_episodes=10):
        """Evaluate agent on current difficulty window"""
        print("\nEvaluating agent...")
        results = {}
        
        for difficulty in self.current_difficulties:
            env = self.env_manager.get_or_create_env(difficulty)
            success_count = 0
            
            for _ in range(num_episodes):
                obs, infos = env.reset()
                done = False
                
                while not done:
                    valid_actions = [
                        a for a in infos["admissible_commands"]
                        if a.lower() not in ['inventory', 'look']
                    ]
                    
                    action, _ = self.agent.get_action(env, obs, infos, valid_actions)
                    obs, reward, done, infos = env.step(action)
                    self.agent.update_state_after_action(obs, reward, done, infos)

                    
                    if done and reward > 0:
                        success_count += 1
            
            results[difficulty] = success_count / num_episodes
            print(f"Difficulty {difficulty} success rate: {results[difficulty]:.2f}")
            
        return results
    
    def train(self, num_epochs):
        """Main training loop"""
        print("Starting training...")
        self.initialize_environments()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Collect experience
            self.collect_experience()
            
            # Train networks
            self.train_networks(self.config.training_config.batches_per_epoch)
            
            # Evaluate
            if (epoch + 1) % self.config.training_config.eval_frequency == 0:
                results = self.evaluate()
                
                # Store metrics
                for diff, rate in results.items():
                    self.metrics[f'eval_success_rate_diff_{diff}'].append(rate)
                
                # Save checkpoint
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
            
            # Consider advancing difficulty window
            if self.should_advance_difficulty():
                self.advance_difficulty_window()
    
    def should_advance_difficulty(self):
        """Check if we should advance to harder difficulties"""
        if len(self.metrics['eval_success_rate_diff_' + str(min(self.current_difficulties))]) < 3:
            return False
            
        # Get last 3 evaluation results for easiest difficulty
        recent_results = self.metrics['eval_success_rate_diff_' + str(min(self.current_difficulties))][-3:]
        return all(r >= self.config.training_config.advance_threshold for r in recent_results)
    
    def advance_difficulty_window(self):
        """Move difficulty window up"""
        print("\nAdvancing difficulty window...")
        old_min = min(self.current_difficulties)
        old_max = max(self.current_difficulties)
        
        # Update current difficulties
        self.current_difficulties = range(
            old_min + 1,
            old_min + 1 + self.config.training_config.window_size
        )
        
        # Initialize new environments
        for diff in self.current_difficulties:
            if diff > old_max:
                self.env_manager.get_or_create_env(diff)
    
    def save_checkpoint(self, filename):
        """Save training state"""
        torch.save({
            'agent_state': self.agent.state_dict(),
            'optimizer_state': self.agent.optimizer.state_dict(),
            'buffer_state': self.buffer.get_stats(),
            'metrics': dict(self.metrics),
            'current_difficulties': self.current_difficulties,
            'config': self.config
        }, filename)
        print(f"\nSaved checkpoint to {filename}")