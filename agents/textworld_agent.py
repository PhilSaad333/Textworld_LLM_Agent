import torch
import math
from transformers import AutoTokenizer
from copy import deepcopy
import torch.nn.functional as F
from models.network import TextWorldPolicyNetwork, TextWorldValueNetwork, RoomPredictionNetwork
import copy
import textworld.gym
import re
import random





class TextWorldAgent:
    def __init__(self, config):
        """
        Initialize TextWorld agent with MCTS-based action selection
        
        Args:
            config: TextWorldConfig instance containing:
                - mcts_config: Configuration for MCTS search
                - model_config: Configuration for neural networks
                - Other general agent settings
        """
        self.config = config
        self.true_state = {
            'history': [],
            'current_room': None,
            'valid_actions': [],
            'observation': None,
            'infos': None,
            'step_count': 0,
            'game_info': {
                'difficulty': config.game_config.treasure_level,
                'max_steps': config.max_steps
            }
        }
        self.reset()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize text processor
        self.text_processor = TextWorldStateProcessor(
            self.device,
            config.model_config.model_name
        )
        
        # Initialize networks and move to device
        self.policy_network = TextWorldPolicyNetwork(config.model_config).to(self.device)
        self.value_network = TextWorldValueNetwork(config.model_config).to(self.device)
        self.room_prediction_network = RoomPredictionNetwork(
            config.model_config,
            self.text_processor
        ).to(self.device)
        
        # Set networks to evaluation mode by default
        self.policy_network.eval()
        self.value_network.eval()
        self.room_prediction_network.eval()
        
        # Optional: Initialize optimizer if you plan to do any fine-tuning
        if hasattr(config.model_config, 'learning_rate'):
            self.optimizer = torch.optim.Adam([
                {'params': self.policy_network.parameters()},
                {'params': self.value_network.parameters()}
            ], lr=config.model_config.learning_rate)

    def reset(self):
        """Reset agent state"""
        self.goal = None
        self.known_rooms = set()
        self.true_state = {
            'history': [],
            'current_room': None,
            'valid_actions': [],
            'observation': None,
            'infos': None,
            'step_count': 0,
            'game_info': {
                'difficulty': self.config.game_config.treasure_level,
                'max_steps': self.config.max_steps
            }
        }

    def reset_environment(self, env):
        """Reset environment and agent state"""
        self.reset()
        obs, infos = env.reset()
        return obs, infos

    def verify_state(self, env, state):
        """Verify that a state can be reached"""
        # Save current state
        saved_obs = self.true_state['observation']
        saved_infos = self.true_state['infos']
        
        # Try to reach the state
        success, obs, infos = self.restore_state(env, state['history'])
        
        # Restore original state
        if saved_obs is not None:
            success, _, _ = self.restore_state(env, self.true_state['history'])
            if not success:
                print("WARNING: Failed to restore original state after verification")
        
        return success

    def parse_goal(self, initial_obs):
        """Extract goal from initial observation"""
        # Look for text between "there is something I need you to do" and "Alright, thanks!"
        goal_match = re.search(r"there is something I need you to do for me\.(.*?)Alright, thanks!", initial_obs, re.DOTALL)
        if goal_match:
            self.goal = goal_match.group(1).strip()
            print(f"DEBUG - Parsed goal: {self.goal}")
    
    def get_action(self, env, obs, infos, valid_actions, step=0):
        """Get next action using MCTS search"""
        print("\nDEBUG - get_action called")
        print(f"DEBUG - Current observation: {obs[:100]}...")
        
        # Parse goal from initial observation if not done yet
        if self.goal is None and "there is something I need you to do" in obs:
            self.parse_goal(obs)
        
        # Reset state if this is the first step of a new episode
        if step == 0:
            self.true_state = {
                'history': [],
                'done': False,
                'step_count': 0,
                'game_info': {
                    'difficulty': self.config.game_config.treasure_level,
                    'max_steps': self.config.max_steps
                }
            }
            print("DEBUG - Reset agent state for new episode")
        
        # Check if we're in a terminal state using environment info or state
        done = infos.get('done', False) or self.true_state.get('done', False)
        if done:
            print("DEBUG - In terminal state (from environment or state)")
            empty_stats = {
                'visit_counts': {},
                'q_values': {},
                'prior_probs': {},
                'room_predictions': {}
            }
            return None, empty_stats
        
        # Update current state before search
        current_room = self._get_room_name(obs)
        if current_room:
            self.known_rooms.add(current_room)
            
        # Update true state
        self.true_state.update({
            'observation': obs,
            'infos': infos,
            'valid_actions': valid_actions,
            'current_room': current_room,
            'step_count': step,
            'done': done
        })
        
        print(f"DEBUG - Updated true state:")
        print(f"  Step: {step}/{self.config.max_steps}")
        print(f"  Game difficulty: {self.config.game_config.treasure_level}")
        print(f"  Current room: {current_room}")
        print(f"  History before action: {self.true_state.get('history', [])}")
        print(f"  Done: {self.true_state.get('done', False)}")
        
        # Run MCTS with current state
        selected_action, search_stats = self.run_mcts_search(
            env,
            self.true_state.copy(),
            valid_actions
        )
        
        if selected_action is None:
            print("DEBUG - No valid action selected")
            return None, search_stats
        
        # Take the actual action and get the real result
        temp_obs, temp_reward, temp_done, temp_infos = env.step(selected_action)
        
        # Update true state with the REAL done flag from the environment
        self.true_state.update({
            'observation': temp_obs,
            'infos': temp_infos,
            'done': temp_done,  # This is crucial!
            'history': self.true_state.get('history', []) + [selected_action]
        })
        
        return selected_action, search_stats

    def _get_network_predictions(self, state_repr, valid_actions):
        """Get policy and value predictions from networks"""
        with torch.no_grad():
            # Process state and actions
            state_tokens = self.text_processor.process_state(state_repr)
            action_tokens = self.text_processor.process_actions(valid_actions)
            
            # Get policy predictions (handles batched actions internally)
            policy_logits = self.policy_network(state_tokens, action_tokens)
            scaled_logits = policy_logits.squeeze(0) / self.config.mcts_config.temperature
            probs = F.softmax(scaled_logits, dim=-1)
            
            # Convert to dictionary
            action_probs = {
                action: prob.item()
                for action, prob in zip(valid_actions, probs)
            }
            
            # Get Q-values (process one action at a time)
            q_values = {}
            for i, action in enumerate(valid_actions):
                curr_action_tokens = {
                    k: v[i:i+1] for k, v in action_tokens.items()
                }
                q_value = self.value_network(state_tokens, curr_action_tokens)
                q_values[action] = q_value.squeeze().item()
            
            # Get room predictions (processes one action at a time)
            room_predictions = {}
            for i, action in enumerate(valid_actions):
                curr_action_tokens = {
                    k: v[i:i+1] for k, v in action_tokens.items()
                }
                
                action_room_logits = self.room_prediction_network(
                    state_tokens,
                    curr_action_tokens,
                    list(self.known_rooms)
                )
                
                room_probs = F.softmax(action_room_logits.squeeze(0), dim=-1)
                probs_list = room_probs.tolist()
                
                # Structure predictions as expected by test
                known_rooms_dict = {}
                for room, prob in zip(list(self.known_rooms), probs_list[:-1]):
                    known_rooms_dict[room] = prob
                    
                room_predictions[action] = {
                    'known_rooms': known_rooms_dict,
                    'unseen': probs_list[-1]  # Last probability is for unseen rooms
                }
            
            return action_probs, q_values, room_predictions

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

    def _apply_temperature(self, logits, temperature):
        """
        Apply temperature scaling to logits
        
        Args:
            logits (torch.Tensor): Raw logits
            temperature (float): Temperature parameter
            
        Returns:
            torch.Tensor: Scaled logits
        """
        if temperature == 0:  # Handle zero temperature (greedy)
            max_logit = torch.max(logits, dim=-1, keepdim=True)[0]
            return (logits == max_logit).float()
        else:
            return F.softmax(logits / temperature, dim=-1)
        
    def _select_action(self, node):
        """Select action using UCB1"""
        # Ensure we have valid actions
        valid_actions = node.valid_actions
        if not valid_actions:
            return None
        
        # If not all actions are explored, choose an unexplored one
        unexplored = [a for a in valid_actions if a not in node.children]
        if unexplored:
            return random.choice(unexplored)
        
        # Calculate UCB1 for each action
        exploration_constant = 1.0
        total_visits = sum(child.visit_count for child in node.children.values())
        
        def ucb1(action, child):
            exploitation = child.q_value / child.visit_count if child.visit_count > 0 else 0
            exploration = exploration_constant * math.sqrt(math.log(total_visits) / child.visit_count)
            return exploitation + exploration
        
        # Only consider valid actions
        valid_children = {a: c for a, c in node.children.items() if a in valid_actions}
        if not valid_children:
            return None
        
        return max(valid_children.items(), key=lambda x: ucb1(x[0], x[1]))[0]


    def run_mcts_search(self, env, state, valid_actions, n_simulations=None):
        """Run MCTS search using Q-values and room predictions"""
        if n_simulations is None:
            n_simulations = self.config.mcts_config.n_simulations
        
        # Check if we're in a terminal state first
        if state.get('done', False):
            print("DEBUG - In terminal state, skipping MCTS")
            return None, {}
            
        # Filter out inventory and look actions
        valid_actions = [a for a in valid_actions if a.lower() not in ['inventory', 'look']]
        if not valid_actions:
            print("DEBUG - No valid actions after filtering!")
            return None, {}
            
        # Initialize root node with copy of current state
        root = MCTSNode(state, valid_actions)
        
        # Get initial predictions for root node
        state_repr = self._build_full_observation(
            state['observation'],
            state['current_room'],
            state['history']
        )
        print(f"DEBUG - MCTS root state: {state['current_room']}")
        print(f"DEBUG - MCTS valid actions: {valid_actions}")
        
        # Get network predictions
        action_probs, q_values, room_predictions = self._get_network_predictions(state_repr, valid_actions)
        print(f"DEBUG - Action probs: {action_probs}")
        print(f"DEBUG - Q-values: {q_values}")
        
        # Expand root with predictions
        root.expand(action_probs, room_predictions)
        
        # Check if we're already in a terminal state
        if state.get('done', False):
            print("DEBUG - Root is already in terminal state")
            return None, {}
        
        # Initialize children with immediate rewards
        for action in valid_actions:
            next_state, next_valid_actions, is_terminal, reward = self.simulate_action(env, action, root)
            if is_terminal:
                # For terminal states, create child node with terminal value
                root.children[action] = MCTSNode(
                    next_state,
                    [],  # No valid actions in terminal state
                    parent=root,
                    parent_action=action
                )
                root.children[action].terminal_value = reward
                root.children[action].q_value = reward
                root.children[action].visit_count = 1
                print(f"DEBUG - Initialized terminal child for action {action} with reward {reward}")
            else:
                next_valid_actions = [a for a in next_valid_actions if a.lower() not in ['inventory', 'look']]
                root.children[action] = MCTSNode(
                    next_state,
                    next_valid_actions,
                    parent=root,
                    parent_action=action
                )
                root.children[action].q_value = reward
                root.children[action].visit_count = 1
                print(f"DEBUG - Initialized child for action {action} with reward {reward}")
        
        # Run simulations
        for i in range(n_simulations):
            print(f"\nDEBUG - Starting simulation {i+1}")
            node = root
            search_path = [(node, 0)]
            accumulated_reward = 0
            
            # Selection: traverse tree to leaf node
            while node.children and not node.state['step_count'] >= self.config.max_steps:
                action = self._select_action(node)
                if action is None:  # No valid actions available
                    print(f"DEBUG - No valid actions at depth {len(search_path)}")
                    break
                    
                # Get reward for this action
                next_state, next_valid_actions, is_terminal, reward = self.simulate_action(env, action, node)
                accumulated_reward += reward
                print(f"DEBUG - Selected action {action}, reward {reward}")
                
                if action not in node.children:
                    # Handle terminal states
                    if is_terminal:
                        node.children[action] = MCTSNode(
                            next_state,
                            [],  # No valid actions in terminal state
                            parent=node,
                            parent_action=action
                        )
                        node.children[action].terminal_value = accumulated_reward
                        print(f"DEBUG - Created terminal node with value {accumulated_reward}")
                        break
                    
                    # Expand new non-terminal node
                    next_valid_actions = [a for a in next_valid_actions if a.lower() not in ['inventory', 'look']]
                    node.children[action] = MCTSNode(
                        next_state,
                        next_valid_actions,
                        parent=node,
                        parent_action=action
                    )
                    
                    # Get predictions for new node
                    next_state_repr = self._build_full_observation(
                        next_state['observation'],
                        next_state['current_room'],
                        next_state['history']
                    )
                    action_probs, q_values, room_predictions = self._get_network_predictions(
                        next_state_repr, 
                        next_valid_actions
                    )
                    node.children[action].expand(action_probs, room_predictions)
                    print(f"DEBUG - Expanded new node for action {action}")
                
                node = node.children[action]
                search_path.append((node, accumulated_reward))
                
                if hasattr(node, 'terminal_value'):
                    print("DEBUG - Reached terminal node")
                    break
                    
            # Evaluate leaf node if non-terminal
            if not hasattr(node, 'terminal_value') and not node.state['step_count'] >= self.config.max_steps:
                state_repr = self._build_full_observation(
                    node.state['observation'],
                    node.state['current_room'],
                    node.state['history']
                )
                
                # Get predictions for evaluation
                _, q_values, _ = self._get_network_predictions(state_repr, node.valid_actions)
                value = max(q_values.values()) if q_values else 0
                print(f"DEBUG - Evaluated leaf node with value {value}")
            else:
                value = getattr(node, 'terminal_value', accumulated_reward)
                print(f"DEBUG - Terminal node with value {value}")
            
            # Backup
            for node, reward in reversed(search_path):
                node.visit_count += 1
                if node.parent is not None:
                    node.q_value += reward + value
                print(f"DEBUG - Updated node with visit count {node.visit_count}, q_value {node.q_value}")
        
        if not root.children:
            print("DEBUG - Root has no children after search!")
            return None, {}
        
        # Select action using visit counts
        selected_action = max(
            root.children.items(),
            key=lambda x: x[1].visit_count
        )[0]
        
        # Collect statistics
        search_stats = {
            'visit_counts': {action: child.visit_count for action, child in root.children.items()},
            'q_values': {action: child.q_value/child.visit_count if child.visit_count > 0 else 0 
                        for action, child in root.children.items()},
            'prior_probs': root.prior_probs,
            'room_predictions': root.room_predictions
        }
        
        return selected_action, search_stats
        
    def simulate_action(self, env, action, node):
        """Simulate taking an action from a state"""
        print(f"\nDEBUG - simulate_action called with action: {action}")
        print(f"DEBUG - Current node state done flag: {node.state.get('done', False)}")
        
        # Save current state
        saved_obs = node.state['observation']
        saved_infos = node.state['infos']
        
        # Check if current state is terminal
        if node.state.get('done', False):
            print("DEBUG - Current state is terminal, cannot simulate further")
            return (
                node.state,
                [],
                True,
                0
            )
        
        # Try to restore to current state if we have history
        if node.state['history']:
            print(f"DEBUG - Attempting to restore state with history: {node.state['history']}")
            success, obs, infos = self.restore_state(env, node.state['history'])
            if not success:
                print("DEBUG - State restoration failed")
                return None, None, True, 0
        else:
            print("DEBUG - No history to restore, using current state")
            obs = saved_obs
            infos = saved_infos
        
        # Take action
        print(f"DEBUG - Taking action: {action}")
        obs, reward, done, infos = env.step(action)
        print(f"DEBUG - Action result: reward={reward}, done={done}")
        
        # Create next state - with special handling for terminal states
        next_state = {
            'observation': obs,
            'infos': infos,
            'history': node.state['history'] + [action],
            'current_room': self._get_room_name(obs),
            'step_count': node.state['step_count'] + 1,
            'valid_actions': [] if done else infos['admissible_commands'],  # Empty list for terminal states
            'game_info': node.state['game_info'],
            'done': done
        }
        
        # For terminal states, immediately return without trying to restore
        if done:
            print("DEBUG - Reached terminal state, skipping state restoration")
            return next_state, [], True, reward
        
        # Restore original state if not terminal
        if saved_obs is not None:
            print("DEBUG - Attempting to restore original state")
            success, _, _ = self.restore_state(env, node.state['history'])
            if not success:
                print("WARNING: Failed to restore original state after simulation")
        
        return next_state, infos['admissible_commands'], done, reward





    
    """History tracking for both real episodes and rollouts"""
    




    def create_rollout_state(self):
        """Create initial state for MCTS rollout"""
        return {
            'observation': None,
            'infos': None,
            'history': [],
            'step_count': 0
        }


    def reset_environment(self):
        """Reset environment and clear agent state"""
        self.known_rooms = set()
        self.episode_history = []
        
        # Clear any MCTS-related state
        if hasattr(self, 'root_node'):
            self.root_node = None
        
        # Reset networks to evaluation mode
        self.policy_network.eval()
        self.value_network.eval()
        self.room_prediction_network.eval()
        
        return self.env.reset()

    def save_agent_state(self, save_path):
        """
        Save agent's state including networks and configuration
        
        Args:
            save_path (str): Path to save agent state
        """
        state_dict = {
            'policy_network': self.policy_network.state_dict(),
            'value_network': self.value_network.state_dict(),
            'room_prediction_network': self.room_prediction_network.state_dict(),
            'config': self.config,
            'known_rooms': self.known_rooms,
            'episode_history': self.episode_history,
            'goal': self.goal
        }
        torch.save(state_dict, save_path)

    def load_agent_state(self, load_path):
        """
        Load agent's state from a file
        
        Args:
            load_path (str): Path to load agent state from
        """
        state_dict = torch.load(load_path)
        
        # Load network weights
        self.policy_network.load_state_dict(state_dict['policy_network'])
        self.value_network.load_state_dict(state_dict['value_network'])
        self.room_prediction_network.load_state_dict(state_dict['room_prediction_network'])
        
        # Load other state
        self.config = state_dict['config']
        self.known_rooms = state_dict['known_rooms']
        self.episode_history = state_dict['episode_history']
        self.goal = state_dict['goal']
        
        # Reset MCTS state
        if hasattr(self, 'root_node'):
            self.root_node = None
        
        # Set networks to evaluation mode
        self.policy_network.eval()
        self.value_network.eval()
        self.room_prediction_network.eval()

    def update_history(self, location, action):
        """Add new location-action pair to history"""
        self.episode_history.append((location, action))








    """Observation processing and formatting"""




    def _format_location_action(self, location, action):
        """Format a single location-action pair for history"""
        return f"At {location}: {action}"
    
    def _clean_observation(self, obs, infos):
        """Extract room name and clean observation text"""
        # Look for room name in standard format "-= Room Name =-"
        room_match = re.search(r'-= (.+) =-', obs)
        room_name = room_match.group(1) if room_match else None
        
        # Clean observation by removing room header
        clean_obs = obs
        if room_match:
            clean_obs = obs.replace(room_match.group(0), '').strip()
        
        return clean_obs, room_name

    def _build_full_observation(self, current_obs, room_name, history):
        """Build full observation string including history"""
        full_obs = []
        
        # Add current observation
        full_obs.append(current_obs)
        
        # Add current room if known
        if room_name:
            full_obs.append(f"Current room: {room_name}")
        
        # Add action history
        if history:
            history_str = "Previous actions:"
            for action in history:
                history_str += f"\n- {action}"
            full_obs.append(history_str)
        
        # Add goal if known
        if self.goal:
            full_obs.append(f"Goal: {self.goal}")
        
        # Add game info
        full_obs.append(f"Game difficulty: {self.config.game_config.treasure_level}")
        full_obs.append(f"Max steps: {self.config.max_steps}")
        
        return " ".join(full_obs)

    def _extract_initial_goal(self, obs):
        """Extract goal text from initial observation"""
        def is_ascii_art(line):
            ascii_chars = set('\\|_$*=.+-/')
            special_char_count = sum(1 for c in line if c in ascii_chars)
            return special_char_count > (len(line) * 0.1)

        def is_meaningful_text(line):
            stripped = line.strip()
            return (
                stripped and
                not is_ascii_art(stripped) and
                not stripped.isspace() and
                len(stripped) > 5
            )

        lines = [line.strip() for line in obs.split('\n')]
        
        for line in lines:
            if is_meaningful_text(line):
                return line
                
        return None
    
    def restore_state(self, env, history):
        """Restore environment to a specific state"""
        obs, infos = env.reset()
        
        print(f"DEBUG - Restoring state with history: {history}")
        for i, action in enumerate(history):
            print(f"DEBUG - Replaying action {i}: {action}")
            obs, reward, done, infos = env.step(action)
            if done:
                print(f"DEBUG - Restoration failed at action {i}: {action}")
                print(f"DEBUG - Environment observation: {obs}")
                return False, obs, infos
        
        return True, obs, infos

    def verify_state(self, expected_state):
        """Verify environment matches expected state"""
        obs, infos = self.env.get_state() if hasattr(self.env, 'get_state') else (None, None)
        
        if obs is None or infos is None:
            return False
        
        current_state = {
            'current_room': self._get_room_name(obs),
            'valid_actions': [a for a in infos['admissible_commands'] 
                             if a.lower() not in ['inventory', 'look']]
        }
        
        return (current_state['current_room'] == expected_state['current_room'] and 
                set(current_state['valid_actions']) == set(expected_state['valid_actions']))

    def _get_room_name(self, obs):
        """Extract room name from observation"""
        room_match = re.search(r'-= (.+) =-', obs)
        return room_match.group(1) if room_match else None


class MCTSNode:
    def __init__(self, state, valid_actions, parent=None, parent_action=None):
        """
        Initialize MCTS node
        
        Args:
            state (dict): Contains:
                - observation: Current observation text
                - infos: Environment info dict
                - history: List of actions taken
                - step_count: Number of steps taken
                - game_info: Game configuration info
            valid_actions (list): List of valid actions from this state
            parent: Parent node (None for root)
            parent_action: Action that led to this node
        """
        self.state = state
        self.valid_actions = [a for a in valid_actions if a.lower() not in ['inventory', 'look']]
        self.parent = parent
        self.parent_action = parent_action
        self.children = {}
        self.visit_count = 0
        self.q_value = 0
        
    def expand(self, prior_probs, room_predictions):
        """Initialize children with prior probabilities and room predictions"""
        self.prior_probs = prior_probs
        self.room_predictions = room_predictions

    @property
    def value(self):
        """Get mean Q-value"""
        if self.visit_count == 0:
            return 0
        return self.q_value / self.visit_count


class TextWorldStateProcessor:
    def __init__(self, device, model_name, max_length=512):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
    def process_state(self, text):
        """Process a single state observation"""
        tokens = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        return {k: v.to(self.device) for k, v in tokens.items()}
    
    def process_actions(self, actions):
        """Process a list of actions"""
        tokens = self.tokenizer(
            actions,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        return {k: v.to(self.device) for k, v in tokens.items()}
    
    def process_text(self, text):
        """Process any text input (single string)"""
        tokens = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        return {k: v.to(self.device) for k, v in tokens.items()}

