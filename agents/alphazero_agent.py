import math
import torch
import random
import gymnasium as gym

class AlphaZeroAgent:
    def __init__(self, config, model=None):
        """
        Initialize the agent with the given config and optional model.
        Will handle MCTS parameters like c_puct, num_simulations, etc.
        
        Args:
            config (dict): Configuration dictionary containing:
                - num_simulations (int): Number of MCTS simulations per move
                - c_puct (float): Exploration constant for PUCT formula
                - temperature (float): Initial temperature for action selection
                - dirichlet_alpha (float): Alpha parameter for Dirichlet noise
                - dirichlet_epsilon (float): Fraction of Dirichlet noise to add to priors
            model: Neural network model that provides policy and value predictions
        """
        self.config = config
        self.model = model
        
        # MCTS parameters
        self.num_simulations = config.get('num_simulations', 100)
        self.c_puct = config.get('c_puct', 1.0)
        self.temperature = config.get('temperature', 1.0)
        
        # Exploration parameters
        self.dirichlet_alpha = config.get('dirichlet_alpha', 0.3)
        self.dirichlet_epsilon = config.get('dirichlet_epsilon', 0.25)
        
        # Statistics tracking
        self.mcts_stats = MCTSStats()
        
        # Current game state
        self.current_node = None

    def select_action(self, state, temperature=None):
        """
        Run MCTS from current state and select action based on visit counts.
        
        Args:
            state: Current environment state
            temperature: Temperature for action selection (uses self.temperature if None)
        
        Returns:
            action: Selected action
            action_probs: Dictionary of action probabilities from MCTS
        """
        if temperature is None:
            temperature = self.temperature
        
        # Run MCTS simulations
        root = self.run_mcts(state)
        
        # Get action probabilities
        action_probs = self.get_action_probs(root, temperature)
        
        # Select action based on probabilities
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        action = random.choices(actions, weights=probs)[0]
        
        # Store statistics for training
        self.store_search_statistics(root)
        
        # Update current node to the chosen action's node
        self.current_node = root.children.get(action)
        
        return action, action_probs

    def run_mcts(self, state):
        """
        Perform MCTS simulations starting from given state.
        Uses model for value and policy predictions during tree traversal.
        
        Args:
            state: Current environment state
        
        Returns: 
            root: Root node of the search tree
        """
        # Initialize root node if needed
        if self.current_node is None:
            self.current_node = Node(prior_probability=1.0)
        root = self.current_node
        
        for _ in range(self.num_simulations):
            # Select
            leaf, path = self.select_leaf(root)
            
            # Expand and evaluate
            value = self.expand_node(leaf, state)
            
            # Backpropagate
            self.backpropagate(path, value)
        
        return root

    def select_leaf(self, node):
        """
        Starting from given node, traverse tree using PUCT formula
        to select path to a leaf node.
        
        Returns:
            leaf_node: Node object representing the selected leaf
            path: List of nodes traversed to reach the leaf
        """
        path = []
        while not node.is_leaf():
            # Find child with maximum UCB score
            max_ucb = max(child.get_ucb_score(self.c_puct) for child in node.children.values())
            best_action = random.choice([a for a, child in node.children.items() 
                                       if child.get_ucb_score(self.c_puct) == max_ucb])
            path.append(node)
            node = node.children[best_action]
            
        path.append(node)
        return node, path

    def expand_node(self, node, state):
        """
        Use model to get policy and value predictions for this state.
        Create child nodes for all possible actions.
        
        Args:
            node: Node object to expand
            state: Current environment state
        
        Returns:
            value: Predicted value for this state
        """
        # Get valid actions from the environment's action space
        valid_actions = list(range(self.env.action_space.n))
        
        # Get policy and value predictions from the model
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        policy_logits, value = self.model(state_tensor)
        
        # Convert policy logits to probabilities and convert to numpy array
        policy = torch.softmax(policy_logits.squeeze(), dim=0).detach().numpy()
        
        # Mask invalid actions and renormalize
        valid_policy = policy[valid_actions]
        valid_policy /= valid_policy.sum()
        
        # Expand node with valid actions and their probabilities
        node.expand(valid_actions, valid_policy)
        
        return value.item()

    def backpropagate(self, path, value):
        """
        Update statistics (visit counts, value totals) for all nodes
        in the path from leaf to root.
        
        Args:
            path: List of nodes traversed in selection
            value: Value to backpropagate
        """
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value

    def get_action_probs(self, node, temperature=1.0):
        """
        Convert visit counts of root children into action probabilities.
        Higher temperature makes distribution more uniform.
        
        Args:
            node: Root node of the search tree
            temperature: Temperature parameter for visit count distribution
        
        Returns: 
            action_probs: Dictionary mapping actions to their probabilities
        """
        visits = {action: child.visit_count for action, child in node.children.items()}
        if temperature == 0:  # Act greedily
            max_visit = max(visits.values())
            actions = [action for action, visit in visits.items() if visit == max_visit]
            probs = {action: 1.0 / len(actions) if action in actions else 0.0 
                    for action in visits.keys()}
        else:
            # Apply temperature
            visits_temp = {action: count ** (1 / temperature) for action, count in visits.items()}
            total = sum(visits_temp.values())
            probs = {action: count / total for action, count in visits_temp.items()}
        
        return probs

    def store_search_statistics(self, node):
        """
        Save the MCTS statistics for later training.
        Records visit counts to use as policy targets.
        """


class Node:
    def __init__(self, prior_probability, parent=None, action=None):
        """
        Initialize a node in the MCTS tree.
        Args:
            prior_probability: P(action) predicted by neural network
            parent: Parent node
            action: Action that led to this node
        """
        self.parent = parent
        self.action = action
        self.prior_p = prior_probability
        
        # Tree structure
        self.children = {}  # map of action -> Node
        
        # MCTS statistics
        self.visit_count = 0
        self.value_sum = 0
        self.state = None  # Optional: store state if memory allows
    
    def get_value(self):
        """Get mean value (Q) for this node"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def get_ucb_score(self, c_puct):
        """
        Calculate UCB score for this node using the PUCT algorithm
        score = Q + U where U = c_puct * P * sqrt(N_parent) / (1 + N)
        """
        if self.parent is None:
            return 0
        
        # U = c_puct * P * sqrt(N_parent) / (1 + N)
        parent_visit_count = self.parent.visit_count
        exploration = (c_puct * self.prior_p * 
                      math.sqrt(parent_visit_count) / (1 + self.visit_count))
        
        return self.get_value() + exploration

    def is_leaf(self):
        """Check if node is a leaf (has no children)"""
        return len(self.children) == 0

    def expand(self, actions, action_priors):
        """
        Expand node with given legal actions and their prior probabilities
        from neural network
        
        Args:
            actions: List of legal actions
            action_priors: List of prior probabilities for each action
        """
        for action, prob in zip(actions, action_priors):
            if action not in self.children:
                self.children[action] = Node(
                    prior_probability=prob,
                    parent=self,
                    action=action
                )


class MCTSStats:
    def __init__(self):
        """
        Store statistics from MCTS for training
        """
        self.visit_counts = []  # Visit count distributions for each move
        self.states = []  # Game states
        self.actions = []  # Actions taken
        self.rewards = []  # Rewards received
        self.current_game_stats = []  # Stats for current game

    def push_state(self, state, action_probs):
        """
        Store state and MCTS policy (visit counts) for current game
        """

    def push_reward(self, reward):
        """Store reward received"""

    def clear_current_game(self):
        """Reset stats for new game"""

    def get_training_data(self):
        """
        Process all stored games into format needed for training
        Returns: states, policies, values
        """