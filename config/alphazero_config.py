# config/alphazero_config.py

class AlphaZeroConfig:
    def __init__(self):
        # Network Architecture
        self.num_residual_blocks = 5
        self.num_channels = 128
        
        # MCTS parameters
        self.num_simulations = 50  # Number of MCTS simulations per move
        self.c_puct = 1.0  # Exploration constant for PUCT algorithm
        
        # Training parameters
        self.batch_size = 64
        self.num_epochs = 100
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        
        # Self-play parameters
        self.num_self_play_episodes = 100
        self.temperature = 1.0  # Initial temperature for action selection
        self.temperature_threshold = 10  # Number of moves before temperature drops
        
        # Replay Buffer
        self.replay_buffer_size = 10000
        self.num_iterations = 100  # Number of training iterations
        
        # Environment parameters (to be set based on specific environment)
        self.state_dim = None
        self.action_dim = None
        self.discrete_actions = True

    def update_for_environment(self, env):
        """Update configuration based on the specific environment"""
        import numpy as np
        import gymnasium as gym  # Changed to gymnasium

        # Set state dimensions
        if isinstance(env.observation_space, gym.spaces.Box):
            self.state_dim = env.observation_space.shape
        else:
            raise ValueError("Unsupported observation space")

        # Set action dimensions
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim = env.action_space.n
            self.discrete_actions = True
        elif isinstance(env.action_space, gym.spaces.Box):
            self.action_dim = env.action_space.shape[0]
            self.discrete_actions = False
        else:
            raise ValueError("Unsupported action space")