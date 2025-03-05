

class Trajectory:
    """Container for a single gameplay trajectory"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.action_probs = []
        self.valid_actions = []
        self.format_checks = []
        
    def compute_returns(self, gamma=0.99):
        """Compute discounted returns"""
        # Calculate returns for each step

class RolloutCollector:
    """Collects gameplay trajectories"""
    
    def __init__(self, agent, env_manager, config):
        self.agent = agent
        self.env_manager = env_manager
        self.config = config
        
    def collect_trajectories(self, n_trajectories, difficulty):
        """Collect n gameplay trajectories at specified difficulty"""
        # Play games and collect experience