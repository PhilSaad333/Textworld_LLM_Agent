import gymnasium as gym

class GymWrapper:
    def __init__(self, env_name="CartPole-v1", render_mode=None):
        """
        Initialize the Gym environment wrapper.
        
        Args:
            env_name (str): Name of the Gym environment to create
            render_mode (str, optional): Rendering mode ('human', 'rgb_array', None)
                                       None means no rendering
        """
        self.env_name = env_name
        self.render_mode = render_mode
        self.env = gym.make(env_name, render_mode=render_mode)
        
        # Store environment properties
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def reset(self, seed=None):
        """
        Reset the environment.
        
        Args:
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            observation: Initial observation
        """
        if seed is not None:
            self.env.reset(seed=seed)
            
        observation, info = self.env.reset()
        return observation
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take in the environment
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info
    
    def close(self):
        """Close the environment."""
        self.env.close()
        
    def get_action_space(self):
        """Return the action space of the environment."""
        return self.action_space
    
    def get_observation_space(self):
        """Return the observation space of the environment."""
        return self.observation_space
    
    def seed(self, seed=None):
        """Set the seed for the environment."""
        return self.env.seed(seed)