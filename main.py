from environment import GymWrapper
from config import AlphaZeroConfig


def main():
    # Create environment with specified type
    env_name = "CartPole-v1"  # We can easily change this to other environments
    env = GymWrapper(env_name, render_mode="human")
    
    # Create and update config
    config = AlphaZeroConfig()
    config.update_for_environment(env.env)
    
    # Print some information about the environment
    print(f"Environment: {env_name}")
    print(f"State dimension: {config.state_dim}")
    print(f"Action dimension: {config.action_dim}")
    print(f"Discrete actions: {config.discrete_actions}")
    
    # Run a few episodes with random actions to test environment
    num_episodes = 3
    max_steps = 1000
    
    for episode in range(num_episodes):
        observation = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # For now, just take random actions
            action = env.env.action_space.sample()
            
            # Take a step in the environment
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            
            if done:
                print(f"Episode {episode + 1} finished after {step + 1} steps with reward {episode_reward}")
                break
    
    env.close()


if __name__ == "__main__":
    main()