from agents.textworld_llm_agent import TextWorldLLMAgent

class RLAgent(TextWorldLLMAgent):
    """Extends TextWorldLLMAgent with RL capabilities"""
    
    def __init__(self, config, model_path=None, use_map=True):
        super().__init__(config, training_mode=True, model_path=model_path, use_map=use_map)
        # RL-specific initialization
        
    def get_action_probabilities(self, obs, valid_actions):
        """Return probability distribution over valid actions"""
        # Use the LLM to generate probabilities for each action
        
    def update_policy(self, trajectories, optimizer):
        """Update policy based on collected trajectories"""
        # Apply GRPO update