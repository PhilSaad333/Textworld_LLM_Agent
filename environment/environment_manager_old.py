import os
import textworld
import numpy as np
import torch
from dataclasses import replace
from config.config import GameType

class DifficultyEnvironmentManager:
    """Manages environments for different difficulty levels and game types"""
    def __init__(self, config):
        self.config = config
        self.environments = {}  # difficulty -> env mapping
        self.game_files = {}   # difficulty -> game_file mapping
        self.env_ids = {}      # difficulty -> env_id mapping
        
    def get_or_create_env(self, difficulty):
        """Get existing env or create new one for difficulty"""
        if difficulty not in self.environments:
            # Update config with the new difficulty
            if self.config.game_config.game_type == GameType.TREASURE:
                self.config.game_config.treasure_level = difficulty
            elif self.config.game_config.game_type == GameType.COINS:
                self.config.game_config.coin_level = difficulty
            else:
                raise ValueError(f"Unsupported game type for difficulty-based environments: {self.config.game_config.game_type}")
            
            # Create game file if needed
            success = self.config.create_game()
            if not success:
                raise RuntimeError(f"Failed to create game for difficulty {difficulty}")
            
            # Cache the game file and env_id
            self.game_files[difficulty] = self.config.game_file
            self.env_ids[difficulty] = self.config.env_id
            
            # Create and cache the environment
            self.environments[difficulty] = textworld.gym.make(self.config.env_id)
            
        return self.environments[difficulty]
    
    def cleanup(self):
        """Clean up resources"""
        for env in self.environments.values():
            env.close()
        self.environments.clear()
        self.game_files.clear()
        self.env_ids.clear()
        
        # Clear any CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def reset_environment(self, difficulty):
        """Reset specific difficulty environment"""
        if difficulty in self.environments:
            self.environments[difficulty].close()
            del self.environments[difficulty]
            del self.game_files[difficulty]
            del self.env_ids[difficulty]
        return self.get_or_create_env(difficulty)
    
    @property
    def max_difficulty(self):
        """Get maximum difficulty level based on game type"""
        if self.config.game_config.game_type == GameType.TREASURE:
            return 30
        elif self.config.game_config.game_type == GameType.COINS:
            return 300
        else:
            raise ValueError(f"Unsupported game type for difficulty-based environments: {self.config.game_config.game_type}")

