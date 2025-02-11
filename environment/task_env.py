import os
import numpy as np
import textworld
import gymnasium as gym
from typing import Dict, Any, Optional
from dataclasses import dataclass
import random
from math import floor

# Import EnvInfos for proper request specification.
from textworld import EnvInfos

@dataclass
class TaskConfig:
    max_steps: int = 100
    request_infos: EnvInfos = None
    grammar_flags: dict = None
    scale: int = 10    # With higher scale, complexity increases less frequently.
    
    def __post_init__(self):
        if self.request_infos is None:
            # Specify the infos to request as an EnvInfos instance.
            self.request_infos = EnvInfos(
                description=True,
                inventory=True,
                intermediate_reward=True,
                admissible_commands=True,
                objective=True,
                win_facts=False  # Disable win_facts to avoid extra processing.
            )
        if self.grammar_flags is None:
            # Use a more verbose quest objective by default.
            self.grammar_flags = {"only_last_action": False}


class TaskEnvManager:
    def __init__(self, config: TaskConfig):
        self.config = config
        self.envs: Dict[int, Any] = {}    # difficulty -> env mapping
        self.games_collection: Dict[int, str] = {}  # difficulty -> game_file mapping
        
        # Initialize random number generators.
        self.seed(1234)
        
        # Basic vocabulary for our games.
        self.vocab = [
            "go", "north", "south", "east", "west",
            "take", "drop", "inventory", "look",
            "coin", "room", "door"
        ]
        
    def seed(self, seed: int):
        """Initialize all random number generators."""
        self.rng_games = np.random.RandomState(seed + 1)  # For shuffling games.
        self.rng_make = np.random.RandomState(seed + 2)  # For generating games.
        self.seed_cmds = seed + 3  # For shuffling commands.
        
        # Fixed seeds for consistent game generation.
        self.seed_map = self.rng_make.randint(65635)
        self.seed_objects = self.rng_make.randint(65635)
        self.seed_quest = self.rng_make.randint(65635)
        self.seed_grammar = self.rng_make.randint(65635)
        self.seed_inform7 = self.rng_make.randint(65635)
    
    def _get_seeds_for_level(self, level: int) -> dict:
        """Return a dictionary of random state objects for game generation."""
        rngs = {
            "map": np.random.RandomState(self.seed_map + level),
            "objects": np.random.RandomState(self.seed_objects + level),
            "quest": np.random.RandomState(self.seed_quest + level),
            "grammar": np.random.RandomState(self.seed_grammar + level),
            "inform7": np.random.RandomState(self.seed_inform7 + level)
        }
        return rngs
    
    def _make_game(self, level: int, rngs: dict) -> str:
        """Create a new game file for the given difficulty level."""
        # Compute effective difficulty based on scale.
        # For example, with scale=5, levels 1-5 all have effective_diff = 1.
        effective_diff = ((level - 1) // self.config.scale) + 1
        
        # Generate a unique game name incorporating both level and scale.
        game_name = f"twcc_level{level}_scale{self.config.scale}_step{self.config.max_steps}"
        game_file = os.path.join("gen_games", f"{game_name}.ulx")
        
        try:
            # Remove an existing game file to avoid conflicts.
            if os.path.exists(game_file):
                os.remove(game_file)
            
            # Gradually scale world parameters using effective difficulty.
            world_size = min(effective_diff + 2, 15)
            nb_objects = min(effective_diff + 1, 10)
            
            # Create a world. Rooms and objects scale with effective difficulty (but are capped).
            world = textworld.generator.make_world(
                world_size=world_size,
                nb_objects=nb_objects,
                rngs=rngs
            )
            
            # Set quest options dynamically using effective difficulty.
            quest_options = textworld.GameOptions()
            if effective_diff < 3:
                quest_options.chaining.min_length = 1
                quest_options.chaining.max_length = effective_diff + 1
                quest_options.chaining.min_breadth = 1
                quest_options.chaining.max_breadth = 2
                quest_options.chaining.min_depth = 1
                quest_options.chaining.max_depth = 2
            elif effective_diff <= 10:
                # Determine available resources
                available_objects = nb_objects       # equals min(effective_diff + 1, 10)
                available_rooms = world_size           # equals min(effective_diff + 2, 15)
                if available_objects < 6:
                    # With too few objects, use a simpler quest structure.
                    quest_options.chaining.min_length = 1
                    quest_options.chaining.max_length = available_objects
                    quest_options.chaining.min_breadth = 1
                    quest_options.chaining.max_breadth = 1
                    quest_options.chaining.min_depth = 1
                    quest_options.chaining.max_depth = 1
                else:
                    quest_options.chaining.min_length = 3
                    quest_options.chaining.max_length = min(6, available_objects)
                    if available_rooms < 7:
                        quest_options.chaining.min_breadth = 1
                        quest_options.chaining.max_breadth = 1
                        quest_options.chaining.min_depth = 1
                        quest_options.chaining.max_depth = 1
                    else:
                        quest_options.chaining.min_breadth = 1
                        quest_options.chaining.max_breadth = 2
                        quest_options.chaining.min_depth = 1
                        quest_options.chaining.max_depth = 2
            else:
                quest_options.chaining.min_length = 4
                quest_options.chaining.max_length = min(6 + ((effective_diff - 10) // 5), 10)
                quest_options.chaining.min_breadth = 2
                quest_options.chaining.max_breadth = 3
                quest_options.chaining.min_depth = 2
                quest_options.chaining.max_depth = 3
            
            # Generate quest(s) for the world.
            quests = textworld.generator.make_quest(world, quest_options)
            if not isinstance(quests, list):
                quests = [quests]
            quest = quests[0]
            
            # Optionally, enhance the quest objective if a detailed solution is available.
            try:
                if hasattr(quest, "solution") and quest.solution:
                    verbose_plan = " -> ".join(quest.solution)
                    quest.objective += f"\nPlan: {verbose_plan}"
            except Exception:
                pass
            
            # Create the grammar using a mapping of options. Theme is supplied explicitly.
            grammar_options = {"theme": "house", **self.config.grammar_flags}
            grammar = textworld.generator.make_grammar(grammar_options)
            
            # Create the game using make_game_with which accepts the world, quests, and grammar.
            game = textworld.generator.make_game_with(
                world=world,
                quests=[quest],
                grammar=grammar
            )
            
            # Compile and save the game.
            comp_options = textworld.GameOptions()
            comp_options.path = game_file
            game_file = textworld.generator.compile_game(game, comp_options)
            return game_file
            
        except Exception as e:
            print(f"Failed to create game for level {level}: {str(e)}")
            return None
    
    def get_or_create_env(self, level: int) -> Optional[gym.Env]:
        """Get an existing environment or create a new one for the given difficulty level."""
        if level not in self.envs:
            rngs = self._get_seeds_for_level(level)
            game_file = self._make_game(level, rngs)
            
            if game_file is None:
                return None
                
            self.games_collection[level] = game_file
            
            # Register the game file and get an environment id.
            uid = textworld.gym.register_games(
                [game_file],
                request_infos=self.config.request_infos,
                max_episode_steps=self.config.max_steps
            )
            # Use TextWorld's gym.make to create the environment properly.
            self.envs[level] = textworld.gym.make(uid)
                
        return self.envs[level]
    
    def cleanup(self):
        """Clean up environments and clear cached games."""
        for env in self.envs.values():
            env.close()
        self.envs.clear()
        self.games_collection.clear()
