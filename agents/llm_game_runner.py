import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple, Union

from config.config import TextWorldConfig, EvalConfig
from environment.task_env import TaskEnvManager
from agents.textworld_llm_agent import TextWorldLLMAgent

class LLMGameRunner:
    """Enhanced game runner for comprehensive evaluation of TextWorld agents"""
    
    def __init__(
        self,
        agent: TextWorldLLMAgent,
        env_manager: TaskEnvManager,
        config: TextWorldConfig,
        log_dir: str = "./logs",
        eval_config: Optional[EvalConfig] = None
    ):
        """Initialize the game runner
        
        Args:
            agent: The agent to evaluate
            env_manager: The environment manager
            config: The TextWorld configuration
            log_dir: Directory to save logs
            eval_config: Optional evaluation configuration
        """
        self.agent = agent
        self.env_manager = env_manager
        self.config = config
        self.log_dir = log_dir
        
        # Use provided eval_config or the one from config
        self.eval_config = eval_config or config.eval_config
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize evaluation statistics
        self.reset_statistics()
        
        # Set up logging
        self.log_file = None
        if self.eval_config.log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = self.eval_config.log_path or os.path.join(log_dir, f"evaluation_{timestamp}.log")
            self.log_file = open(self.log_path, "w")
            self.log(f"Evaluation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.log(f"Model: {agent.model_name}")
            self.log(f"Evaluation config: {vars(self.eval_config)}")
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
    
    def reset_statistics(self):
        """Reset evaluation statistics"""
        self.statistics = {
            # Game performance metrics
            "games_played": 0,
            "games_succeeded": 0,
            "total_steps": 0,
            "successful_steps": 0,
            "total_score": 0,
            "completion_times": [],
            
            # Format metrics
            "command_tag_usage": 0,
            "room_tag_usage": 0,
            "format_correct": 0,
            "format_errors": Counter(),
            
            # Room prediction metrics
            "room_predictions": 0,
            "room_predictions_correct": 0,
            "new_room_predictions": 0,
            "new_room_predictions_correct": 0,
            
            # Token usage metrics
            "input_tokens": [],
            "output_tokens": [],
            
            # Action metrics
            "valid_actions": 0,
            "invalid_actions": 0,
            "action_counts": Counter(),
            
            # Per difficulty statistics
            "per_difficulty": defaultdict(lambda: {
                "games_played": 0,
                "games_succeeded": 0,
                "total_steps": 0,
                "successful_steps": 0,
                "total_score": 0,
                "completion_times": [],
                "format_correct": 0,
                "total_actions": 0
            }),
            
            # Beam search statistics
            "beam_diversity": [],
            "beam_format_correctness": []
        }
    
    def log(self, message: str):
        """Log a message to the console and log file
        
        Args:
            message: The message to log
        """
        print(message)
        if self.log_file:
            self.log_file.write(message + "\n")
            self.log_file.flush()
    
    def play_game(
        self, 
        difficulty: int = 1, 
        max_steps: int = 20,
        log: bool = True,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Play a game at the specified difficulty level
        
        Args:
            difficulty: The difficulty level
            max_steps: Maximum number of steps
            log: Whether to log game progress
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary with game record
        """
        # Create environment
        env = self.env_manager.get_or_create_env(difficulty)
        
        # Reset agent
        self.agent.reset()
        
        # Initialize game record
        game_record = {
            "difficulty": difficulty,
            "trajectory": [],
            "metrics": {
                "success": False,
                "steps": 0,
                "score": 0,
                "format_check_passed": 0,
                "room_prediction_correct": 0,
                "start_time": time.time(),
                "end_time": None,
                "completion_time": None,
                "token_counts": {
                    "input": [],
                    "output": []
                }
            }
        }
        
        # Start game
        obs, infos = env.reset()
        done = False
        step = 0
        
        if log:
            self.log(f"\n{'='*50}")
            self.log(f"Starting game at difficulty {difficulty}")
            self.log(f"{'='*50}\n")
            self.log(f"Initial observation:\n{obs}\n")
        
        # Extract goal
        if self.agent.goal is None:
            self.agent.goal = self.agent.parse_goal(obs)
            if log:
                self.log(f"Goal: {self.agent.goal}\n")
        
        # Game loop
        while not done and step < max_steps:
            # Get action from agent
            action_start_time = time.time()
            action, info = self.agent.get_action(env, obs, infos, infos["admissible_commands"], step)
            action_time = time.time() - action_start_time
            
            # Take action in environment
            next_obs, reward, done, next_infos = env.step(action)
            
            # Get current room
            current_room = self.agent._get_room_name(obs)
            next_room = self.agent._get_room_name(next_obs)
            
            # Check if room prediction was correct
            room_prediction = info.get("predicted_room", None)
            room_prediction_correct = False
            
            if room_prediction:
                # Handle "New Room" prediction
                if room_prediction.lower() == "new room":
                    room_prediction_correct = next_room not in self.agent.known_rooms
                    self.statistics["new_room_predictions"] += 1
                    if room_prediction_correct:
                        self.statistics["new_room_predictions_correct"] += 1
                else:
                    # Regular room prediction
                    room_prediction_correct = room_prediction.lower() == next_room.lower()
                
                self.statistics["room_predictions"] += 1
                if room_prediction_correct:
                    self.statistics["room_predictions_correct"] += 1
            
            # Check format correctness
            format_check = info.get("format_check", {})
            format_check_passed = format_check.get("has_command_tags", False) and format_check.get("has_room_tags", False)
            
            if format_check.get("has_command_tags", False):
                self.statistics["command_tag_usage"] += 1
            
            if format_check.get("has_room_tags", False):
                self.statistics["room_tag_usage"] += 1
            
            if format_check_passed:
                self.statistics["format_correct"] += 1
                game_record["metrics"]["format_check_passed"] += 1
            else:
                # Track format errors
                if not format_check.get("has_command_tags", False):
                    self.statistics["format_errors"]["missing_command_tags"] += 1
                if not format_check.get("has_room_tags", False):
                    self.statistics["format_errors"]["missing_room_tags"] += 1
            
            # Track token counts if available
            if hasattr(self.agent, 'tokenizer'):
                # Count input tokens
                prompt = self.agent.format_prompt(obs, infos["admissible_commands"], current_room)
                input_tokens = len(self.agent.tokenizer.encode(prompt))
                game_record["metrics"]["token_counts"]["input"].append(input_tokens)
                self.statistics["input_tokens"].append(input_tokens)
                
                # Count output tokens (if available in info)
                if "completion" in info:
                    output_tokens = len(self.agent.tokenizer.encode(info["completion"]))
                    game_record["metrics"]["token_counts"]["output"].append(output_tokens)
                    self.statistics["output_tokens"].append(output_tokens)
            
            # Track action validity
            if action in infos["admissible_commands"]:
                self.statistics["valid_actions"] += 1
            else:
                self.statistics["invalid_actions"] += 1
            
            # Track action frequency
            self.statistics["action_counts"][action] += 1
            
            # Record step
            step_record = {
                "step": step,
                "observation": obs,
                "action": action,
                "reward": reward,
                "next_observation": next_obs,
                "done": done,
                "current_room": current_room,
                "next_room": next_room,
                "format_check_passed": format_check_passed,
                "room_prediction": room_prediction,
                "room_prediction_correct": room_prediction_correct,
                "action_time": action_time
            }
            
            # Add beam search statistics if available
            if "all_completions" in info:
                step_record["all_completions"] = info["all_completions"]
                
                # Calculate beam diversity (Jaccard similarity between completions)
                if len(info["all_completions"]) > 1:
                    diversity_scores = []
                    for i in range(len(info["all_completions"])):
                        for j in range(i+1, len(info["all_completions"])):
                            comp1 = set(info["all_completions"][i].split())
                            comp2 = set(info["all_completions"][j].split())
                            jaccard = len(comp1.intersection(comp2)) / len(comp1.union(comp2))
                            diversity_scores.append(1 - jaccard)  # Higher is more diverse
                    
                    if diversity_scores:
                        avg_diversity = sum(diversity_scores) / len(diversity_scores)
                        step_record["beam_diversity"] = avg_diversity
                        self.statistics["beam_diversity"].append(avg_diversity)
                
                # Calculate format correctness across beam
                if "all_format_checks" in info:
                    format_correct_ratio = sum(1 for fc in info["all_format_checks"] 
                                              if fc.get("has_command_tags", False) and fc.get("has_room_tags", False)) / len(info["all_format_checks"])
                    step_record["beam_format_correctness"] = format_correct_ratio
                    self.statistics["beam_format_correctness"].append(format_correct_ratio)
            
            game_record["trajectory"].append(step_record)
            
            if log and verbose:
                self.log(f"\nStep {step}:")
                self.log(f"Room: {current_room}")
                self.log(f"Action: {action}")
                self.log(f"Reward: {reward}")
                self.log(f"Format check passed: {format_check_passed}")
                self.log(f"Room prediction: {room_prediction}")
                self.log(f"Room prediction correct: {room_prediction_correct}")
                self.log(f"Next room: {next_room}")
            
            # Update agent state
            self.agent.update_state_after_action(next_obs, reward, done, next_infos)
            
            # Update for next step
            obs, infos = next_obs, next_infos
            step += 1
            
            # Check for success
            if done and reward > 0:
                game_record["metrics"]["success"] = True
                if log:
                    self.log(f"\n🎉 Game completed successfully in {step} steps!")
            
        # Record final metrics
        game_record["metrics"]["steps"] = step
        game_record["metrics"]["score"] = reward if done and reward > 0 else 0
        game_record["metrics"]["end_time"] = time.time()
        game_record["metrics"]["completion_time"] = game_record["metrics"]["end_time"] - game_record["metrics"]["start_time"]
        
        # Update statistics
        self.statistics["games_played"] += 1
        self.statistics["total_steps"] += step
        
        if game_record["metrics"]["success"]:
            self.statistics["games_succeeded"] += 1
            self.statistics["successful_steps"] += step
            self.statistics["total_score"] += game_record["metrics"]["score"]
            self.statistics["completion_times"].append(game_record["metrics"]["completion_time"])
        
        # Update per-difficulty statistics
        diff_stats = self.statistics["per_difficulty"][difficulty]
        diff_stats["games_played"] += 1
        diff_stats["total_steps"] += step
        diff_stats["total_actions"] += step
        diff_stats["format_correct"] += game_record["metrics"]["format_check_passed"]
        
        if game_record["metrics"]["success"]:
            diff_stats["games_succeeded"] += 1
            diff_stats["successful_steps"] += step
            diff_stats["total_score"] += game_record["metrics"]["score"]
            diff_stats["completion_times"].append(game_record["metrics"]["completion_time"])
        
        if log:
            self.log(f"\nGame completed in {step} steps")
            self.log(f"Success: {game_record['metrics']['success']}")
            self.log(f"Score: {game_record['metrics']['score']}")
            self.log(f"Format check passed: {game_record['metrics']['format_check_passed']}/{step} steps ({game_record['metrics']['format_check_passed']/step*100:.1f}%)")
            
            if hasattr(self.agent, 'tokenizer'):
                avg_input_tokens = sum(game_record["metrics"]["token_counts"]["input"]) / len(game_record["metrics"]["token_counts"]["input"])
                self.log(f"Average input tokens: {avg_input_tokens:.1f}")
                
                if game_record["metrics"]["token_counts"]["output"]:
                    avg_output_tokens = sum(game_record["metrics"]["token_counts"]["output"]) / len(game_record["metrics"]["token_counts"]["output"])
                    self.log(f"Average output tokens: {avg_output_tokens:.1f}")
        
        # Clean up
        env.close()
        
        return game_record
    
    def evaluate_on_difficulties(
        self, 
        difficulties: List[int], 
        games_per_difficulty: int = 1,
        max_steps: int = 20,
        verbose: bool = False,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Evaluate the agent on multiple difficulty levels
        
        Args:
            difficulties: List of difficulty levels
            games_per_difficulty: Number of games to play per difficulty
            max_steps: Maximum steps per game
            verbose: Whether to print detailed information
            save_results: Whether to save results to file
            
        Returns:
            Dictionary with evaluation results
        """
        # Reset statistics
        self.reset_statistics()
        
        # Start time
        start_time = time.time()
        
        # Create timestamp for this evaluation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Log evaluation parameters
        self.log(f"\n{'='*50}")
        self.log(f"Starting evaluation on difficulties: {difficulties}")
        self.log(f"Games per difficulty: {games_per_difficulty}")
        self.log(f"Max steps per game: {max_steps}")
        self.log(f"{'='*50}\n")
        
        # Play games at each difficulty
        all_game_records = []
        
        for difficulty in difficulties:
            self.log(f"\n{'='*50}")
            self.log(f"Evaluating difficulty level {difficulty}")
            self.log(f"{'='*50}\n")
            
            for game_num in range(games_per_difficulty):
                self.log(f"\nPlaying game {game_num+1}/{games_per_difficulty} at difficulty {difficulty}")
                
                try:
                    # Play a game at this difficulty
                    game_record = self.play_game(
                        difficulty=difficulty, 
                        max_steps=max_steps,
                        log=True,
                        verbose=verbose
                    )
                    
                    # Add to all game records
                    all_game_records.append(game_record)
                    
                except Exception as e:
                    self.log(f"Error playing game at difficulty {difficulty}: {str(e)}")
                    # Create a failed game record
                    failed_game = {
                        "difficulty": difficulty,
                        "trajectory": [],
                        "metrics": {
                            "success": False,
                            "steps": 0,
                            "score": 0,
                            "format_check_passed": 0,
                            "room_prediction_correct": 0,
                            "error": str(e)
                        }
                    }
                    all_game_records.append(failed_game)
                    
                    # Update statistics for failed game
                    self.statistics["games_played"] += 1
                    self.statistics["per_difficulty"][difficulty]["games_played"] += 1
        
        # Calculate evaluation time
        eval_time = time.time() - start_time
        
        # Compile results
        results = self.compile_evaluation_results(all_game_records, eval_time)
        
        # Print summary
        self.print_evaluation_summary(results)
        
        # Save results if requested
        if save_results:
            results_path = self.save_evaluation_results(results, timestamp)
            self.log(f"\nResults saved to {results_path}")
        
        return results
    
    def compile_evaluation_results(self, game_records: List[Dict], eval_time: float) -> Dict[str, Any]:
        """Compile evaluation results from game records
        
        Args:
            game_records: List of game records
            eval_time: Total evaluation time
            
        Returns:
            Dictionary with compiled results
        """
        # Basic statistics
        total_games = self.statistics["games_played"]
        successful_games = self.statistics["games_succeeded"]
        success_rate = successful_games / total_games if total_games > 0 else 0
        
        # Calculate average steps
        avg_steps = self.statistics["total_steps"] / total_games if total_games > 0 else 0
        avg_successful_steps = self.statistics["successful_steps"] / successful_games if successful_games > 0 else 0
        
        # Format statistics
        total_actions = self.statistics["total_steps"]
        format_correct_rate = self.statistics["format_correct"] / total_actions if total_actions > 0 else 0
        command_tag_rate = self.statistics["command_tag_usage"] / total_actions if total_actions > 0 else 0
        room_tag_rate = self.statistics["room_tag_usage"] / total_actions if total_actions > 0 else 0
        
        # Room prediction statistics
        room_prediction_accuracy = (self.statistics["room_predictions_correct"] / 
                                   self.statistics["room_predictions"] if self.statistics["room_predictions"] > 0 else 0)
        new_room_prediction_accuracy = (self.statistics["new_room_predictions_correct"] / 
                                       self.statistics["new_room_predictions"] if self.statistics["new_room_predictions"] > 0 else 0)
        
        # Token statistics
        avg_input_tokens = sum(self.statistics["input_tokens"]) / len(self.statistics["input_tokens"]) if self.statistics["input_tokens"] else 0
        avg_output_tokens = sum(self.statistics["output_tokens"]) / len(self.statistics["output_tokens"]) if self.statistics["output_tokens"] else 0
        
        # Action statistics
        valid_action_rate = self.statistics["valid_actions"] / total_actions if total_actions > 0 else 0
        most_common_actions = self.statistics["action_counts"].most_common(5)
        
        # Beam search statistics
        avg_beam_diversity = sum(self.statistics["beam_diversity"]) / len(self.statistics["beam_diversity"]) if self.statistics["beam_diversity"] else 0
        avg_beam_format_correctness = sum(self.statistics["beam_format_correctness"]) / len(self.statistics["beam_format_correctness"]) if self.statistics["beam_format_correctness"] else 0
        
        # Per difficulty statistics
        per_difficulty_results = {}
        for difficulty, stats in self.statistics["per_difficulty"].items():
            games_played = stats["games_played"]
            if games_played == 0:
                continue
                
            games_succeeded = stats["games_succeeded"]
            success_rate_diff = games_succeeded / games_played if games_played > 0 else 0
            avg_steps_diff = stats["total_steps"] / games_played if games_played > 0 else 0
            avg_successful_steps_diff = stats["successful_steps"] / games_succeeded if games_succeeded > 0 else 0
            format_correct_rate_diff = stats["format_correct"] / stats["total_actions"] if stats["total_actions"] > 0 else 0
            
            per_difficulty_results[difficulty] = {
                "games_played": games_played,
                "games_succeeded": games_succeeded,
                "success_rate": success_rate_diff,
                "avg_steps": avg_steps_diff,
                "avg_successful_steps": avg_successful_steps_diff,
                "format_correct_rate": format_correct_rate_diff
            }
        
        # Compile results
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation_time": eval_time,
            "overall": {
                "games_played": total_games,
                "games_succeeded": successful_games,
                "success_rate": success_rate,
                "avg_steps": avg_steps,
                "avg_successful_steps": avg_successful_steps,
                "format_correct_rate": format_correct_rate,
                "command_tag_rate": command_tag_rate,
                "room_tag_rate": room_tag_rate,
                "room_prediction_accuracy": room_prediction_accuracy,
                "new_room_prediction_accuracy": new_room_prediction_accuracy,
                "avg_input_tokens": avg_input_tokens,
                "avg_output_tokens": avg_output_tokens,
                "valid_action_rate": valid_action_rate,
                "most_common_actions": most_common_actions,
                "avg_beam_diversity": avg_beam_diversity,
                "avg_beam_format_correctness": avg_beam_format_correctness
            },
            "per_difficulty": per_difficulty_results,
            "format_errors": dict(self.statistics["format_errors"]),
            "game_records": game_records
        }
        
        return results
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print a summary of evaluation results
        
        Args:
            results: Dictionary with evaluation results
        """
        overall = results["overall"]
        per_difficulty = results["per_difficulty"]
        
        self.log("\n\n" + "="*50)
        self.log("Evaluation Summary")
        self.log("="*50)
        
        self.log(f"\nOverall Statistics:")
        self.log(f"  Games Played: {overall['games_played']}")
        self.log(f"  Success Rate: {overall['success_rate']*100:.1f}%")
        self.log(f"  Average Steps: {overall['avg_steps']:.1f}")
        self.log(f"  Average Steps (Successful Games): {overall['avg_successful_steps']:.1f}")
        self.log(f"  Format Correctness: {overall['format_correct_rate']*100:.1f}%")
        self.log(f"  Command Tag Usage: {overall['command_tag_rate']*100:.1f}%")
        self.log(f"  Room Tag Usage: {overall['room_tag_rate']*100:.1f}%")
        self.log(f"  Room Prediction Accuracy: {overall['room_prediction_accuracy']*100:.1f}%")
        self.log(f"  New Room Prediction Accuracy: {overall['new_room_prediction_accuracy']*100:.1f}%")
        self.log(f"  Valid Action Rate: {overall['valid_action_rate']*100:.1f}%")
        
        if overall['avg_input_tokens'] > 0:
            self.log(f"  Average Input Tokens: {overall['avg_input_tokens']:.1f}")
        if overall['avg_output_tokens'] > 0:
            self.log(f"  Average Output Tokens: {overall['avg_output_tokens']:.1f}")
        
        if overall['most_common_actions']:
            self.log("\n  Most Common Actions:")
            for action, count in overall['most_common_actions']:
                self.log(f"    - {action}: {count} times")
        
        if results["format_errors"]:
            self.log("\n  Format Errors:")
            for error_type, count in results["format_errors"].items():
                self.log(f"    - {error_type}: {count} times")
        
        self.log("\nPer Difficulty Statistics:")
        for difficulty, stats in sorted(per_difficulty.items()):
            self.log(f"\n  Difficulty {difficulty}:")
            self.log(f"    Games Played: {stats['games_played']}")
            self.log(f"    Success Rate: {stats['success_rate']*100:.1f}%")
            self.log(f"    Average Steps: {stats['avg_steps']:.1f}")
            self.log(f"    Format Correctness: {stats['format_correct_rate']*100:.1f}%")
        
        self.log(f"\nEvaluation completed in {results['evaluation_time']:.1f} seconds")
    
    def save_evaluation_results(self, results: Dict[str, Any], timestamp: str) -> str:
        """Save evaluation results to file
        
        Args:
            results: Dictionary with evaluation results
            timestamp: Timestamp string
            
        Returns:
            Path to saved results file
        """
        # Create results directory
        results_dir = os.path.join(self.log_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed JSON results
        json_path = os.path.join(results_dir, f"evaluation_results_{timestamp}.json")
        
        # Create a copy of results without game records to save space
        results_summary = {k: v for k, v in results.items() if k != "game_records"}
        
        with open(json_path, "w") as f:
            json.dump(results_summary, f, indent=2)
        
        # Save human-readable summary
        summary_path = os.path.join(results_dir, f"evaluation_summary_{timestamp}.txt")
        
        with open(summary_path, "w") as f:
            f.write("Evaluation Results\n")
            f.write("="*50 + "\n\n")
            
            overall = results["overall"]
            f.write("Overall Statistics:\n")
            f.write(f"  Games Played: {overall['games_played']}\n")
            f.write(f"  Success Rate: {overall['success_rate']*100:.1f}%\n")
            f.write(f"  Average Steps: {overall['avg_steps']:.1f}\n")
            f.write(f"  Average Steps (Successful Games): {overall['avg_successful_steps']:.1f}\n")
            f.write(f"  Format Correctness: {overall['format_correct_rate']*100:.1f}%\n")
            f.write(f"  Command Tag Usage: {overall['command_tag_rate']*100:.1f}%\n")
            f.write(f"  Room Tag Usage: {overall['room_tag_rate']*100:.1f}%\n")
            f.write(f"  Room Prediction Accuracy: {overall['room_prediction_accuracy']*100:.1f}%\n")
            f.write(f"  New Room Prediction Accuracy: {overall['new_room_prediction_accuracy']*100:.1f}%\n")
            f.write(f"  Valid Action Rate: {overall['valid_action_rate']*100:.1f}%\n")
            
            if overall['avg_input_tokens'] > 0:
                f.write(f"  Average Input Tokens: {overall['avg_input_tokens']:.1f}\n")
            if overall['avg_output_tokens'] > 0:
                f.write(f"  Average Output Tokens: {overall['avg_output_tokens']:.1f}\n")
            
            if overall['most_common_actions']:
                f.write("\n  Most Common Actions:\n")
                for action, count in overall['most_common_actions']:
                    f.write(f"    - {action}: {count} times\n")
            
            if results["format_errors"]:
                f.write("\n  Format Errors:\n")
                for error_type, count in results["format_errors"].items():
                    f.write(f"    - {error_type}: {count} times\n")
            
            f.write("\nPer Difficulty Statistics:\n")
            for difficulty, stats in sorted(results["per_difficulty"].items()):
                f.write(f"\n  Difficulty {difficulty}:\n")
                f.write(f"    Games Played: {stats['games_played']}\n")
                f.write(f"    Success Rate: {stats['success_rate']*100:.1f}%\n")
                f.write(f"    Average Steps: {stats['avg_steps']:.1f}\n")
                f.write(f"    Format Correctness: {stats['format_correct_rate']*100:.1f}%\n")
            
            f.write(f"\nEvaluation completed in {results['evaluation_time']:.1f} seconds\n")
        
        # Generate CSV for easy import into spreadsheets
        csv_path = os.path.join(results_dir, f"evaluation_metrics_{timestamp}.csv")
        
        # Prepare data for CSV
        csv_data = []
        
        # Add overall metrics
        overall_row = {
            "difficulty": "overall",
            "games_played": overall["games_played"],
            "success_rate": overall["success_rate"],
            "avg_steps": overall["avg_steps"],
            "format_correct_rate": overall["format_correct_rate"],
            "command_tag_rate": overall["command_tag_rate"],
            "room_tag_rate": overall["room_tag_rate"],
            "room_prediction_accuracy": overall["room_prediction_accuracy"],
            "valid_action_rate": overall["valid_action_rate"]
        }
        csv_data.append(overall_row)
        
        # Add per-difficulty metrics
        for difficulty, stats in sorted(results["per_difficulty"].items()):
            diff_row = {
                "difficulty": difficulty,
                "games_played": stats["games_played"],
                "success_rate": stats["success_rate"],
                "avg_steps": stats["avg_steps"],
                "format_correct_rate": stats["format_correct_rate"]
            }
            csv_data.append(diff_row)
        
        # Write CSV
        try:
            import pandas as pd
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
        except ImportError:
            # Fallback if pandas is not available
            with open(csv_path, "w") as f:
                # Write header
                f.write(",".join(csv_data[0].keys()) + "\n")
                # Write rows
                for row in csv_data:
                    f.write(",".join(str(v) for v in row.values()) + "\n")
        
        return summary_path

    def compare_models(
        self,
        model_paths: List[str],
        model_names: List[str],
        difficulties: List[int],
        games_per_difficulty: int = 1,
        max_steps: int = 20,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Compare multiple models on the same set of games
        
        Args:
            model_paths: List of paths to model checkpoints
            model_names: List of names for the models (for display)
            difficulties: List of difficulty levels
            games_per_difficulty: Number of games per difficulty
            max_steps: Maximum steps per game
            save_results: Whether to save results
            
        Returns:
            Dictionary with comparison results
        """
        if len(model_paths) != len(model_names):
            raise ValueError("Number of model paths must match number of model names")
        
        # Create timestamp for this comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Log comparison parameters
        self.log(f"\n{'='*50}")
        self.log(f"Starting model comparison")
        self.log(f"Models: {model_names}")
        self.log(f"Difficulties: {difficulties}")
        self.log(f"Games per difficulty: {games_per_difficulty}")
        self.log(f"{'='*50}\n")
        
        # Store results for each model
        all_results = {}
        
        # Evaluate each model
        for i, (model_path, model_name) in enumerate(zip(model_paths, model_names)):
            self.log(f"\n{'='*50}")
            self.log(f"Evaluating model {i+1}/{len(model_paths)}: {model_name}")
            self.log(f"{'='*50}\n")
            
            # Load model
            try:
                # Create a new agent with this model
                agent = TextWorldLLMAgent(
                    config=self.config,
                    model_path=model_path,
                    use_map=getattr(self.agent, 'use_map', False)
                )
                
                # Update the game runner's agent
                original_agent = self.agent
                self.agent = agent
                
                # Evaluate the model
                results = self.evaluate_on_difficulties(
                    difficulties=difficulties,
                    games_per_difficulty=games_per_difficulty,
                    max_steps=max_steps,
                    verbose=False,
                    save_results=False
                )
                
                # Store results
                all_results[model_name] = results
                
                # Restore original agent
                self.agent = original_agent
                
            except Exception as e:
                self.log(f"Error evaluating model {model_name}: {str(e)}")
                all_results[model_name] = {"error": str(e)}
        
        # Compile comparison results
        comparison = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models": model_names,
            "difficulties": difficulties,
            "games_per_difficulty": games_per_difficulty,
            "results": all_results
        }
        
        # Print comparison summary
        self.print_model_comparison(comparison)
        
        # Save comparison if requested
        if save_results:
            comparison_path = self.save_model_comparison(comparison, timestamp)
            self.log(f"\nComparison saved to {comparison_path}")
        
        return comparison

    def print_model_comparison(self, comparison: Dict[str, Any]):
        """Print a summary of model comparison results
        
        Args:
            comparison: Dictionary with comparison results
        """
        models = comparison["models"]
        results = comparison["results"]
        
        self.log("\n\n" + "="*50)
        self.log("Model Comparison Summary")
        self.log("="*50)
        
        # Check if we have valid results for all models
        valid_models = [model for model in models if "error" not in results[model]]
        
        if not valid_models:
            self.log("\nNo valid results to compare")
            return
        
        # Create comparison table for key metrics
        metrics = [
            "success_rate", 
            "avg_steps", 
            "format_correct_rate", 
            "room_prediction_accuracy",
            "valid_action_rate"
        ]
        
        metric_names = {
            "success_rate": "Success Rate (%)",
            "avg_steps": "Avg Steps",
            "format_correct_rate": "Format Correct (%)",
            "room_prediction_accuracy": "Room Prediction (%)",
            "valid_action_rate": "Valid Actions (%)"
        }
        
        # Print header
        header = "Metric".ljust(25)
        for model in valid_models:
            header += model.ljust(20)
        self.log("\n" + header)
        self.log("-" * (25 + 20 * len(valid_models)))
        
        # Print each metric
        for metric in metrics:
            row = metric_names[metric].ljust(25)
            for model in valid_models:
                if metric in ["success_rate", "format_correct_rate", "room_prediction_accuracy", "valid_action_rate"]:
                    value = f"{results[model]['overall'][metric]*100:.1f}%"
                else:
                    value = f"{results[model]['overall'][metric]:.1f}"
                row += value.ljust(20)
            self.log(row)
        
        # Print per-difficulty success rates
        self.log("\n" + "="*50)
        self.log("Success Rate by Difficulty")
        self.log("="*50)
        
        # Get all difficulties across all models
        all_difficulties = set()
        for model in valid_models:
            all_difficulties.update(results[model]["per_difficulty"].keys())
        
        # Print header
        header = "Difficulty".ljust(15)
        for model in valid_models:
            header += model.ljust(20)
        self.log("\n" + header)
        self.log("-" * (15 + 20 * len(valid_models)))
        
        # Print success rate for each difficulty
        for difficulty in sorted(all_difficulties):
            row = f"{difficulty}".ljust(15)
            for model in valid_models:
                if (str(difficulty) in results[model]["per_difficulty"] or 
                    difficulty in results[model]["per_difficulty"]):
                    diff_key = str(difficulty) if str(difficulty) in results[model]["per_difficulty"] else difficulty
                    value = f"{results[model]['per_difficulty'][diff_key]['success_rate']*100:.1f}%"
                else:
                    value = "N/A"
                row += value.ljust(20)
            self.log(row)

    def save_model_comparison(self, comparison: Dict[str, Any], timestamp: str) -> str:
        """Save model comparison results to file
    
    Args:
            comparison: Dictionary with comparison results
            timestamp: Timestamp string
            
        Returns:
            Path to saved comparison file
        """
        # Create comparison directory
        comparison_dir = os.path.join(self.log_dir, "comparisons")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Save detailed JSON results
        json_path = os.path.join(comparison_dir, f"model_comparison_{timestamp}.json")
        
        with open(json_path, "w") as f:
            json.dump(comparison, f, indent=2)
        
        # Save human-readable summary
        summary_path = os.path.join(comparison_dir, f"model_comparison_summary_{timestamp}.txt")
        
        with open(summary_path, "w") as f:
            models = comparison["models"]
            results = comparison["results"]
            
            f.write("Model Comparison Results\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Models compared: {', '.join(models)}\n")
            f.write(f"Difficulties: {comparison['difficulties']}\n")
            f.write(f"Games per difficulty: {comparison['games_per_difficulty']}\n\n")
            
            # Check if we have valid results for all models
            valid_models = [model for model in models if "error" not in results[model]]
            
            if not valid_models:
                f.write("\nNo valid results to compare\n")
                return summary_path
            
            # Create comparison table for key metrics
            metrics = [
                "success_rate", 
                "avg_steps", 
                "format_correct_rate", 
                "room_prediction_accuracy",
                "valid_action_rate"
            ]
            
            metric_names = {
                "success_rate": "Success Rate (%)",
                "avg_steps": "Avg Steps",
                "format_correct_rate": "Format Correct (%)",
                "room_prediction_accuracy": "Room Prediction (%)",
                "valid_action_rate": "Valid Actions (%)"
            }
            
            # Write header
            header = "Metric".ljust(25)
            for model in valid_models:
                header += model.ljust(20)
            f.write("\n" + header + "\n")
            f.write("-" * (25 + 20 * len(valid_models)) + "\n")
            
            # Write each metric
            for metric in metrics:
                row = metric_names[metric].ljust(25)
                for model in valid_models:
                    if metric in ["success_rate", "format_correct_rate", "room_prediction_accuracy", "valid_action_rate"]:
                        value = f"{results[model]['overall'][metric]*100:.1f}%"
                    else:
                        value = f"{results[model]['overall'][metric]:.1f}"
                    row += value.ljust(20)
                f.write(row + "\n")
            
            # Write per-difficulty success rates
            f.write("\n" + "="*50 + "\n")
            f.write("Success Rate by Difficulty\n")
            f.write("="*50 + "\n")
            
            # Get all difficulties across all models
            all_difficulties = set()
            for model in valid_models:
                all_difficulties.update(results[model]["per_difficulty"].keys())
            
            # Write header
            header = "Difficulty".ljust(15)
            for model in valid_models:
                header += model.ljust(20)
            f.write("\n" + header + "\n")
            f.write("-" * (15 + 20 * len(valid_models)) + "\n")
            
            # Write success rate for each difficulty
            for difficulty in sorted(all_difficulties):
                row = f"{difficulty}".ljust(15)
                for model in valid_models:
                    if (str(difficulty) in results[model]["per_difficulty"] or 
                        difficulty in results[model]["per_difficulty"]):
                        diff_key = str(difficulty) if str(difficulty) in results[model]["per_difficulty"] else difficulty
                        value = f"{results[model]['per_difficulty'][diff_key]['success_rate']*100:.1f}%"
                    else:
                        value = "N/A"
                    row += value.ljust(20)
                f.write(row + "\n")
        
        # Generate CSV for easy import into spreadsheets
        csv_path = os.path.join(comparison_dir, f"model_comparison_{timestamp}.csv")
        
        try:
            # Create DataFrame for comparison
            import pandas as pd
            
            # Prepare data for overall metrics
            overall_data = []
            
            for model in valid_models:
                model_data = {"model": model}
                for metric in metrics:
                    if metric in results[model]["overall"]:
                        if metric in ["success_rate", "format_correct_rate", "room_prediction_accuracy", "valid_action_rate"]:
                            model_data[metric] = results[model]["overall"][metric] * 100  # Convert to percentage
                        else:
                            model_data[metric] = results[model]["overall"][metric]
                overall_data.append(model_data)
            
            # Create and save overall metrics DataFrame
            overall_df = pd.DataFrame(overall_data)
            overall_df.to_csv(csv_path, index=False)
            
            # Prepare data for per-difficulty metrics
            difficulty_data = []
            
            for difficulty in sorted(all_difficulties):
                for model in valid_models:
                    if (str(difficulty) in results[model]["per_difficulty"] or 
                        difficulty in results[model]["per_difficulty"]):
                        diff_key = str(difficulty) if str(difficulty) in results[model]["per_difficulty"] else difficulty
                        diff_stats = results[model]["per_difficulty"][diff_key]
                        
                        row_data = {
                            "model": model,
                            "difficulty": difficulty,
                            "success_rate": diff_stats["success_rate"] * 100,  # Convert to percentage
                            "avg_steps": diff_stats["avg_steps"],
                            "format_correct_rate": diff_stats["format_correct_rate"] * 100  # Convert to percentage
                        }
                        difficulty_data.append(row_data)
            
            # Create and save per-difficulty DataFrame
            difficulty_df = pd.DataFrame(difficulty_data)
            difficulty_df.to_csv(os.path.join(comparison_dir, f"model_comparison_by_difficulty_{timestamp}.csv"), index=False)
            
        except ImportError:
            # Fallback if pandas is not available
            with open(csv_path, "w") as f:
                # Write header
                header = "model," + ",".join(metrics)
                f.write(header + "\n")
                
                # Write data for each model
                for model in valid_models:
                    row = model
                    for metric in metrics:
                        if metric in results[model]["overall"]:
                            if metric in ["success_rate", "format_correct_rate", "room_prediction_accuracy", "valid_action_rate"]:
                                value = results[model]["overall"][metric] * 100  # Convert to percentage
                            else:
                                value = results[model]["overall"][metric]
                            row += f",{value}"
                        else:
                            row += ",N/A"
                    f.write(row + "\n")
        
        return summary_path

    def analyze_beam_search(
        self, 
        difficulty: int = 5, 
        num_games: int = 1,
        max_steps: int = 20,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Analyze beam search behavior in detail
        
        Args:
            difficulty: Difficulty level
            num_games: Number of games to play
            max_steps: Maximum steps per game
            save_results: Whether to save results
            
        Returns:
            Dictionary with beam search analysis
        """
        # Create timestamp for this analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Log analysis parameters
        self.log(f"\n{'='*50}")
        self.log(f"Starting beam search analysis")
        self.log(f"Difficulty: {difficulty}")
        self.log(f"Number of games: {num_games}")
        self.log(f"{'='*50}\n")
        
        # Store all completions and their metrics
        all_completions = []
        
        # Play games and collect beam search data
        for game_num in range(num_games):
            self.log(f"\nPlaying game {game_num+1}/{num_games} at difficulty {difficulty}")
            
            try:
                # Create environment
                env = self.env_manager.get_or_create_env(difficulty)
                
                # Reset agent
                self.agent.reset()
                
                # Start game
                obs, infos = env.reset()
                done = False
                step = 0
                
                # Extract goal
                if self.agent.goal is None:
                    self.agent.goal = self.agent.parse_goal(obs)
                
                # Game loop
                while not done and step < max_steps:
                    # Get current room
                    current_room = self.agent._get_room_name(obs)
                    
                    # Format prompt
                    prompt = self.agent.format_prompt(obs, infos["admissible_commands"], current_room)
                    
                    # Generate multiple completions with beam search
                    completions = self.agent.generate_response(
                        prompt,
                        num_beams=self.eval_config.num_beams,
                        num_return_sequences=self.eval_config.num_return_sequences,
                        do_sample=self.eval_config.do_sample,
                        temperature=self.eval_config.temperature,
                        top_p=self.eval_config.top_p,
                        top_k=self.eval_config.top_k
                    )
                    
                    # Check format for each completion
                    format_checks = [self.agent.check_format(comp) for comp in completions]
                    
                    # Extract actions for each completion
                    actions = []
                    for i, completion in enumerate(completions):
                        format_check = format_checks[i]
                        if format_check["has_command_tags"] and format_check["command"] in infos["admissible_commands"]:
                            action = format_check["command"]
                        else:
                            # Find any valid action in the completion
                            action = next((a for a in infos["admissible_commands"] if a in completion), None)
                        actions.append(action)
                    
                    # Store completion data
                    for i, completion in enumerate(completions):
                        format_check = format_checks[i]
                        action = actions[i]
                        
                        # Count tokens
                        input_tokens = len(self.agent.tokenizer.encode(prompt))
                        output_tokens = len(self.agent.tokenizer.encode(completion))
                        
                        completion_data = {
                            "game": game_num,
                            "step": step,
                            "beam_idx": i,
                            "completion": completion,
                            "has_command_tags": format_check["has_command_tags"],
                            "has_room_tags": format_check["has_room_tags"],
                            "command": format_check["command"],
                            "room": format_check["room"],
                            "action": action,
                            "action_valid": action in infos["admissible_commands"] if action else False,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens
                        }
                        
                        all_completions.append(completion_data)
                    
                    # Take action with the first valid action
                    valid_action_idx = next((i for i, a in enumerate(actions) if a in infos["admissible_commands"]), 0)
                    action = actions[valid_action_idx] if valid_action_idx < len(actions) else infos["admissible_commands"][0]
                    
                    # Take action in environment
                    obs, reward, done, infos = env.step(action)
                    
                    # Update agent state
                    self.agent.update_state_after_action(obs, reward, done, infos)
                    
                    step += 1
                
                # Clean up
                env.close()
                
            except Exception as e:
                self.log(f"Error during beam search analysis: {str(e)}")
        
        # Analyze beam search data
        analysis = self._analyze_beam_search_data(all_completions)
        
        # Print analysis summary
        self._print_beam_search_analysis(analysis)
        
        # Save analysis if requested
        if save_results:
            analysis_path = self._save_beam_search_analysis(analysis, all_completions, timestamp)
            self.log(f"\nBeam search analysis saved to {analysis_path}")
        
        return analysis

    def _analyze_beam_search_data(self, completions: List[Dict]) -> Dict[str, Any]:
        """Analyze beam search data
        
        Args:
            completions: List of completion data dictionaries
            
        Returns:
            Dictionary with analysis results
        """
        # Calculate overall statistics
        total_completions = len(completions)
        
        # Format statistics
        command_tag_count = sum(1 for c in completions if c["has_command_tags"])
        room_tag_count = sum(1 for c in completions if c["has_room_tags"])
        both_tags_count = sum(1 for c in completions if c["has_command_tags"] and c["has_room_tags"])
        
        # Action validity
        valid_action_count = sum(1 for c in completions if c["action_valid"])
        
        # Token statistics
        input_tokens = [c["input_tokens"] for c in completions]
        output_tokens = [c["output_tokens"] for c in completions]
        
        # Beam position statistics
        by_beam_position = {}
        beam_positions = set(c["beam_idx"] for c in completions)
        
        for position in sorted(beam_positions):
            position_completions = [c for c in completions if c["beam_idx"] == position]
            position_count = len(position_completions)
            
            command_tag_rate = sum(1 for c in position_completions if c["has_command_tags"]) / position_count
            room_tag_rate = sum(1 for c in position_completions if c["has_room_tags"]) / position_count
            both_tags_rate = sum(1 for c in position_completions if c["has_command_tags"] and c["has_room_tags"]) / position_count
            valid_action_rate = sum(1 for c in position_completions if c["action_valid"]) / position_count
            
            by_beam_position[position] = {
                "count": position_count,
                "command_tag_rate": command_tag_rate,
                "room_tag_rate": room_tag_rate,
                "both_tags_rate": both_tags_rate,
                "valid_action_rate": valid_action_rate,
                "avg_output_tokens": sum(c["output_tokens"] for c in position_completions) / position_count
            }
        
        # Diversity analysis
        # Calculate average Jaccard similarity between completions in the same step
        diversity_scores = []
        
        # Group completions by game and step
        by_game_step = {}
        for c in completions:
            key = (c["game"], c["step"])
            if key not in by_game_step:
                by_game_step[key] = []
            by_game_step[key].append(c)
        
        # Calculate diversity for each step
        for game_step, step_completions in by_game_step.items():
            if len(step_completions) <= 1:
                continue
                
            step_diversity_scores = []
            for i in range(len(step_completions)):
                for j in range(i+1, len(step_completions)):
                    comp1 = set(step_completions[i]["completion"].split())
                    comp2 = set(step_completions[j]["completion"].split())
                    
                    if not comp1 or not comp2:
                        continue
                        
                    jaccard = len(comp1.intersection(comp2)) / len(comp1.union(comp2))
                    step_diversity_scores.append(1 - jaccard)  # Higher is more diverse
            
            if step_diversity_scores:
                avg_step_diversity = sum(step_diversity_scores) / len(step_diversity_scores)
                diversity_scores.append(avg_step_diversity)
        
        avg_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0
        
        # Compile analysis results
        analysis = {
            "total_completions": total_completions,
            "command_tag_rate": command_tag_count / total_completions if total_completions > 0 else 0,
            "room_tag_rate": room_tag_count / total_completions if total_completions > 0 else 0,
            "both_tags_rate": both_tags_count / total_completions if total_completions > 0 else 0,
            "valid_action_rate": valid_action_count / total_completions if total_completions > 0 else 0,
            "avg_input_tokens": sum(input_tokens) / len(input_tokens) if input_tokens else 0,
            "avg_output_tokens": sum(output_tokens) / len(output_tokens) if output_tokens else 0,
            "max_output_tokens": max(output_tokens) if output_tokens else 0,
            "avg_diversity": avg_diversity,
            "by_beam_position": by_beam_position
        }
        
        return analysis

    def _print_beam_search_analysis(self, analysis: Dict[str, Any]):
        """Print beam search analysis summary
        
        Args:
            analysis: Dictionary with analysis results
        """
        self.log("\n\n" + "="*50)
        self.log("Beam Search Analysis")
        self.log("="*50)
        
        self.log(f"\nTotal completions analyzed: {analysis['total_completions']}")
        self.log(f"Command tag usage: {analysis['command_tag_rate']*100:.1f}%")
        self.log(f"Room tag usage: {analysis['room_tag_rate']*100:.1f}%")
        self.log(f"Both tags usage: {analysis['both_tags_rate']*100:.1f}%")
        self.log(f"Valid action rate: {analysis['valid_action_rate']*100:.1f}%")
        self.log(f"Average input tokens: {analysis['avg_input_tokens']:.1f}")
        self.log(f"Average output tokens: {analysis['avg_output_tokens']:.1f}")
        self.log(f"Maximum output tokens: {analysis['max_output_tokens']}")
        self.log(f"Average diversity (Jaccard): {analysis['avg_diversity']:.3f}")
        
        self.log("\nPerformance by Beam Position:")
        self.log("-" * 80)
        header = "Position".ljust(10) + "Command Tags".ljust(15) + "Room Tags".ljust(15) + "Both Tags".ljust(15) + "Valid Actions".ljust(15) + "Avg Tokens".ljust(15)
        self.log(header)
        self.log("-" * 80)
        
        for position, stats in sorted(analysis["by_beam_position"].items()):
            row = f"{position}".ljust(10)
            row += f"{stats['command_tag_rate']*100:.1f}%".ljust(15)
            row += f"{stats['room_tag_rate']*100:.1f}%".ljust(15)
            row += f"{stats['both_tags_rate']*100:.1f}%".ljust(15)
            row += f"{stats['valid_action_rate']*100:.1f}%".ljust(15)
            row += f"{stats['avg_output_tokens']:.1f}".ljust(15)
            self.log(row)

    def _save_beam_search_analysis(self, analysis: Dict[str, Any], completions: List[Dict], timestamp: str) -> str:
        """Save beam search analysis to file
        
        Args:
            analysis: Dictionary with analysis results
            completions: List of completion data dictionaries
            timestamp: Timestamp string
            
        Returns:
            Path to saved analysis file
        """
        # Create analysis directory
        analysis_dir = os.path.join(self.log_dir, "beam_analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Save detailed JSON analysis
        json_path = os.path.join(analysis_dir, f"beam_analysis_{timestamp}.json")
        
        with open(json_path, "w") as f:
            json.dump(analysis, f, indent=2)
        
        # Save human-readable summary
        summary_path = os.path.join(analysis_dir, f"beam_analysis_summary_{timestamp}.txt")
        
        with open(summary_path, "w") as f:
            f.write("Beam Search Analysis\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Total completions analyzed: {analysis['total_completions']}\n")
            f.write(f"Command tag usage: {analysis['command_tag_rate']*100:.1f}%\n")
            f.write(f"Room tag usage: {analysis['room_tag_rate']*100:.1f}%\n")
            f.write(f"Both tags usage: {analysis['both_tags_rate']*100:.1f}%\n")
            f.write(f"Valid action rate: {analysis['valid_action_rate']*100:.1f}%\n")
            f.write(f"Average input tokens: {analysis['avg_input_tokens']:.1f}\n")
            f.write(f"Average output tokens: {analysis['avg_output_tokens']:.1f}\n")
            f.write(f"Maximum output tokens: {analysis['max_output_tokens']}\n")
            f.write(f"Average diversity (Jaccard): {analysis['avg_diversity']:.3f}\n\n")
            
            f.write("Performance by Beam Position:\n")
            f.write("-" * 80 + "\n")
            header = "Position".ljust(10) + "Command Tags".ljust(15) + "Room Tags".ljust(15) + "Both Tags".ljust(15) + "Valid Actions".ljust(15) + "Avg Tokens".ljust(15)
            f.write(header + "\n")
            f.write("-" * 80 + "\n")
            
            for position, stats in sorted(analysis["by_beam_position"].items()):
                row = f"{position}".ljust(10)
                row += f"{stats['command_tag_rate']*100:.1f}%".ljust(15)
                row += f"{stats['room_tag_rate']*100:.1f}%".ljust(15)
                row += f"{stats['both_tags_rate']*100:.1f}%".ljust(15)
                row += f"{stats['valid_action_rate']*100:.1f}%".ljust(15)
                row += f"{stats['avg_output_tokens']:.1f}".ljust(15)
                f.write(row + "\n")
        
        # Save all completions data as CSV
        csv_path = os.path.join(analysis_dir, f"beam_completions_{timestamp}.csv")
        
        try:
            import pandas as pd
            
            # Convert completions to DataFrame
            df = pd.DataFrame(completions)
            
            # Select relevant columns
            columns = [
                "game", "step", "beam_idx", "has_command_tags", "has_room_tags", 
                "command", "room", "action", "action_valid", "input_tokens", "output_tokens"
            ]
            
            # Save to CSV
            df[columns].to_csv(csv_path, index=False)
            
        except ImportError:
            # Fallback if pandas is not available
            with open(csv_path, "w") as f:
                # Write header
                header = "game,step,beam_idx,has_command_tags,has_room_tags,command,room,action,action_valid,input_tokens,output_tokens"
                f.write(header + "\n")
                
                # Write data
                for c in completions:
                    row = f"{c['game']},{c['step']},{c['beam_idx']},{c['has_command_tags']},{c['has_room_tags']},"
                    row += f"\"{c['command']}\",\"{c['room']}\",\"{c['action']}\",{c['action_valid']},{c['input_tokens']},{c['output_tokens']}"
                    f.write(row + "\n")
        
        return summary_path

    def analyze_temperature_effects(
        self, 
        difficulty: int = 5,
        temperatures: List[float] = [0.1, 0.5, 0.7, 1.0, 1.5],
        games_per_temp: int = 1,
        max_steps: int = 20,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Analyze the effects of different temperature settings
        
        Args:
            difficulty: Difficulty level
            temperatures: List of temperature values to test
            games_per_temp: Number of games per temperature
            max_steps: Maximum steps per game
            save_results: Whether to save results
            
        Returns:
            Dictionary with temperature analysis
        """
        # Create timestamp for this analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Log analysis parameters
        self.log(f"\n{'='*50}")
        self.log(f"Starting temperature analysis")
        self.log(f"Difficulty: {difficulty}")
        self.log(f"Temperatures: {temperatures}")
        self.log(f"Games per temperature: {games_per_temp}")
        self.log(f"{'='*50}\n")
        
        # Store results for each temperature
        temp_results = {}
        
        # Save original temperature
        original_temp = self.eval_config.temperature
        
        # Test each temperature
        for temp in temperatures:
            self.log(f"\n{'='*50}")
            self.log(f"Testing temperature: {temp}")
            self.log(f"{'='*50}\n")
            
            # Set temperature
            self.eval_config.temperature = temp
            
            # Reset statistics
            self.reset_statistics()
            
            # Play games at this temperature
            all_game_records = []
            
            for game_num in range(games_per_temp):
                self.log(f"\nPlaying game {game_num+1}/{games_per_temp} at temperature {temp}")
                
                try:
                    # Play a game
                    game_record = self.play_game(
                        difficulty=difficulty,
                        max_steps=max_steps,
                        log=True,
                        verbose=False
                    )
                    
                    # Add to all game records
                    all_game_records.append(game_record)
                    
                except Exception as e:
                    self.log(f"Error playing game at temperature {temp}: {str(e)}")
                    # Create a failed game record
                    failed_game = {
                        "difficulty": difficulty,
                        "trajectory": [],
                        "metrics": {
                            "success": False,
                            "steps": 0,
                            "score": 0,
                            "format_check_passed": 0,
                            "room_prediction_correct": 0,
                            "error": str(e)
                        }
                    }
                    all_game_records.append(failed_game)
            
            # Compile results for this temperature
            results = self.compile_evaluation_results(all_game_records, 0)
            temp_results[temp] = results
        
        # Restore original temperature
        self.eval_config.temperature = original_temp
        
        # Compile temperature analysis
        analysis = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "difficulty": difficulty,
            "temperatures": temperatures,
            "games_per_temp": games_per_temp,
            "results": temp_results
        }
        
        # Print temperature analysis summary
        self._print_temperature_analysis(analysis)
        
        # Save analysis if requested
        if save_results:
            analysis_path = self._save_temperature_analysis(analysis, timestamp)
            self.log(f"\nTemperature analysis saved to {analysis_path}")
        
        return analysis

    def _print_temperature_analysis(self, analysis: Dict[str, Any]):
        """Print temperature analysis summary
        
        Args:
            analysis: Dictionary with temperature analysis
        """
        temperatures = analysis["temperatures"]
        results = analysis["results"]
        
        self.log("\n\n" + "="*50)
        self.log("Temperature Analysis Summary")
        self.log("="*50)
        
        # Create comparison table for key metrics
        metrics = [
            "success_rate", 
            "avg_steps", 
            "format_correct_rate", 
            "room_prediction_accuracy",
            "valid_action_rate",
            "avg_beam_diversity"
        ]
        
        metric_names = {
            "success_rate": "Success Rate (%)",
            "avg_steps": "Avg Steps",
            "format_correct_rate": "Format Correct (%)",
            "room_prediction_accuracy": "Room Prediction (%)",
            "valid_action_rate": "Valid Actions (%)",
            "avg_beam_diversity": "Beam Diversity"
        }
        
        # Print header
        header = "Metric".ljust(25)
        for temp in temperatures:
            header += f"T={temp}".ljust(15)
        self.log("\n" + header)
        self.log("-" * (25 + 15 * len(temperatures)))
        
        # Print each metric
        for metric in metrics:
            row = metric_names[metric].ljust(25)
            for temp in temperatures:
                if temp in results and "overall" in results[temp]:
                    if metric in ["success_rate", "format_correct_rate", "room_prediction_accuracy", "valid_action_rate"]:
                        value = f"{results[temp]['overall'][metric]*100:.1f}%"
                    elif metric == "avg_beam_diversity":
                        value = f"{results[temp]['overall'][metric]:.3f}"
                    else:
                        value = f"{results[temp]['overall'][metric]:.1f}"
                else:
                    value = "N/A"
                row += value.ljust(15)
            self.log(row)

    def _save_temperature_analysis(self, analysis: Dict[str, Any], timestamp: str) -> str:
        """Save temperature analysis to file
        
        Args:
            analysis: Dictionary with temperature analysis
            timestamp: Timestamp string
            
        Returns:
            Path to saved analysis file
        """
        # Create analysis directory
        analysis_dir = os.path.join(self.log_dir, "temperature_analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Save detailed JSON analysis
        json_path = os.path.join(analysis_dir, f"temperature_analysis_{timestamp}.json")
        
        with open(json_path, "w") as f:
            json.dump(analysis, f, indent=2)
        
        # Save human-readable summary
        summary_path = os.path.join(analysis_dir, f"temperature_analysis_summary_{timestamp}.txt")
        
        with open(summary_path, "w") as f:
            temperatures = analysis["temperatures"]
            results = analysis["results"]
            
            f.write("Temperature Analysis\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Difficulty: {analysis['difficulty']}\n")
            f.write(f"Temperatures tested: {temperatures}\n")
            f.write(f"Games per temperature: {analysis['games_per_temp']}\n\n")
            
            # Create comparison table for key metrics
            metrics = [
                "success_rate", 
                "avg_steps", 
                "format_correct_rate", 
                "room_prediction_accuracy",
                "valid_action_rate",
                "avg_beam_diversity"
            ]
            
            metric_names = {
                "success_rate": "Success Rate (%)",
                "avg_steps": "Avg Steps",
                "format_correct_rate": "Format Correct (%)",
                "room_prediction_accuracy": "Room Prediction (%)",
                "valid_action_rate": "Valid Actions (%)",
                "avg_beam_diversity": "Beam Diversity"
            }
            
            # Write header
            header = "Metric".ljust(25)
            for temp in temperatures:
                header += f"T={temp}".ljust(15)
            f.write("\n" + header + "\n")
            f.write("-" * (25 + 15 * len(temperatures)) + "\n")
            
            # Write each metric
            for metric in metrics:
                row = metric_names[metric].ljust(25)
                for temp in temperatures:
                    if temp in results and "overall" in results[temp]:
                        if metric in ["success_rate", "format_correct_rate", "room_prediction_accuracy", "valid_action_rate"]:
                            value = f"{results[temp]['overall'][metric]*100:.1f}%"
                        elif metric == "avg_beam_diversity":
                            value = f"{results[temp]['overall'][metric]:.3f}"
                        else:
                            value = f"{results[temp]['overall'][metric]:.1f}"
                    else:
                        value = "N/A"
                    row += value.ljust(15)
                f.write(row + "\n")
        
        # Generate CSV for easy import into spreadsheets
        csv_path = os.path.join(analysis_dir, f"temperature_analysis_{timestamp}.csv")
        
        try:
            import pandas as pd
            
            # Prepare data for CSV
            data = []
            
            for temp in temperatures:
                if temp in results and "overall" in results[temp]:
                    row_data = {"temperature": temp}
                    
                    for metric in metrics:
                        if metric in results[temp]["overall"]:
                            row_data[metric] = results[temp]["overall"][metric]
            data.append(row_data)
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)
            
        except ImportError:
            # Fallback if pandas is not available
            with open(csv_path, "w") as f:
                # Write header
                header = "temperature," + ",".join(metrics)
                f.write(header + "\n")
                
                # Write data for each temperature
                for temp in temperatures:
                    if temp in results and "overall" in results[temp]:
                        row = f"{temp}"
                        
                        for metric in metrics:
                            if metric in results[temp]["overall"]:
                                if metric in ["success_rate", "format_correct_rate", "room_prediction_accuracy", "valid_action_rate"]:
                                    value = results[temp]["overall"][metric] * 100  # Convert to percentage
                                else:
                                    value = results[temp]["overall"][metric]
                            row += f",{value}"
                        else:
                            row += ",N/A"
                    
                    f.write(row + "\n")
        
        return summary_path

    def generate_report(self, results_dir: str = None) -> str:
        """Generate a comprehensive report from all evaluation results
        
        Args:
            results_dir: Directory containing evaluation results (default: self.log_dir/results)
            
        Returns:
            Path to the generated report
        """
        # Use default results directory if not specified
        if results_dir is None:
            results_dir = os.path.join(self.log_dir, "results")
        
        # Check if directory exists
        if not os.path.exists(results_dir):
            self.log(f"Results directory not found: {results_dir}")
            return None
        
        # Create timestamp for this report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create report directory
        report_dir = os.path.join(self.log_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)
        
        # Report path
        report_path = os.path.join(report_dir, f"evaluation_report_{timestamp}.md")
        
        # Find all JSON result files
        json_files = [f for f in os.listdir(results_dir) if f.endswith(".json") and "results" in f]
        
        if not json_files:
            self.log(f"No result files found in {results_dir}")
            return None
        
        # Load all results
        all_results = []
        
        for json_file in json_files:
            file_path = os.path.join(results_dir, json_file)
            try:
                with open(file_path, "r") as f:
                    results = json.load(f)
                    results["file"] = json_file
                    all_results.append(results)
            except Exception as e:
                self.log(f"Error loading {json_file}: {str(e)}")
        
        # Sort results by timestamp
        all_results.sort(key=lambda r: r.get("timestamp", ""))
        
        # Generate report
        with open(report_path, "w") as f:
            f.write("# TextWorld LLM Agent Evaluation Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary of All Evaluations\n\n")
            
            # Create summary table
            f.write("| Date | Games | Success Rate | Format Correct | Room Prediction |\n")
            f.write("|------|-------|--------------|----------------|----------------|\n")
            
            for result in all_results:
                if "overall" in result:
                    date = result.get("timestamp", "Unknown")
                    games = result["overall"]["games_played"]
                    success_rate = f"{result['overall']['success_rate']*100:.1f}%"
                    format_correct = f"{result['overall']['format_correct_rate']*100:.1f}%"
                    room_prediction = f"{result['overall']['room_prediction_accuracy']*100:.1f}%"
                    
                    f.write(f"| {date} | {games} | {success_rate} | {format_correct} | {room_prediction} |\n")
            
            # Detailed results for each evaluation
            f.write("\n## Detailed Results\n\n")
            
            for i, result in enumerate(all_results):
                if "overall" in result:
                    f.write(f"### Evaluation {i+1}: {result.get('timestamp', 'Unknown')}\n\n")
                    
                    f.write("#### Overall Statistics\n\n")
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    
                    metrics = [
                        ("Games Played", result["overall"]["games_played"]),
                        ("Success Rate", f"{result['overall']['success_rate']*100:.1f}%"),
                        ("Average Steps", f"{result['overall']['avg_steps']:.1f}"),
                        ("Format Correctness", f"{result['overall']['format_correct_rate']*100:.1f}%"),
                        ("Command Tag Usage", f"{result['overall']['command_tag_rate']*100:.1f}%"),
                        ("Room Tag Usage", f"{result['overall']['room_tag_rate']*100:.1f}%"),
                        ("Room Prediction Accuracy", f"{result['overall']['room_prediction_accuracy']*100:.1f}%"),
                        ("Valid Action Rate", f"{result['overall']['valid_action_rate']*100:.1f}%")
                    ]
                    
                    for metric, value in metrics:
                        f.write(f"| {metric} | {value} |\n")
                    
                    # Per difficulty results
                    if "per_difficulty" in result:
                        f.write("\n#### Results by Difficulty\n\n")
                        f.write("| Difficulty | Games | Success Rate | Avg Steps | Format Correct |\n")
                        f.write("|------------|-------|--------------|-----------|----------------|\n")
                        
                        for difficulty, stats in sorted(result["per_difficulty"].items()):
                            diff = difficulty
                            games = stats["games_played"]
                            success_rate = f"{stats['success_rate']*100:.1f}%"
                            avg_steps = f"{stats['avg_steps']:.1f}"
                            format_correct = f"{stats['format_correct_rate']*100:.1f}%"
                            
                            f.write(f"| {diff} | {games} | {success_rate} | {avg_steps} | {format_correct} |\n")
                    
                    f.write("\n")
        
        self.log(f"Report generated: {report_path}")
        return report_path
