"""
Script to update the files in the Colab environment with the simplified GRPO implementation.
"""

import os
import sys
import importlib
import subprocess

def run_command(cmd):
    """Run a shell command and print the output"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}")
    return result

def reload_modules():
    """Reload the modules"""
    modules_to_reload = [
        "training.trainer",
        "training.optimizer"
    ]
    
    print("Reloading modules...")
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
            print(f"Reloaded {module_name}")
        else:
            print(f"Module {module_name} not loaded yet")

def update_files():
    """Update the files in the Colab environment"""
    # Create the directories if they don't exist
    os.makedirs("/content/Textworld_LLM_Agent/training", exist_ok=True)
    
    # Update optimizer.py
    optimizer_content = """
# Paste the entire content of the updated optimizer.py file here
"""
    
    with open("/content/Textworld_LLM_Agent/training/optimizer.py", "w") as f:
        f.write(optimizer_content)
    
    # Update trainer.py
    trainer_content = """
# Paste the entire content of the updated trainer.py file here, or just the _train_with_custom_grpo method
"""
    
    with open("/content/Textworld_LLM_Agent/training/trainer.py", "r") as f:
        trainer_code = f.read()
    
    # Replace the _train_with_custom_grpo method
    # This is a simple approach - for a more robust solution, you might want to use a proper code parser
    start_marker = "def _train_with_custom_grpo(self, save_model_path=None):"
    end_marker = "def _convert_data_for_huggingface(self):"
    
    start_idx = trainer_code.find(start_marker)
    end_idx = trainer_code.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        # Extract the code before and after the method
        before_code = trainer_code[:start_idx]
        after_code = trainer_code[end_idx:]
        
        # Replace the method with the updated version
        updated_method = """def _train_with_custom_grpo(self, save_model_path=None):
        \"\"\"
        Train the agent using our custom GRPO implementation with pre-collected data
        
        Args:
            save_model_path: Path to save the trained model
            
        Returns:
            Dictionary with training metrics
        \"\"\"
        print("Training with custom GRPO implementation...")
        
        # Convert gameplay data to trajectories format expected by our GRPO optimizer
        trajectories = self._convert_data_for_custom_grpo()
        
        if not trajectories:
            raise ValueError("No valid trajectories found. Please check your gameplay data.")
        
        # Print training parameters
        print(f"Training with pre-collected data:")
        print(f"  trajectories: {len(trajectories)} episodes with {sum(len(t['steps']) for t in trajectories)} total steps")
        print(f"  save_path: {save_model_path}")
        
        # Train using our custom GRPO optimizer with pre-collected trajectories
        metrics = self.grpo_optimizer.train(
            agent=self.agent,
            trajectories=trajectories,  # Pass pre-collected trajectories
            save_path=save_model_path
        )
        
        return metrics
        
    """
        
        # Combine the code
        updated_code = before_code + updated_method + after_code
        
        # Write the updated code back to the file
        with open("/content/Textworld_LLM_Agent/training/trainer.py", "w") as f:
            f.write(updated_code)
    
    # Create test_grpo_fixed.py
    test_content = """
# Paste the entire content of the updated test_grpo_fixed.py file here
"""
    
    with open("/content/Textworld_LLM_Agent/test_grpo_fixed.py", "w") as f:
        f.write(test_content)
    
    print("Files updated successfully!")

# Run the update
update_files()

# Reload the modules
reload_modules()

print("\nDone! You can now run the test_grpo_fixed.py script.")
print("Run: %run /content/Textworld_LLM_Agent/test_grpo_fixed.py") 