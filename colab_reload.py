"""
Simple script to pull the latest changes from GitHub and reload the modules in Colab.
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

# Copy the updated files to the Colab environment
print("Copying updated files to the Colab environment...")

# Create the optimizer.py file
optimizer_path = "/content/Textworld_LLM_Agent/training/optimizer.py"
print(f"Creating {optimizer_path}...")
with open(optimizer_path, "w") as f:
    f.write("""
# Paste the entire content of the updated optimizer.py file here
""")

# Create the trainer.py file
trainer_path = "/content/Textworld_LLM_Agent/training/trainer.py"
print(f"Creating {trainer_path}...")
with open(trainer_path, "w") as f:
    f.write("""
# Paste the entire content of the updated trainer.py file here
""")

# Create the test_grpo_fixed.py file
test_path = "/content/Textworld_LLM_Agent/test_grpo_fixed.py"
print(f"Creating {test_path}...")
with open(test_path, "w") as f:
    f.write("""
# Paste the entire content of the updated test_grpo_fixed.py file here
""")

# Reload the modules
reload_modules()

print("Done! You can now run the test_grpo_fixed.py script.")
print("Run: %run /content/Textworld_LLM_Agent/test_grpo_fixed.py") 