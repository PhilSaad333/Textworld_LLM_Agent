"""
Simple script to pull from GitHub and reload the modules.
Run this in Colab to get the latest code without restarting the session.
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

def pull_and_reload():
    """Pull from GitHub and reload the modules"""
    # Get the repository directory
    repo_dir = "/content/Textworld_LLM_Agent"
    
    # Change to the repository directory
    os.chdir(repo_dir)
    
    # Pull the latest changes
    run_command("git pull origin main")
    
    # Reload the modules
    modules_to_reload = [
        "training.trainer",
        "training.optimizer",
        "config.rl_config",
        "config.config"
    ]
    
    print("Reloading modules...")
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
            print(f"Reloaded {module_name}")
        else:
            print(f"Module {module_name} not loaded yet")
    
    print("\nPull and reload completed!")

if __name__ == "__main__":
    pull_and_reload() 