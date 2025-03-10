"""
Simple script to sync with GitHub and reload the modules.
Run this in Colab after making changes to the code.
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

def sync_and_reload():
    """Sync with GitHub and reload the modules"""
    # Get the repository directory
    repo_dir = "/content/Textworld_LLM_Agent"
    
    # Change to the repository directory
    os.chdir(repo_dir)
    
    # Configure Git (if needed)
    username = input("Enter your GitHub username: ")
    email = input("Enter your GitHub email: ")
    run_command(f'git config --global user.name "{username}"')
    run_command(f'git config --global user.email "{email}"')
    
    # Add all changes
    run_command("git add .")
    
    # Commit the changes
    commit_message = input("Enter commit message: ")
    run_command(f'git commit -m "{commit_message}"')
    
    # Push the changes
    run_command("git push origin main")
    
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
    
    print("\nSync and reload completed!")

if __name__ == "__main__":
    sync_and_reload() 