"""
Simple script to pull from GitHub and reload the modules.
Run this in Colab to get the latest code without restarting the session.
"""

import os
import sys
import importlib
import subprocess
import glob

def run_command(cmd):
    """Run a shell command and print the output"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}")
    return result

def discover_modules(repo_dir):
    """Discover all Python modules in the repository"""
    modules = []
    
    # Walk through the repository and find all Python files
    for root, dirs, files in os.walk(repo_dir):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        # Process Python files
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                # Get the relative path from the repo directory
                rel_path = os.path.relpath(os.path.join(root, file), repo_dir)
                # Convert path to module name
                module_name = os.path.splitext(rel_path)[0].replace(os.path.sep, '.')
                modules.append(module_name)
    
    return modules

def pull_and_reload():
    """Pull from GitHub and reload the modules"""
    # Get the repository directory
    repo_dir = "/content/Textworld_LLM_Agent"
    
    # Change to the repository directory
    os.chdir(repo_dir)
    
    # Pull the latest changes
    run_command("git pull origin main")
    
    # Core modules that should always be reloaded
    core_modules = [
        "training.trainer",
        "training.optimizer",
        "training.rollout",
        "training.fine_tuning",
        "config.rl_config",
        "config.config",
        "agents.textworld_llm_agent",
        "agents.llm_game_runner",
        "environment.task_env",
        "environment.environment_manager_old"
    ]
    
    # Discover all modules in the repository
    all_modules = discover_modules(repo_dir)
    
    # Combine core modules with discovered modules, removing duplicates
    modules_to_reload = list(set(core_modules + all_modules))
    modules_to_reload.sort()  # Sort for readability
    
    print(f"\nReloading {len(modules_to_reload)} modules...")
    reloaded_count = 0
    skipped_count = 0
    
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            try:
                importlib.reload(sys.modules[module_name])
                print(f"✓ Reloaded {module_name}")
                reloaded_count += 1
            except Exception as e:
                print(f"✗ Error reloading {module_name}: {str(e)}")
        else:
            print(f"⚠ Module {module_name} not loaded yet")
            skipped_count += 1
    
    print(f"\nReload summary: {reloaded_count} reloaded, {skipped_count} skipped")
    print("\nPull and reload completed!")
    
    # Return the list of modules for reference
    return modules_to_reload

if __name__ == "__main__":
    pull_and_reload() 