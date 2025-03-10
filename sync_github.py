"""
Script to sync with GitHub and reload the files without restarting the Colab session.
"""

import os
import sys
import importlib
import subprocess
from IPython.display import clear_output

def run_command(cmd, verbose=True):
    """Run a shell command and print the output"""
    if verbose:
        print(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if verbose:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}")
    
    return result

def sync_with_github(repo_dir, branch="main", commit_message=None, push=False):
    """
    Sync local changes with GitHub
    
    Args:
        repo_dir: Path to the repository directory
        branch: Branch to sync with
        commit_message: Commit message (if None, will prompt for one)
        push: Whether to push changes to GitHub
    """
    # Change to the repository directory
    os.chdir(repo_dir)
    
    # Check if there are any changes
    status = run_command("git status -s", verbose=False)
    if not status.stdout.strip():
        print("No changes to commit.")
        return
    
    # Show the changes
    print("Changes to commit:")
    run_command("git status -s")
    
    # Add all changes
    run_command("git add .")
    
    # Commit the changes
    if commit_message is None:
        commit_message = input("Enter commit message: ")
    
    run_command(f'git commit -m "{commit_message}"')
    
    # Push the changes if requested
    if push:
        run_command(f"git push origin {branch}")
        print(f"Changes pushed to {branch} branch.")
    else:
        print("Changes committed locally. Use 'git push origin <branch>' to push to GitHub.")

def reload_modules(module_names):
    """
    Reload specified modules
    
    Args:
        module_names: List of module names to reload
    """
    print("Reloading modules...")
    for module_name in module_names:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
            print(f"Reloaded {module_name}")
        else:
            print(f"Module {module_name} not loaded yet")

def pull_from_github(repo_dir, branch="main"):
    """
    Pull latest changes from GitHub
    
    Args:
        repo_dir: Path to the repository directory
        branch: Branch to pull from
    """
    # Change to the repository directory
    os.chdir(repo_dir)
    
    # Pull the latest changes
    run_command(f"git pull origin {branch}")
    print(f"Pulled latest changes from {branch} branch.")

def main():
    # Get the repository directory
    repo_dir = os.getcwd()
    if not os.path.exists(os.path.join(repo_dir, ".git")):
        repo_dir = input("Enter the path to your repository: ")
    
    # Show menu
    print("\nGitHub Sync Menu:")
    print("1. Commit local changes")
    print("2. Push local changes to GitHub")
    print("3. Pull latest changes from GitHub")
    print("4. Reload modules")
    print("5. Full sync (commit, push, pull, reload)")
    print("6. Exit")
    
    choice = input("\nEnter your choice (1-6): ")
    
    if choice == "1":
        commit_message = input("Enter commit message: ")
        sync_with_github(repo_dir, commit_message=commit_message, push=False)
    
    elif choice == "2":
        run_command(f"cd {repo_dir} && git push origin main")
    
    elif choice == "3":
        pull_from_github(repo_dir)
    
    elif choice == "4":
        modules_to_reload = [
            "training.trainer",
            "training.optimizer",
            "config.rl_config",
            "config.config"
        ]
        reload_modules(modules_to_reload)
    
    elif choice == "5":
        commit_message = input("Enter commit message: ")
        sync_with_github(repo_dir, commit_message=commit_message, push=True)
        pull_from_github(repo_dir)
        modules_to_reload = [
            "training.trainer",
            "training.optimizer",
            "config.rl_config",
            "config.config"
        ]
        reload_modules(modules_to_reload)
    
    elif choice == "6":
        print("Exiting...")
        return
    
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 