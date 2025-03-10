"""
Script to update the test_grpo_fixed.py file in the Colab environment.
"""

import os

# Create the directories if they don't exist
os.makedirs("/content/Textworld_LLM_Agent", exist_ok=True)

# Update test_grpo_fixed.py
print("Updating test_grpo_fixed.py...")

# Create the updated test script
test_script = """
# Paste the entire content of the updated test_grpo_fixed.py file here
"""

with open("/content/Textworld_LLM_Agent/test_grpo_fixed.py", "w") as f:
    f.write(test_script)

print("Updated test_grpo_fixed.py")
print("\nDone! You can now run the test_grpo_fixed.py script.")
print("Run: %run /content/Textworld_LLM_Agent/test_grpo_fixed.py") 