"""
Script to update the optimizer.py file with the gradient-related changes.
"""

import os
import sys
import importlib

def reload_modules():
    """Reload the modules"""
    modules_to_reload = [
        "training.optimizer"
    ]
    
    print("Reloading modules...")
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
            print(f"Reloaded {module_name}")
        else:
            print(f"Module {module_name} not loaded yet")

# Create the directories if they don't exist
os.makedirs("/content/Textworld_LLM_Agent/training", exist_ok=True)

# Update the optimizer.py file
print("Updating the optimizer.py file...")

# Read the current optimizer.py file
with open("/content/Textworld_LLM_Agent/training/optimizer.py", "r") as f:
    optimizer_code = f.read()

# Update the _compute_logprobs method
optimizer_code = optimizer_code.replace(
    "def _compute_logprobs(self, model, inputs, output_tokens, batch_idx=0):",
    "def _compute_logprobs(self, model, inputs, output_tokens, batch_idx=0, with_grad=False):"
)

# Update the with_grad parameter documentation
optimizer_code = optimizer_code.replace(
    "            batch_idx: Index in the batch",
    "            batch_idx: Index in the batch\n            with_grad: Whether to compute gradients (True for new policy, False for old policy)"
)

# Update the forward pass to use with_grad
optimizer_code = optimizer_code.replace(
    "        # Forward pass through the model\n        with torch.no_grad():\n            outputs = model(**model_inputs)",
    "        # Forward pass through the model\n        if with_grad:\n            # Compute with gradients for the new policy\n            outputs = model(**model_inputs)\n        else:\n            # Compute without gradients for the old policy\n            with torch.no_grad():\n                outputs = model(**model_inputs)"
)

# Update the old_logprobs call
optimizer_code = optimizer_code.replace(
    "old_logprobs = self._compute_logprobs(agent.model, inputs, output_tokens, i)",
    "old_logprobs = self._compute_logprobs(agent.model, inputs, output_tokens, i, with_grad=False)"
)

# Update the new_logprobs call
optimizer_code = optimizer_code.replace(
    "new_logprobs = self._compute_logprobs(agent.model, inputs, output_tokens, i)",
    "new_logprobs = self._compute_logprobs(agent.model, inputs, output_tokens, i, with_grad=True)"
)

# Update the optimizer initialization
optimizer_code = optimizer_code.replace(
    "optimizer = torch.optim.AdamW(agent.model.parameters(), lr=self.learning_rate)",
    "optimizer = torch.optim.AdamW(\n            [p for p in agent.model.parameters() if p.requires_grad],\n            lr=self.learning_rate\n        )"
)

# Update the model training mode comment
optimizer_code = optimizer_code.replace(
    "# Move model to device",
    "# Make sure model is in training mode"
)

# Write the updated code back to the file
with open("/content/Textworld_LLM_Agent/training/optimizer.py", "w") as f:
    f.write(optimizer_code)

print("Updated the optimizer.py file")

# Reload the modules
reload_modules()

print("\nDone! You can now run the test_grpo_fixed.py script.")
print("Run: %run /content/Textworld_LLM_Agent/test_grpo_fixed.py") 