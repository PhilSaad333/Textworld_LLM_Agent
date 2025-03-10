"""
Script to update the optimizer.py file in the Colab environment.
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

# Update the _compute_logprobs method in optimizer.py
print("Updating the _compute_logprobs method in optimizer.py...")

# Read the current optimizer.py file
with open("/content/Textworld_LLM_Agent/training/optimizer.py", "r") as f:
    optimizer_code = f.read()

# Find the _compute_logprobs method
start_marker = "def _compute_logprobs(self, model, inputs, output_tokens, batch_idx=0):"
end_marker = "def"

start_idx = optimizer_code.find(start_marker)
if start_idx == -1:
    print("Could not find the _compute_logprobs method in optimizer.py")
    sys.exit(1)

# Find the end of the method (the next def)
end_idx = optimizer_code.find(end_marker, start_idx + len(start_marker))
if end_idx == -1:
    # If there's no next def, use the end of the file
    end_idx = len(optimizer_code)
else:
    # Adjust to include the indentation before the next def
    end_idx = optimizer_code.rfind("\n", 0, end_idx) + 1

# Extract the code before and after the method
before_code = optimizer_code[:start_idx]
after_code = optimizer_code[end_idx:]

# Replace the method with the updated version
updated_method = """def _compute_logprobs(self, model, inputs, output_tokens, batch_idx=0):
        \"\"\"
        Compute log probabilities for a given output sequence
        
        Args:
            model: The model to use for computing log probabilities
            inputs: Tokenized input sequence
            output_tokens: Tokenized output sequence
            batch_idx: Index in the batch
            
        Returns:
            Log probabilities for the output sequence
        \"\"\"
        # Get input for this example
        input_ids = inputs.input_ids[batch_idx:batch_idx+1]
        attention_mask = inputs.attention_mask[batch_idx:batch_idx+1] if hasattr(inputs, 'attention_mask') else None
        
        # Get output for this example
        output_ids = output_tokens.input_ids[0]
        
        # Check if we're using an encoder-decoder model (like T5) or a decoder-only model (like GPT)
        is_encoder_decoder = getattr(model.config, "is_encoder_decoder", False)
        
        # Prepare model inputs
        model_inputs = {
            "input_ids": input_ids,
        }
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        
        # For encoder-decoder models, we need to provide decoder inputs
        if is_encoder_decoder:
            # For computing log probs, we need to shift the output_ids to create decoder_input_ids
            # The first token of decoder_input_ids is the decoder_start_token_id
            # The rest are the output_ids except the last one
            decoder_start_token_id = getattr(model.config, "decoder_start_token_id", None)
            if decoder_start_token_id is None:
                decoder_start_token_id = getattr(model.config, "pad_token_id", 0)
            
            # Create decoder_input_ids by shifting output_ids right and prepending decoder_start_token_id
            decoder_input_ids = torch.cat([
                torch.tensor([[decoder_start_token_id]], device=self.device),
                output_ids[:-1].unsqueeze(0)
            ], dim=1)
            
            model_inputs["decoder_input_ids"] = decoder_input_ids
        
        # Forward pass through the model
        with torch.no_grad():
            outputs = model(**model_inputs)
            
        # Get logits
        logits = outputs.logits
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Extract log probabilities for the actual output tokens
        token_log_probs = []
        
        if is_encoder_decoder:
            # For encoder-decoder models, we compare each position in the output with the next token
            for i in range(len(output_ids) - 1):  # -1 because we don't need the last token's prediction
                if output_ids[i+1] == model.config.pad_token_id:
                    continue  # Skip pad tokens
                token_log_prob = log_probs[0, i, output_ids[i+1]]
                token_log_probs.append(token_log_prob)
        else:
            # For decoder-only models, we need to handle differently
            # This part remains the same as before
            for i in range(len(output_ids) - 1):  # -1 because we don't need the last token's prediction
                if output_ids[i+1] == model.config.pad_token_id:
                    continue  # Skip pad tokens
                token_log_prob = log_probs[0, i, output_ids[i+1]]
                token_log_probs.append(token_log_prob)
        
        # Combine token log probabilities
        if token_log_probs:
            return torch.stack(token_log_probs).mean().unsqueeze(0)
        else:
            return torch.tensor([0.0], device=self.device)
    
    """

# Combine the code
updated_code = before_code + updated_method + after_code

# Write the updated code back to the file
with open("/content/Textworld_LLM_Agent/training/optimizer.py", "w") as f:
    f.write(updated_code)

print("Updated the _compute_logprobs method in optimizer.py")

# Reload the modules
reload_modules()

print("\nDone! You can now run the test_grpo_fixed.py script.")
print("Run: %run /content/Textworld_LLM_Agent/test_grpo_fixed.py") 