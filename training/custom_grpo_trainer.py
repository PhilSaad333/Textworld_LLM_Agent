from trl import GRPOTrainer
import torch

class CustomGRPOTrainer(GRPOTrainer):
    def _prepare_inputs(self, inputs):
        """
        Modified version of _prepare_inputs that handles problematic sequences
        without changing the original data
        """
        try:
            # Get the device
            device = self.model.device if hasattr(self.model, "device") else self.args.device
            
            # Check if inputs is a dict and has completion_ids
            if not isinstance(inputs, dict) or "completion_ids" not in inputs:
                # If not, let the parent class handle it
                return super()._prepare_inputs(inputs)
            
            # Get the completion IDs
            completion_ids = inputs.pop("completion_ids").to(device)
            
            # Print debug info
            print(f"DEBUG - completion_ids shape: {completion_ids.shape}")
            print(f"DEBUG - completion_ids device: {completion_ids.device}")
            
            # Handle empty sequences
            if completion_ids.size(1) == 0:
                # Add a dummy token (EOS) to each sequence
                completion_ids = torch.full(
                    (completion_ids.size(0), 1), 
                    self.processing_class.eos_token_id, 
                    dtype=torch.long, 
                    device=device
                )
                print("DEBUG - Added dummy EOS token to empty sequence")
            
            # Find EOS tokens
            is_eos = completion_ids == self.processing_class.eos_token_id
            
            # Print debug info
            print(f"DEBUG - is_eos shape: {is_eos.shape}")
            print(f"DEBUG - is_eos.any(dim=1).sum(): {is_eos.any(dim=1).sum()}")
            
            # Handle case where no EOS tokens are found
            if is_eos.any(dim=1).sum() == 0:
                # Add EOS token to the end of each sequence
                eos_tensor = torch.full((completion_ids.size(0), 1), self.processing_class.eos_token_id, dtype=torch.long, device=device)
                completion_ids = torch.cat([completion_ids, eos_tensor], dim=1)
                # Update is_eos
                is_eos = completion_ids == self.processing_class.eos_token_id
                print("DEBUG - Added EOS token to sequences without one")
            
            # Create eos_idx tensor
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=device)
            
            # Find position of first EOS token in each sequence
            # Check if any dimension is 0 before calling argmax
            if is_eos.size(1) > 0 and is_eos.any(dim=1).sum() > 0:
                eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            else:
                print("WARNING - is_eos has a dimension of size 0, cannot call argmax")
                # Set all indices to the last position as a fallback
                eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=device)
            
            # Create completion mask
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            
            # Add back to inputs
            inputs["completion_ids"] = completion_ids
            inputs["completion_mask"] = completion_mask
            
            # Move all inputs to the device
            for k, v in inputs.items():
                if hasattr(v, "to") and callable(v.to):
                    inputs[k] = v.to(device)
            
            return inputs
        except Exception as e:
            print(f"Error in _prepare_inputs: {e}")
            print(f"inputs keys: {list(inputs.keys()) if isinstance(inputs, dict) else 'Not a dict'}")
            if 'completion_ids' in locals():
                print(f"completion_ids shape: {completion_ids.shape}")
                print(f"completion_ids device: {completion_ids.device}")
                if completion_ids.numel() > 0:
                    print(f"completion_ids sample: {completion_ids[0][:10]}")
            if 'is_eos' in locals():
                print(f"is_eos shape: {is_eos.shape}")
                print(f"is_eos.any(dim=1).sum(): {is_eos.any(dim=1).sum()}")
            raise
