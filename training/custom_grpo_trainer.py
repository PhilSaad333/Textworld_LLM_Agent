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
            
            # Get the completion IDs
            completion_ids = inputs.pop("completion_ids").to(device)
            
            # Handle empty sequences
            if completion_ids.size(1) == 0:
                # Add a dummy token (EOS) to each sequence
                completion_ids = torch.full(
                    (completion_ids.size(0), 1), 
                    self.processing_class.eos_token_id, 
                    dtype=torch.long, 
                    device=device
                )
            
            # Find EOS tokens
            is_eos = completion_ids == self.processing_class.eos_token_id
            
            # Handle case where no EOS tokens are found
            if is_eos.any(dim=1).sum() == 0:
                # Add EOS token to the end of each sequence
                eos_tensor = torch.full((completion_ids.size(0), 1), self.processing_class.eos_token_id, dtype=torch.long, device=device)
                completion_ids = torch.cat([completion_ids, eos_tensor], dim=1)
                # Update is_eos
                is_eos = completion_ids == self.processing_class.eos_token_id
            
            # Create eos_idx tensor
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=device)
            
            # Find position of first EOS token in each sequence
            if is_eos.any(dim=1).sum() > 0:
                eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            
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
            # If there's an error, print detailed information and re-raise
            print(f"Error in _prepare_inputs: {e}")
            print(f"completion_ids shape: {completion_ids.shape if 'completion_ids' in locals() else 'Not available'}")
            if 'is_eos' in locals():
                print(f"is_eos shape: {is_eos.shape}")
                print(f"is_eos.any(dim=1).sum(): {is_eos.any(dim=1).sum()}")
            raise
