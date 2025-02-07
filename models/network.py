import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F

class TextWorldBaseNetwork(nn.Module):
    """Shared base network using pretrained transformer"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.model_name)
        
    def forward(self, x):
        """
        Args:
            x: Dictionary of input tensors from tokenizer
        """
        # Create attention mask
        attention_mask = (x['input_ids'] != self.encoder.config.pad_token_id).float()
        
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=x['input_ids'],
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get sequence output
        sequence_output = outputs.last_hidden_state
        
        return sequence_output, attention_mask

class TextWorldPolicyNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base = TextWorldBaseNetwork(config)
        
        # Action scoring components
        self.action_encoder = nn.Linear(768, config.cmd_summary_dim)  # RoBERTa dim -> cmd dim
        self.query_proj = nn.Linear(768, config.cmd_summary_dim)  # RoBERTa dim -> cmd dim
        
        # Multi-head attention for action scoring
        self.action_attention = nn.MultiheadAttention(
            embed_dim=config.cmd_summary_dim,
            num_heads=config.num_cs_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Final scoring layers
        self.score_mlp = nn.Sequential(
            nn.Linear(config.cmd_summary_dim, config.cmd_summary_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.cmd_summary_dim, 1)
        )
        
    def forward(self, obs_tokens, action_tokens):
        """
        Args:
            obs_tokens: Dictionary of observation tensors from tokenizer
            action_tokens: Dictionary of action tensors from tokenizer
        Returns:
            torch.Tensor: Action logits [batch_size, num_actions]
        """
        batch_size = obs_tokens['input_ids'].size(0)
        num_actions = action_tokens['input_ids'].size(0)
        
        # Encode observation
        obs_features, obs_mask = self.base(obs_tokens)  # [batch_size, obs_seq_len, 768]
        
        # Mean pool observation features
        obs_features = obs_features.mean(dim=1)  # [batch_size, 768]
        
        # Process each action separately
        action_scores = []
        for i in range(num_actions):
            # Get current action tokens
            curr_action = {
                k: v[i:i+1] for k, v in action_tokens.items()
            }
            
            # Encode action
            action_features, _ = self.base(curr_action)  # [batch_size, action_seq_len, 768]
            action_features = action_features.mean(dim=1)  # [batch_size, 768]
            
            # Project features to command dimension
            query = self.query_proj(obs_features).unsqueeze(1)  # [batch_size, 1, cmd_dim]
            key = self.action_encoder(action_features).unsqueeze(1)  # [batch_size, 1, cmd_dim]
            value = key
            
            # Score action using attention
            attn_output, _ = self.action_attention(query, key, value)  # [batch_size, 1, cmd_dim]
            score = self.score_mlp(attn_output).squeeze(-1)  # [batch_size, 1]
            action_scores.append(score)
        
        # Combine all action scores
        logits = torch.cat(action_scores, dim=1)  # [batch_size, num_actions]
        return logits

class TextWorldValueNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base = TextWorldBaseNetwork(config)
        
        # Action encoding
        self.action_encoder = nn.Linear(768, config.cmd_summary_dim)
        
        # Q-value prediction layers
        self.q_head = nn.Sequential(
            nn.Linear(768 + config.cmd_summary_dim, config.cmd_summary_dim),  # Combined features
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.cmd_summary_dim, 1)
        )
        
    def forward(self, obs_tokens, action_tokens=None):
        """
        Args:
            obs_tokens: Tokenized observation [batch_size, seq_len]
            action_tokens: Optional tokenized action [batch_size, action_seq_len]
                         If None, only encodes the state
        Returns:
            torch.Tensor: Q-value prediction [batch_size, 1]
        """
        # Encode observation
        obs_features, obs_mask = self.base(obs_tokens)
        obs_features = obs_features.mean(dim=1)  # [batch_size, dim]
        
        if action_tokens is None:
            raise ValueError("Action tokens required for Q-value prediction")
            
        # Encode action
        action_features, _ = self.base(action_tokens)
        action_features = action_features.mean(dim=1)  # [batch_size, dim]
        
        # Combine state and action features
        combined_features = torch.cat([
            obs_features,
            self.action_encoder(action_features)
        ], dim=-1)
        
        # Predict Q-value
        q_value = self.q_head(combined_features)
        
        return q_value

class RoomPredictionNetwork(nn.Module):
    def __init__(self, config, text_processor):
        super().__init__()
        self.config = config
        self.base = TextWorldBaseNetwork(config)
        self.text_processor = text_processor
        
        # Room embedding layer
        self.room_embedding = nn.Linear(768, config.room_dim)
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(768 * 2, config.room_dim),  # State + Action -> Room
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.room_dim, config.room_dim)
        )
        
    def forward(self, obs_tokens, action_tokens, known_rooms):
        """
        Predict next room given current state and action
        """
        # Encode current state and action
        state_features, _ = self.base(obs_tokens)
        state_features = state_features.mean(dim=1)  # [batch_size, 768]
        
        action_features, _ = self.base(action_tokens)
        action_features = action_features.mean(dim=1)  # [batch_size, 768]
        
        # Combine features
        combined = torch.cat([state_features, action_features], dim=-1)  # [batch_size, 768*2]
        
        # Get room prediction features
        room_features = self.prediction_head(combined)  # [batch_size, room_dim]
        
        # Compare with known room embeddings
        known_room_logits = []
        for room in known_rooms:
            room_tokens = self.text_processor.process_text(room)
            room_features_raw, _ = self.base(room_tokens)
            room_embedding = self.room_embedding(room_features_raw.mean(dim=1))  # [1, room_dim]
            similarity = torch.matmul(room_features, room_embedding.t())
            known_room_logits.append(similarity)
            
        # Add logit for "unseen" prediction
        unseen_embedding = self.room_embedding(
            torch.zeros(1, 768, device=room_features.device)
        )  # [1, room_dim]
        unseen_logit = torch.matmul(room_features, unseen_embedding.t())
        
        # Combine all logits
        if known_room_logits:
            all_logits = torch.cat([*known_room_logits, unseen_logit], dim=-1)
        else:
            all_logits = unseen_logit
        
        return all_logits
