# gpt_model/model/embedding.py

import torch
import torch.nn as nn

class GPTEmbedding(nn.Module):
    """
    Embedding layer for GPT-style models.
    Combines token embeddings and positional embeddings.
    """
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super().__init__()
        
        # Embedding for tokens: maps token IDs to dense vectors
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Embedding for positions: injects order information
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Save sequence length in case needed later
        self.max_seq_len = max_seq_len

    def forward(self, token_ids):
        """
        Forward pass for embedding.
        
        Args:
            token_ids: Tensor of shape (batch_size, seq_len)
        
        Returns:
            embeddings: Tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len = token_ids.shape

        # Token embeddings: (batch_size, seq_len, embed_dim)
        token_embeds = self.token_embedding(token_ids)

        # Position indices: (batch_size, seq_len)
        positions = torch.arange(seq_len, device=token_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        # Position embeddings
        pos_embeds = self.position_embedding(positions)

        # Final embedding = token + position
        return token_embeds + pos_embeds
