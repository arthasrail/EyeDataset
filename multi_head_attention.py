import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism implementation
    
    Args:
        d_model (int): Dimension of the model (embedding dimension)
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability (default: 0.1)
        
    Attributes:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        d_k (int): Dimension of each attention head (d_model // num_heads)
        
    Example:
        >>> attention = MultiHeadAttention(d_model=512, num_heads=8)
        >>> x = torch.randn(32, 100, 512)  # (batch_size, seq_len, d_model)
        >>> output, attention_weights = attention(x, x, x)
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear transformations for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention
        
        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len, d_k)
            K (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len, d_k)
            V (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len, d_k)
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            tuple: (attention_output, attention_weights)
        """
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention
        
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, query_len, d_model)
            key (torch.Tensor): Key tensor of shape (batch_size, key_len, d_model)
            value (torch.Tensor): Value tensor of shape (batch_size, value_len, d_model)
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            tuple: (output, attention_weights)
        """
        batch_size, query_len, d_model = query.size()
        _, key_len, _ = key.size()
        _, value_len, _ = value.size()
        
        # 1. Linear transformations and split into multiple heads
        Q = self.w_q(query).view(batch_size, query_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, key_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, value_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Apply scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. Concatenate heads and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.d_model
        )
        
        output = self.w_o(attention_output)
        
        return output, attention_weights


class TransformerBlock(nn.Module):
    """
    A complete Transformer block with multi-head attention and feed-forward network
    
    Args:
        d_model (int): Dimension of the model
        num_heads (int): Number of attention heads
        d_ff (int): Dimension of feed-forward network (default: 4 * d_model)
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Forward pass of the transformer block
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Multi-head attention with residual connection and layer norm
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward network with residual connection and layer norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


def create_padding_mask(seq, pad_token=0):
    """
    Create padding mask for sequences
    
    Args:
        seq (torch.Tensor): Input sequence of shape (batch_size, seq_len)
        pad_token (int): Token used for padding (default: 0)
        
    Returns:
        torch.Tensor: Padding mask of shape (batch_size, 1, 1, seq_len)
    """
    mask = (seq != pad_token).unsqueeze(1).unsqueeze(2)
    return mask


def create_look_ahead_mask(size):
    """
    Create look-ahead mask for decoder self-attention
    
    Args:
        size (int): Sequence length
        
    Returns:
        torch.Tensor: Look-ahead mask of shape (1, 1, size, size)
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return (mask == 0).unsqueeze(0).unsqueeze(0)


# Example usage and testing
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Model parameters
    d_model = 512
    num_heads = 8
    seq_len = 100
    batch_size = 32
    
    print("=== Multi-Head Attention Example ===")
    
    # Create model
    attention_model = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    # Create random input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass (self-attention)
    output, attention_weights = attention_model(x, x, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Model parameters: {sum(p.numel() for p in attention_model.parameters()):,}")
    
    print("\n=== Transformer Block Example ===")
    
    # Create transformer block
    transformer_block = TransformerBlock(d_model=d_model, num_heads=num_heads)
    
    # Forward pass
    block_output = transformer_block(x)
    
    print(f"Transformer block output shape: {block_output.shape}")
    print(f"Transformer block parameters: {sum(p.numel() for p in transformer_block.parameters()):,}")
    
    print("\n=== Attention with Mask Example ===")
    
    # Create a simple padding mask (assuming last 10 tokens are padding)
    seq_tokens = torch.randint(1, 1000, (batch_size, seq_len))
    seq_tokens[:, -10:] = 0  # Set last 10 tokens as padding
    
    padding_mask = create_padding_mask(seq_tokens, pad_token=0)
    
    # Forward pass with mask
    masked_output, masked_attention_weights = attention_model(x, x, x, mask=padding_mask)
    
    print(f"Masked output shape: {masked_output.shape}")
    print(f"Padding mask shape: {padding_mask.shape}")
    
    print("\n=== Look-ahead Mask Example ===")
    
    # Create look-ahead mask for decoder
    look_ahead_mask = create_look_ahead_mask(seq_len)
    
    decoder_output, decoder_attention_weights = attention_model(x, x, x, mask=look_ahead_mask)
    
    print(f"Decoder output shape: {decoder_output.shape}")
    print(f"Look-ahead mask shape: {look_ahead_mask.shape}")
    
    print("\n=== Multi-Head Attention Implementation Complete ===")