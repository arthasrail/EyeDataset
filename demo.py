#!/usr/bin/env python3
"""
Simple demonstration of Multi-Head Attention usage

This script shows basic usage examples of the MultiHeadAttention class.
"""

import torch
from multi_head_attention import MultiHeadAttention, TransformerBlock

def basic_attention_demo():
    """Basic multi-head attention demonstration"""
    print("🚀 Basic Multi-Head Attention Demo")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a simple multi-head attention model
    d_model = 256  # Embedding dimension
    num_heads = 8  # Number of attention heads
    seq_len = 50   # Sequence length
    batch_size = 4 # Batch size
    
    attention = MultiHeadAttention(d_model, num_heads).to(device)
    
    # Create random input (representing embedded tokens)
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    print(f"Input shape: {x.shape}")
    print(f"Model parameters: {sum(p.numel() for p in attention.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        output, attention_weights = attention(x, x, x)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print()

def cross_attention_demo():
    """Cross-attention demonstration (encoder-decoder style)"""
    print("🔄 Cross-Attention Demo (Encoder-Decoder)")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parameters
    d_model = 256
    num_heads = 8
    encoder_seq_len = 20  # Source sequence length
    decoder_seq_len = 15  # Target sequence length
    batch_size = 4
    
    attention = MultiHeadAttention(d_model, num_heads).to(device)
    
    # Create encoder and decoder sequences
    encoder_output = torch.randn(batch_size, encoder_seq_len, d_model).to(device)
    decoder_input = torch.randn(batch_size, decoder_seq_len, d_model).to(device)
    
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Decoder input shape: {decoder_input.shape}")
    
    # Cross-attention: decoder queries attend to encoder keys and values
    with torch.no_grad():
        output, attention_weights = attention(
            query=decoder_input,     # Q from decoder
            key=encoder_output,      # K from encoder  
            value=encoder_output     # V from encoder
        )
    
    print(f"Cross-attention output shape: {output.shape}")
    print(f"Cross-attention weights shape: {attention_weights.shape}")
    print()

def transformer_block_demo():
    """Complete transformer block demonstration"""
    print("🧱 Transformer Block Demo")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parameters
    d_model = 256
    num_heads = 8
    seq_len = 30
    batch_size = 4
    
    # Create transformer block
    transformer_block = TransformerBlock(d_model, num_heads).to(device)
    
    # Input sequence
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    print(f"Input shape: {x.shape}")
    print(f"Transformer block parameters: {sum(p.numel() for p in transformer_block.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        output = transformer_block(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.4f}")
    print(f"Output std: {output.std().item():.4f}")
    print()

def attention_visualization_demo():
    """Demonstrate attention weight visualization"""
    print("👁️ Attention Visualization Demo")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Small example for visualization
    d_model = 64
    num_heads = 4
    seq_len = 10
    batch_size = 1
    
    attention = MultiHeadAttention(d_model, num_heads).to(device)
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    with torch.no_grad():
        output, attention_weights = attention(x, x, x)
    
    # Show attention pattern for the first head, first batch
    attn_head_0 = attention_weights[0, 0].cpu().numpy()
    
    print(f"Attention matrix shape for head 0: {attn_head_0.shape}")
    print("Attention weights for head 0 (first 5x5):")
    print(attn_head_0[:5, :5])
    print()

if __name__ == "__main__":
    print("Multi-Head Attention Demonstration")
    print("=" * 60)
    print()
    
    # Run all demos
    basic_attention_demo()
    cross_attention_demo()
    transformer_block_demo()
    attention_visualization_demo()
    
    print("✅ All demonstrations completed successfully!")
    print()
    print("📚 Usage Tips:")
    print("- Use MultiHeadAttention for self-attention or cross-attention")
    print("- Use TransformerBlock for complete transformer layers")
    print("- Adjust d_model and num_heads based on your requirements")
    print("- Remember that d_model must be divisible by num_heads")
    print("- Use masks for padding tokens or causal attention")