# Transformer Implementation from Scratch

A complete implementation of the Transformer architecture (encoder-decoder) built using only core PyTorch primitives.

## Overview
Implemented the full Transformer architecture from the ground up as part of Columbia University's Artificial Intelligence course (COMS 4701). Includes token embeddings, sinusoidal positional encoding, scaled dot-product attention, multi-head attention, encoder blocks, and decoder blocks with masked self-attention and cross-attention.

## Tech Stack
- **Language:** Python
- **Framework:** PyTorch (core primitives only)
- **Components:** nn.Embedding, nn.Linear, custom attention modules

## Key Components Implemented
1. **Token Embedding & Positional Encoding**
   - Learned token embeddings scaled by √d_model
   - Fixed sinusoidal positional encodings (registered as buffer)
   - Numerically stable computation using exponential form

2. **Scaled Dot-Product Attention**
   - `Attention(Q, K, V) = softmax(QKᵀ/√d_k) · V`
   - Optional masking for causal and padding masks

3. **Multi-Head Attention**
   - Split d_model into h heads of dimension d_k = d_model/h
   - Independent Q, K, V projections per head
   - Concatenation and output projection

4. **Transformer Encoder Block**
   - Multi-head self-attention + Add & Norm
   - Position-wise feed-forward network + Add & Norm

5. **Transformer Decoder Block**
   - Masked multi-head self-attention (causal mask)
   - Cross-attention (Q from decoder, K/V from encoder)
   - Position-wise feed-forward network

## Results
- **Achieved 95%+ accuracy on copy-task learning**
- Training loss decreases consistently over epochs
- End-to-end wiring verified through shape tests

## What I Learned
- Attention mechanisms powering modern LLMs
- Tensor manipulation (reshape, transpose, permute, contiguous)
- Masking strategies for autoregressive generation
- PyTorch nn.Module patterns, buffers, and training loops

## Note
This project was completed for a course. Source code is available upon request due to course policy restrictions. I'm happy to walk through the architecture, attention computations, and training process in an interview.
