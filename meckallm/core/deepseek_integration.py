import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import numpy as np
from dataclasses import dataclass

@dataclass
class DeepSeekConfig:
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_experts: int = 256
    num_activated_experts: int = 37
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    max_position_embeddings: int = 32768
    vocab_size: int = 100000
    use_fp8: bool = True
    use_mtp: bool = True  # Multi-Token Prediction

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Latent attention components
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.latent_projection = nn.Linear(config.hidden_size, config.hidden_size)
        
        # FP8 support
        self.use_fp8 = config.use_fp8
        if self.use_fp8:
            self.query = self.query.to(torch.float8_e4m3fn)
            self.key = self.key.to(torch.float8_e4m3fn)
            self.value = self.value.to(torch.float8_e4m3fn)
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = hidden_states.size(0)
        
        # Project queries, keys, and values
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.head_dim)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Calculate attention scores with latent projection
        latent = self.latent_projection(hidden_states)
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / np.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Apply softmax
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value)
        context = context.view(batch_size, -1, self.config.hidden_size)
        
        if output_attentions:
            return context, attention_probs
        return context, None

class DeepSeekMoE(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        
        # Expert layers
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.hidden_size)
            )
            for _ in range(config.num_experts)
        ])
        
        # Router
        self.router = nn.Linear(config.hidden_size, config.num_experts)
        
        # FP8 support
        if config.use_fp8:
            for expert in self.experts:
                expert = expert.to(torch.float8_e4m3fn)
            self.router = self.router.to(torch.float8_e4m3fn)
            
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Calculate routing weights
        router_logits = self.router(hidden_states)
        router_weights = torch.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(
            router_weights,
            self.config.num_activated_experts,
            dim=-1
        )
        
        # Normalize weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Process through selected experts
        expert_outputs = []
        for i in range(self.config.num_activated_experts):
            expert_idx = top_k_indices[:, :, i]
            expert_weight = top_k_weights[:, :, i].unsqueeze(-1)
            
            # Gather expert outputs
            expert_output = torch.stack([
                self.experts[idx](hidden_states[b, :, :])
                for b, idx in enumerate(expert_idx)
            ])
            
            expert_outputs.append(expert_output * expert_weight)
            
        # Combine expert outputs
        output = torch.sum(torch.stack(expert_outputs), dim=0)
        
        return output

class DeepSeekBlock(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        
        # Attention layer
        self.attention = MultiHeadLatentAttention(config)
        
        # MoE layer
        self.moe = DeepSeekMoE(config)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Attention
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attention_output, attention_probs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions
        )
        hidden_states = residual + attention_output
        
        # MoE
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.moe(hidden_states)
        hidden_states = residual + hidden_states
        
        if output_attentions:
            return hidden_states, attention_probs
        return hidden_states, None

class DeepSeekModel(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DeepSeekBlock(config)
            for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size)
        
        # Output head
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Multi-Token Prediction
        self.use_mtp = config.use_mtp
        if self.use_mtp:
            self.mtp_head = nn.Linear(config.hidden_size, config.vocab_size * 4)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1)
        
        # Get embeddings
        position_ids = torch.arange(seq_length, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        hidden_states = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = hidden_states + position_embeddings
        
        # Process through transformer blocks
        all_attentions = []
        for block in self.blocks:
            hidden_states, attention_probs = block(
                hidden_states,
                attention_mask,
                output_attentions
            )
            if output_attentions:
                all_attentions.append(attention_probs)
                
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Output head
        logits = self.head(hidden_states)
        
        # Multi-Token Prediction
        if self.use_mtp:
            mtp_logits = self.mtp_head(hidden_states)
            mtp_logits = mtp_logits.view(
                batch_size,
                seq_length,
                4,
                self.config.vocab_size
            )
            logits = (logits, mtp_logits)
            
        if output_attentions:
            return logits, all_attentions
        return logits, None 