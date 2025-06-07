import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
import os
import gc
import psutil
import logging
from pathlib import Path
import shutil
from transformers import PreTrainedModel
import bitsandbytes as bnb
from safetensors.torch import save_file, load_file
import zstandard as zstd
import mmap
import tempfile

class ResourceManager:
    def __init__(
        self,
        max_memory_gb: float = 4.0,
        max_disk_gb: float = 10.0,
        compression_level: int = 3
    ):
        self.max_memory_gb = max_memory_gb
        self.max_disk_gb = max_disk_gb
        self.compression_level = compression_level
        self.temp_dir = tempfile.mkdtemp()
        self.memory_mapped_files: Dict[str, mmap.mmap] = {}
        
    def optimize_model_memory(self, model: nn.Module) -> nn.Module:
        """Apply memory optimization techniques to model"""
        # Enable gradient checkpointing
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            
        # Convert to 4-bit quantization
        for param in model.parameters():
            if param.dtype == torch.float32:
                param.data = bnb.nn.Params4bit(
                    param.data,
                    requires_grad=param.requires_grad,
                    compress_statistics=True,
                    quant_type="nf4"
                ).to(param.device)
                
        return model
        
    def memory_mapped_storage(
        self,
        tensor: torch.Tensor,
        name: str
    ) -> torch.Tensor:
        """Store tensor in memory-mapped file"""
        # Create memory-mapped file
        file_path = os.path.join(self.temp_dir, f"{name}.mmap")
        size = tensor.nelement() * tensor.element_size()
        
        with open(file_path, "wb") as f:
            f.write(b"\0" * size)
            
        # Memory map the file
        with open(file_path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            self.memory_mapped_files[name] = mm
            
            # Copy tensor data to memory-mapped file
            tensor_np = tensor.cpu().numpy()
            mm.write(tensor_np.tobytes())
            
        # Create new tensor that uses memory-mapped storage
        return torch.from_numpy(
            np.frombuffer(mm, dtype=tensor.dtype)
        ).reshape(tensor.shape)
        
    def compress_checkpoint(
        self,
        state_dict: Dict[str, torch.Tensor],
        path: str
    ) -> None:
        """Compress and save model checkpoint"""
        # Convert tensors to numpy arrays
        compressed_state = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                compressed_state[key] = value.cpu().numpy()
            else:
                compressed_state[key] = value
                
        # Save with safetensors
        save_file(compressed_state, path)
        
        # Compress the file
        with open(path, "rb") as f_in:
            with open(f"{path}.zst", "wb") as f_out:
                cctx = zstd.ZstdCompressor(level=self.compression_level)
                cctx.copy_stream(f_in, f_out)
                
        # Remove original file
        os.remove(path)
        
    def decompress_checkpoint(self, path: str) -> Dict[str, torch.Tensor]:
        """Load and decompress model checkpoint"""
        # Decompress the file
        with open(f"{path}.zst", "rb") as f_in:
            with open(path, "wb") as f_out:
                dctx = zstd.ZstdDecompressor()
                dctx.copy_stream(f_in, f_out)
                
        # Load with safetensors
        state_dict = load_file(path)
        
        # Convert numpy arrays back to tensors
        for key, value in state_dict.items():
            if isinstance(value, np.ndarray):
                state_dict[key] = torch.from_numpy(value)
                
        return state_dict
        
    def optimize_disk_usage(self, model_dir: str) -> None:
        """Optimize disk usage for model files"""
        # Remove unnecessary files
        for file in os.listdir(model_dir):
            if file.endswith((".bin", ".pt", ".pth")):
                file_path = os.path.join(model_dir, file)
                # Compress model files
                with open(file_path, "rb") as f_in:
                    with open(f"{file_path}.zst", "wb") as f_out:
                        cctx = zstd.ZstdCompressor(level=self.compression_level)
                        cctx.copy_stream(f_in, f_out)
                os.remove(file_path)
                
    def cleanup(self) -> None:
        """Clean up temporary files and memory-mapped files"""
        # Close memory-mapped files
        for mm in self.memory_mapped_files.values():
            mm.close()
            
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        return {
            "ram_used_gb": psutil.Process().memory_info().rss / 1024**3,
            "ram_available_gb": psutil.virtual_memory().available / 1024**3,
            "disk_used_gb": psutil.disk_usage("/").used / 1024**3,
            "disk_free_gb": psutil.disk_usage("/").free / 1024**3
        }
        
    def check_resources(self) -> bool:
        """Check if resource usage is within limits"""
        usage = self.get_memory_usage()
        
        if usage["ram_used_gb"] > self.max_memory_gb:
            return False
            
        if usage["disk_used_gb"] > self.max_disk_gb:
            return False
            
        return True

class EfficientModelLoader:
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        
    def load_model(
        self,
        model_path: str,
        device: str = "cuda"
    ) -> PreTrainedModel:
        """Load model with resource optimization"""
        # Check available resources
        if not self.resource_manager.check_resources():
            raise RuntimeError("Insufficient resources to load model")
            
        # Load model with safetensors
        model = PreTrainedModel.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        
        # Apply memory optimizations
        model = self.resource_manager.optimize_model_memory(model)
        
        return model
        
    def save_model(
        self,
        model: PreTrainedModel,
        path: str
    ) -> None:
        """Save model with resource optimization"""
        # Get model state dict
        state_dict = model.state_dict()
        
        # Compress and save checkpoint
        self.resource_manager.compress_checkpoint(state_dict, path)
        
        # Optimize disk usage
        self.resource_manager.optimize_disk_usage(os.path.dirname(path)) 