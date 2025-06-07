import torch
from typing import List, Dict, Any
import os
from pathlib import Path
import json
from tqdm import tqdm
import argparse

from meckallm.learning.multimodal_learner import MultiModalConfig, MultiModalLearner

def load_dataset(data_dir: str) -> List[Dict[str, Any]]:
    """Load dataset from directory"""
    dataset = []
    
    # Walk through data directory
    for root, _, files in os.walk(data_dir):
        # Find matching triplets (text, image, audio)
        text_files = [f for f in files if f.endswith('.txt')]
        
        for text_file in text_files:
            base_name = text_file[:-4]  # Remove .txt extension
            image_file = f"{base_name}.jpg"
            audio_file = f"{base_name}.wav"
            
            # Check if all files exist
            if (os.path.exists(os.path.join(root, image_file)) and
                os.path.exists(os.path.join(root, audio_file))):
                
                # Load text
                with open(os.path.join(root, text_file), 'r') as f:
                    text = f.read().strip()
                
                # Create dataset entry
                dataset.append({
                    "text": text,
                    "image_path": os.path.join(root, image_file),
                    "audio_path": os.path.join(root, audio_file),
                    "target": torch.randn(768)  # Placeholder target
                })
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Train multi-modal learning system")
    parser.add_argument("--data_dir", type=str, required=True,
                      help="Directory containing training data")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save model checkpoints")
    parser.add_argument("--num_epochs", type=int, default=10,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--fusion_method", type=str, default="attention",
                      choices=["attention", "cross_attention", "concat"],
                      help="Fusion method to use")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.data_dir)
    print(f"Loaded {len(dataset)} samples")
    
    # Create configuration
    config = MultiModalConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        fusion_method=args.fusion_method
    )
    
    # Initialize learner
    print("Initializing learner...")
    learner = MultiModalLearner(config)
    
    # Train model
    print("Starting training...")
    losses = learner.train(dataset)
    
    # Save training history
    history = {
        "losses": losses,
        "config": vars(config)
    }
    with open(os.path.join(args.output_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save model state
    print("Saving model state...")
    learner.save_state(os.path.join(args.output_dir, "model_state.pt"))
    
    print("Training complete!")

if __name__ == "__main__":
    main()