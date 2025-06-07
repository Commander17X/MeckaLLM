import torch
from meckallm.learning.autonomous_learner import AutonomousLearner, LearningConfig
from transformers import AutoTokenizer
import logging
import json
from pathlib import Path
import time

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize configuration
    config = LearningConfig(
        base_model="deepseek-ai/deepseek-coder-33b-instruct",
        auxiliary_model="facebook/opt-66b",
        knowledge_model="mistralai/Mistral-7B-v0.1",
        learning_rate=1e-5,
        batch_size=4,
        gradient_accumulation_steps=8,
        max_length=8192,
        use_4bit=True,
        use_8bit=False,
        use_flash_attention=True,
        use_gradient_checkpointing=True
    )
    
    # Initialize learner
    learner = AutonomousLearner(config)
    logger.info("Initialized autonomous learner")
    
    # Example training data
    training_data = [
        {
            "input": "Write a function to calculate the Fibonacci sequence",
            "output": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        },
        {
            "input": "Implement a binary search algorithm",
            "output": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"
        }
    ]
    
    # Tokenize training data
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    
    # Training loop
    logger.info("Starting training loop")
    for epoch in range(3):
        epoch_loss = 0
        for item in training_data:
            try:
                # Tokenize input and output
                inputs = tokenizer(
                    item["input"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.max_length
                )
                
                labels = tokenizer(
                    item["output"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.max_length
                )
                
                # Training step
                metrics = learner.train_step(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=labels["input_ids"]
                )
                
                epoch_loss += metrics["loss"]
                
                # Log climate metrics
                logger.info(f"Climate metrics: {metrics['climate_metrics']}")
                
            except Exception as e:
                logger.error(f"Error during training: {e}")
                continue
                
        logger.info(f"Epoch {epoch + 1} completed. Average loss: {epoch_loss / len(training_data)}")
        
    # Save model state
    save_path = Path("models/autonomous_learner")
    save_path.mkdir(parents=True, exist_ok=True)
    learner.save_state(str(save_path / "model_state.pt"))
    logger.info(f"Saved model state to {save_path}")
    
    # Test generation
    test_prompts = [
        "Write a function to sort a list using quicksort",
        "Implement a depth-first search algorithm",
        "Create a function to find the longest common subsequence"
    ]
    
    logger.info("Testing generation")
    for prompt in test_prompts:
        try:
            generated = learner.generate(
                prompt=prompt,
                max_length=200,
                temperature=0.7
            )
            logger.info(f"\nPrompt: {prompt}\nGenerated: {generated}\n")
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            continue
            
if __name__ == "__main__":
    main() 