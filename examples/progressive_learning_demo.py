import logging
from meckallm.learning.progressive_learner import ProgressiveLearner, ProgressiveLearningConfig
import time
import signal
import sys

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nStopping progressive learning...")
    if learner:
        learner.stop_learning()
    sys.exit(0)

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize configuration
    config = ProgressiveLearningConfig(
        learning_rate=1e-5,
        batch_size=4,
        max_length=8192,
        update_interval=60.0,
        max_memory_gb=4.0,
        max_disk_gb=10.0,
        compression_level=3,
        capture_screen=True,
        capture_keystrokes=True,
        capture_mouse=True,
        capture_apps=True,
        capture_system=True,
        data_dir="data/progressive_learning",
        model_dir="models/progressive_learning"
    )
    
    # Initialize learner
    global learner
    learner = ProgressiveLearner(config)
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start learning
    logger.info("Starting progressive learning...")
    learner.start_learning()
    
    try:
        while True:
            # Get insights every 5 minutes
            insights = learner.get_insights()
            logger.info("Current insights:")
            logger.info(f"Application usage: {insights['application_usage']}")
            logger.info(f"Interaction patterns: {insights['interaction_patterns']}")
            logger.info(f"Resource usage: {insights['resource_usage']}")
            
            time.sleep(300)  # Wait 5 minutes
            
    except KeyboardInterrupt:
        logger.info("Stopping progressive learning...")
        learner.stop_learning()
        
if __name__ == "__main__":
    main() 