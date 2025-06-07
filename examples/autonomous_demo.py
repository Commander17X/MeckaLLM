import asyncio
from meckallm.agents.autonomous_controller import AutonomousController, Task

async def main():
    # Initialize controller
    controller = AutonomousController()
    
    # Create some example tasks
    tasks = [
        Task(
            type="browser",
            description="Open YouTube and search for a video",
            parameters={
                "url": "https://www.youtube.com",
                "actions": [
                    {"type": "click", "x": 500, "y": 100},  # Search box
                    {"type": "type", "text": "quantum computing tutorial"},
                    {"type": "click", "x": 600, "y": 100}  # Search button
                ]
            },
            priority=2
        ),
        Task(
            type="music",
            description="Play music on Spotify",
            parameters={
                "action": "play"
            },
            priority=1
        ),
        Task(
            type="discord",
            description="Send a message on Discord",
            parameters={
                "action": "send_message",
                "message": "Hello from MeckaLLM!"
            },
            priority=3
        ),
        Task(
            type="minecraft",
            description="Start Minecraft and move around",
            parameters={
                "actions": [
                    {"type": "key", "key": "w"},
                    {"type": "key", "key": "space"},
                    {"type": "mouse", "x": 500, "y": 500}
                ]
            },
            priority=1
        )
    ]
    
    # Add tasks to queue
    for task in tasks:
        await controller.add_task(task)
    
    # Start processing tasks
    try:
        await controller.run()
    except KeyboardInterrupt:
        print("\nStopping controller...")
        controller.stop()
        
    # Print final status
    print("\nTask Status:")
    for status in controller.get_task_status():
        print(f"- {status['type']}: {status['status']}")

if __name__ == "__main__":
    asyncio.run(main()) 