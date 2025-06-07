from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json
import asyncio
from typing import List, Dict
import psutil
import GPUtil
from datetime import datetime
import uvicorn

app = FastAPI(title="MeckaLLM Monitor")

# Mount static files
app.mount("/static", StaticFiles(directory="meckallm/web/static"), name="static")

class SystemMonitor:
    def __init__(self):
        self.connected_clients: List[WebSocket] = []
        self.metrics_history = []
        
    async def broadcast_metrics(self):
        while True:
            if self.connected_clients:
                metrics = self.get_system_metrics()
                for client in self.connected_clients:
                    try:
                        await client.send_json(metrics)
                    except:
                        self.connected_clients.remove(client)
            await asyncio.sleep(1)
            
    def get_system_metrics(self) -> Dict:
        """Get comprehensive system metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'usage': psutil.cpu_percent(),
                'temperature': self.get_cpu_temperature(),
                'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 0
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'used': psutil.virtual_memory().used,
                'percent': psutil.virtual_memory().percent
            },
            'gpu': self.get_gpu_metrics(),
            'quantum_metrics': self.get_quantum_metrics()
        }
        self.metrics_history.append(metrics)
        return metrics
        
    def get_gpu_metrics(self) -> Dict:
        """Get GPU metrics"""
        try:
            gpus = GPUtil.getGPUs()
            return {
                f'gpu_{i}': {
                    'name': gpu.name,
                    'load': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'temperature': gpu.temperature
                }
                for i, gpu in enumerate(gpus)
            }
        except:
            return {}
            
    def get_cpu_temperature(self) -> float:
        """Get CPU temperature"""
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return max(temp.current for temp in temps['coretemp'])
            return 0.0
        except:
            return 0.0
            
    def get_quantum_metrics(self) -> Dict:
        """Get quantum metrics"""
        return {
            'entanglement': 0.85,  # Placeholder
            'coherence': 0.92,     # Placeholder
            'superposition': 0.78  # Placeholder
        }

monitor = SystemMonitor()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(monitor.broadcast_metrics())

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the dashboard HTML"""
    with open("meckallm/web/static/index.html", "r") as f:
        return f.read()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics"""
    await websocket.accept()
    monitor.connected_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        monitor.connected_clients.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 