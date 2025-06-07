import pyautogui
import keyboard
import mouse
import win32gui
import win32con
import win32process
import psutil
import time
from typing import Tuple, List, Optional
import numpy as np
from dataclasses import dataclass
import cv2
import pytesseract
from PIL import ImageGrab
import logging

@dataclass
class ScreenRegion:
    x: int
    y: int
    width: int
    height: int
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

class SystemController:
    def __init__(self):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Configure PyAutoGUI
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.1
        
        # Initialize screen properties
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Initialize OCR
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
    def get_active_window(self) -> Tuple[str, int]:
        """Get the currently active window and its process ID"""
        window = win32gui.GetForegroundWindow()
        _, process_id = win32process.GetWindowThreadProcessId(window)
        title = win32gui.GetWindowText(window)
        return title, process_id
        
    def get_window_region(self, window_title: str) -> Optional[ScreenRegion]:
        """Get the region of a window by title"""
        def callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                if window_title.lower() in win32gui.GetWindowText(hwnd).lower():
                    rect = win32gui.GetWindowRect(hwnd)
                    extra.append(ScreenRegion(
                        x=rect[0],
                        y=rect[1],
                        width=rect[2] - rect[0],
                        height=rect[3] - rect[1]
                    ))
        
        windows = []
        win32gui.EnumWindows(callback, windows)
        return windows[0] if windows else None
        
    def capture_screen(self, region: Optional[ScreenRegion] = None) -> np.ndarray:
        """Capture screen or region of screen"""
        if region:
            screenshot = ImageGrab.grab(bbox=region.to_tuple())
        else:
            screenshot = ImageGrab.grab()
        return np.array(screenshot)
        
    def find_text_on_screen(self, text: str, region: Optional[ScreenRegion] = None) -> List[Tuple[int, int]]:
        """Find text on screen using OCR"""
        image = self.capture_screen(region)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        locations = []
        for i, word in enumerate(data['text']):
            if text.lower() in word.lower():
                x = data['left'][i]
                y = data['top'][i]
                locations.append((x, y))
                
        return locations
        
    def find_image_on_screen(self, template_path: str, confidence: float = 0.8) -> Optional[Tuple[int, int]]:
        """Find image on screen using template matching"""
        template = cv2.imread(template_path)
        screenshot = self.capture_screen()
        
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= confidence:
            return max_loc
        return None
        
    def click(self, x: int, y: int, button: str = 'left'):
        """Click at specified coordinates"""
        mouse.move(x, y)
        mouse.click(button=button)
        
    def type_text(self, text: str, interval: float = 0.1):
        """Type text with specified interval between keystrokes"""
        keyboard.write(text, delay=interval)
        
    def press_key(self, key: str):
        """Press a specific key"""
        keyboard.press_and_release(key)
        
    def hold_key(self, key: str):
        """Hold a key down"""
        keyboard.press(key)
        
    def release_key(self, key: str):
        """Release a held key"""
        keyboard.release(key)
        
    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5):
        """Perform drag operation"""
        mouse.move(start_x, start_y)
        mouse.drag(end_x, end_y, duration=duration)
        
    def scroll(self, amount: int):
        """Scroll by specified amount"""
        mouse.wheel(amount)
        
    def wait_for_window(self, window_title: str, timeout: float = 10.0) -> bool:
        """Wait for window to appear"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.get_window_region(window_title):
                return True
            time.sleep(0.1)
        return False
        
    def wait_for_text(self, text: str, region: Optional[ScreenRegion] = None, timeout: float = 10.0) -> bool:
        """Wait for text to appear on screen"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.find_text_on_screen(text, region):
                return True
            time.sleep(0.1)
        return False
        
    def wait_for_image(self, template_path: str, confidence: float = 0.8, timeout: float = 10.0) -> bool:
        """Wait for image to appear on screen"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.find_image_on_screen(template_path, confidence):
                return True
            time.sleep(0.1)
        return False
        
    def get_process_list(self) -> List[Tuple[str, int]]:
        """Get list of running processes"""
        processes = []
        for proc in psutil.process_iter(['name', 'pid']):
            try:
                processes.append((proc.info['name'], proc.info['pid']))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return processes
        
    def is_process_running(self, process_name: str) -> bool:
        """Check if process is running"""
        return any(proc[0].lower() == process_name.lower() for proc in self.get_process_list())
        
    def start_process(self, path: str) -> bool:
        """Start a process"""
        try:
            psutil.Popen(path)
            return True
        except Exception as e:
            self.logger.error(f"Failed to start process: {e}")
            return False
            
    def close_window(self, window_title: str) -> bool:
        """Close window by title"""
        def callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                if window_title.lower() in win32gui.GetWindowText(hwnd).lower():
                    win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
                    extra.append(True)
        
        closed = []
        win32gui.EnumWindows(callback, closed)
        return bool(closed)
        
    def maximize_window(self, window_title: str) -> bool:
        """Maximize window by title"""
        def callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                if window_title.lower() in win32gui.GetWindowText(hwnd).lower():
                    win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
                    extra.append(True)
        
        maximized = []
        win32gui.EnumWindows(callback, maximized)
        return bool(maximized)
        
    def minimize_window(self, window_title: str) -> bool:
        """Minimize window by title"""
        def callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                if window_title.lower() in win32gui.GetWindowText(hwnd).lower():
                    win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
                    extra.append(True)
        
        minimized = []
        win32gui.EnumWindows(callback, minimized)
        return bool(minimized) 