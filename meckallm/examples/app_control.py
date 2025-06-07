import os
import sys
import time
import json
import logging
import keyboard
import pyautogui
import psutil
import win32gui
import win32con
import win32process
import win32api
import win32com.client
import speech_recognition as sr
from datetime import datetime
from gtts import gTTS
from transformers import pipeline
from rich.console import Console
from rich.table import Table
import webbrowser
import pyautogui
import cv2
import numpy as np
import yt_dlp
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import re
from PIL import ImageGrab
import tempfile
import pygame
import threading
from typing import Optional, Tuple, Dict, List, Any, Union

class SystemUtils:
    """Utility class for system operations."""
    
    def is_process_running(self, process_name: str) -> bool:
        """Check if a process is running."""
        try:
            for proc in psutil.process_iter(['name']):
                if proc.info['name'].lower() == process_name.lower():
                    return True
            return False
        except Exception:
            return False
    
    def wait_for_window(self, window_title: str, timeout: int = 10) -> bool:
        """Wait for a window to appear."""
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                if win32gui.FindWindow(None, window_title):
                    return True
                time.sleep(0.5)
            return False
        except Exception:
            return False

class AppControl:
    """Handles application control and learning through voice commands."""
    
    def __init__(self, console: Console = None):
        """Initialize the AppControl class."""
        self.console = console if console else Console()
        self.system = SystemUtils()
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300  # Lower threshold for better sensitivity
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Shorter pause detection
        
        # Initialize summarization
        try:
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            self.console.print("[green]Summarization model loaded successfully[/green]")
        except Exception as e:
            self.console.print(f"[yellow]Summarization model not available: {e}[/yellow]")
            self.summarizer = None
        
        # Initialize game paths
        self.game_paths = {
            'minecraft': r"C:\Program Files (x86)\Minecraft Launcher\MinecraftLauncher.exe",
            'fortnite': r"C:\Program Files\Epic Games\Fortnite\FortniteGame\Binaries\Win64\FortniteClient-Win64-Shipping.exe",
            'valorant': r"C:\Riot Games\VALORANT\live\VALORANT.exe",
            'spotify': r"C:\Users\Public\Desktop\Spotify.lnk",
            'notepad': r"C:\Windows\System32\notepad.exe",
            'discord': r"C:\Program Files (x86)\Discord\Discord.exe"
        }
        
        # Initialize notes directory
        self.notes_dir = os.path.join(os.path.expanduser("~"), "Documents", "MeckaLLM_Notes")
        self.stories_dir = os.path.join(self.notes_dir, "Stories")
        os.makedirs(self.notes_dir, exist_ok=True)
        os.makedirs(self.stories_dir, exist_ok=True)
        
        # Initialize BraveYouTube if available
        self.brave_youtube = None
        try:
            from brave_youtube import BraveYouTube
            self.brave_youtube = BraveYouTube(self.console)
            self.console.print("[green]BraveYouTube module initialized successfully[/green]")
        except ImportError:
            self.console.print("[yellow]BraveYouTube module not available - YouTube features will be disabled[/yellow]")
        except Exception as e:
            self.console.print(f"[red]Error initializing BraveYouTube: {e}[/red]")
        
        # Start app detection
        self._start_app_detection()
        
        self.voice_commands = {
            'nl': {
                'stop': ['stop', 'sluit', 'beëindig', 'close'],
                'start': ['start', 'open', 'begin', 'launch'],
                'pause': ['pauzeer', 'wacht', 'pause'],
                'resume': ['hervat', 'ga door', 'resume'],
                'learn': ['leer van', 'analyseer', 'bestudeer', 'learn from'],
                'search': ['zoek', 'search'],
                'cleanup': {
                    'all': ['clean', 'clean up', 'clean all']
                }
            },
            'en': {
                'stop': ['stop', 'close', 'end', 'terminate'],
                'start': ['start', 'open', 'begin', 'launch'],
                'pause': ['pause', 'wait', 'hold'],
                'resume': ['resume', 'continue', 'proceed'],
                'learn': ['learn from', 'analyze', 'study'],
                'search': ['search'],
                'cleanup': {
                    'all': ['clean', 'clean up', 'clean all']
                }
            }
        }
        
        self.active_apps = {}
        self.app_history = {}
        self.learning_data = {}
        self.last_update = datetime.now()
        self.update_interval = 5  # seconds
        self.active_sessions = {}
    
    def _start_app_detection(self):
        """Start periodic app detection in background."""
        def detect_apps():
            while True:
                self._detect_running_apps()
                time.sleep(self.update_interval)
        
        thread = threading.Thread(target=detect_apps, daemon=True)
        thread.start()
    
    def _detect_running_apps(self):
        """Detect all running applications and their states."""
        try:
            current_apps = {}
            
            def callback(hwnd, extra):
                if win32gui.IsWindowVisible(hwnd):
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    try:
                        process = psutil.Process(pid)
                        app_name = process.name()
                        window_title = win32gui.GetWindowText(hwnd)
                        
                        if app_name and window_title:
                            current_apps[app_name] = {
                                'pid': pid,
                                'title': window_title,
                                'path': process.exe(),
                                'memory': process.memory_info().rss,
                                'cpu': process.cpu_percent(),
                                'status': 'running',
                                'last_seen': datetime.now()
                            }
                            
                            # Update learning data
                            self._update_learning_data(app_name, process)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            
            win32gui.EnumWindows(callback, None)
            
            # Update active apps
            self.active_apps = current_apps
            
            # Update app history
            for app_name, app_data in current_apps.items():
                if app_name not in self.app_history:
                    self.app_history[app_name] = []
                self.app_history[app_name].append(app_data)
            
            self.last_update = datetime.now()
            
        except Exception as e:
            self.console.print(f"[red]Fout bij detecteren van applicaties: {e}[/red]")
    
    def _update_learning_data(self, app_name: str, process: psutil.Process):
        """Update learning data for an application."""
        try:
            if app_name not in self.learning_data:
                self.learning_data[app_name] = {
                    'usage_count': 0,
                    'total_runtime': 0,
                    'memory_patterns': [],
                    'cpu_patterns': [],
                    'interaction_times': [],
                    'last_used': None,
                    'common_actions': {},
                    'error_patterns': [],
                    'searches': [],
                    'watch_history': []
                }
            
            data = self.learning_data[app_name]
            data['usage_count'] += 1
            data['last_used'] = datetime.now()
            
            # Track resource usage patterns
            data['memory_patterns'].append(process.memory_info().rss)
            data['cpu_patterns'].append(process.cpu_percent())
            
            # Keep only recent patterns
            max_patterns = 100
            data['memory_patterns'] = data['memory_patterns'][-max_patterns:]
            data['cpu_patterns'] = data['cpu_patterns'][-max_patterns:]
            
            # Track interaction times
            data['interaction_times'].append(datetime.now())
            data['interaction_times'] = data['interaction_times'][-max_patterns:]
            
        except Exception as e:
            self.console.print(f"[red]Fout bij updaten van leerdata voor {app_name}: {e}[/red]")
    
    def is_app_control_prompt(self, prompt: str) -> bool:
        """Check if prompt is related to application control."""
        prompt = prompt.lower()
        
        # Common prefixes and variations
        prefixes = ['ja', 'nee', 'ok', 'oke', 'goed', 'prima', 'start', 'open', 'begin', 'launch']
        
        # Remove common prefixes
        for prefix in prefixes:
            if prompt.startswith(prefix + ' '):
                prompt = prompt[len(prefix) + 1:]
        
        # Check for any command pattern
        command_patterns = {
            'spotify': ['spotify', 'open spotify', 'start spotify', 'spotify openen'],
            'minecraft': ['minecraft', 'open minecraft', 'start minecraft'],
            'fortnite': ['fortnite', 'open fortnite', 'start fortnite'],
            'valorant': ['valorant', 'open valorant', 'start valorant'],
            'brave': ['brave', 'open brave', 'start brave'],
            'youtube': ['youtube', 'open youtube', 'start youtube']
        }
        
        # Check if any command pattern matches
        for commands in command_patterns.values():
            if any(cmd in prompt for cmd in commands):
                return True
        
        return False

    def handle_app_control_prompt(self, prompt: str) -> tuple[bool, str]:
        """Handle application control prompts."""
        try:
            # Clean up the prompt
            prompt = prompt.lower().strip()
            
            # Common prefixes to remove
            prefixes = [
                'ik wil graag', 'ik zou graag', 'kun je', 'kunt u', 'zou je', 'zou u',
                'alsjeblieft', 'graag', 'wil je', 'mag ik', 'ik wil', 'ik zou',
                'please', 'could you', 'would you', 'can you', 'i want', 'i would like'
            ]
            
            # Remove common prefixes
            for prefix in prefixes:
                if prompt.startswith(prefix + ' '):
                    prompt = prompt[len(prefix) + 1:]
            
            # Help command
            if prompt in ['help', 'hulp', 'commando\'s', 'commands']:
                return self._show_help()

            # Direct website URL
            if re.match(r'^[a-zA-Z0-9][a-zA-Z0-9-]*\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?$', prompt):
                return self._control_browser('search', f"zoek {prompt}")

            # Story commands (check these first)
            if re.match(r'^(verhaal|story|vertel|tell|opname|record|schrijf|write)(\s+(start|begin|starten|beginnen|op|on))?$', prompt):
                return self._handle_story('record')
            elif re.match(r'^(verhaal|story|vertel|tell|opname|record|schrijf|write)\s+(stop|stoppen|eindig|end|uit|off)$', prompt):
                return self._handle_story('stop')
            elif re.match(r'^(verhaal|story|vertel|tell|opname|record|schrijf|write)\s+(samenvatting|summary|samenvatten|summarize)$', prompt):
                return self._handle_story('summarize')
            elif re.match(r'^(verhalen|stories|opnames|records)\s+(lijst|list|toon|show|bekijk|view)$', prompt):
                return self._handle_story('list')

            # Notepad commands
            if re.match(r'^(notepad|kladblok|notities|notes|note|schrijf|write)(\s+(openen|starten|start|open|begin|launch))?$', prompt):
                return self._control_notepad('open', prompt)
            elif re.match(r'^(notepad|kladblok|notities|notes|note|schrijf|write)\s+(nieuw|new|maak|create|schrijf|write|op)\s+(.+)$', prompt):
                return self._control_notepad('new', prompt)
            elif re.match(r'^(notepad|kladblok|notities|notes|note|schrijf|write)\s+(lijst|list|toon|show|bekijk|view)$', prompt):
                return self._control_notepad('list', prompt)
            elif re.match(r'^(notepad|kladblok|notities|notes|note|schrijf|write)\s+(verwijder|delete|wis|clear)\s+(.+)$', prompt):
                return self._control_notepad('delete', prompt)

            # Application commands
            if re.match(r'^(open|start)\s+(discord|chat|voice)$', prompt):
                return self._control_application('discord')
            elif re.match(r'^(open|start)\s+(brave|browser|navigator)$', prompt):
                return self._control_browser('open')
            elif re.match(r'^zoek(?:\s+op\s+(youtube|google|wikipedia))?\s+(.+)$', prompt):
                return self._control_browser('search', prompt)
            elif re.match(r'^(download|downloaden)\s+(video|filmpje)$', prompt):
                return self._control_browser('download')
            elif re.match(r'^(analyseer|analyze)\s+(video|filmpje)$', prompt):
                return self._control_browser('analyze')
            elif re.match(r'^(cookies|privacy)\s+(accepteren|accept)$', prompt):
                return self._control_browser('accept_cookies')

            # Spotify commands
            if re.match(r'^(spotify|muziek|music|player|speler)\s+(volume|geluid)\s+(op|to|naar|set|zet)\s+(\d+)(\s*%|\s+procent)?$', prompt):
                return self._control_spotify('volume_set', prompt)
            elif re.match(r'^(spotify|muziek|music|player|speler)\s+(\d+)(\s*%|\s+procent)?\s+(luider|harder)$', prompt):
                return self._control_spotify('volume_up', prompt)
            elif re.match(r'^(spotify|muziek|music|player|speler)\s+(\d+)(\s*%|\s+procent)?\s+(zachter|stillere)$', prompt):
                return self._control_spotify('volume_down', prompt)
            elif re.match(r'^(spotify|muziek|music|player|speler)\s+(luider|harder)$', prompt):
                return self._control_spotify('volume_up', prompt)
            elif re.match(r'^(spotify|muziek|music|player|speler)\s+(zachter|stillere)$', prompt):
                return self._control_spotify('volume_down', prompt)
            elif re.match(r'^(spotify|muziek|music|player|speler)\s+(dempen|mute)$', prompt):
                return self._control_spotify('mute', prompt)
            elif re.match(r'^(spotify|muziek|music|player|speler)\s+(pauzeer|pauze|stop)$', prompt):
                return self._control_spotify('pause', prompt)
            elif re.match(r'^(spotify|muziek|music|player|speler)\s+(hervat|doorgaan|start)$', prompt):
                return self._control_spotify('resume', prompt)
            elif re.match(r'^(spotify|muziek|music|player|speler)\s+(volgende|next|skip)$', prompt):
                return self._control_spotify('next', prompt)
            elif re.match(r'^(spotify|muziek|music|player|speler)\s+(vorige|previous|terug)$', prompt):
                return self._control_spotify('previous', prompt)
            elif re.match(r'^(open|start)\s+(spotify|muziek|music|player|speler)$', prompt):
                return self._control_spotify('open', prompt)

            return False, "Commando niet herkend. Zeg 'help' voor beschikbare commando's."
            
        except Exception as e:
            return False, f"Fout bij verwerking: {str(e)}"

    def _use_run_dialog(self, command: str) -> None:
        """Use Windows Run dialog to execute a command."""
        try:
            # Open Run dialog
            keyboard.press_and_release('win+r')
            time.sleep(0.5)  # Wait for Run dialog
            
            # Type the command
            keyboard.write(command)
            time.sleep(0.2)
            
            # Press Enter
            keyboard.press_and_release('enter')
        except Exception as e:
            print(f"Error using Run dialog: {str(e)}")

    def _control_application(self, app: str) -> tuple[bool, str]:
        """Control application actions."""
        try:
            if app == 'discord':
                if not self.system.is_process_running("Discord.exe"):
                    self._use_run_dialog("discord")
                    return True, "Discord is gestart"
                return True, "Discord is al geopend"
            return False, "Onbekende applicatie"
        except Exception as e:
            return False, f"Fout bij applicatie controle: {str(e)}"

    def _start_application(self, app_name: str) -> tuple[bool, str]:
        """Start an application."""
        try:
            if app_name in self.game_paths:
                path = self.game_paths[app_name]
                if os.path.exists(path):
                    os.startfile(path)
                    self.console.print(f"[green]{app_name} is gestart[/green]")
                    return True, f"{app_name} started"
            
            return False, "Application not found"
            
        except Exception as e:
            self.console.print(f"[red]Error starting application: {e}[/red]")
            return False, f"Error: {str(e)}"
    
    def _send_game_chat(self, game: str, message: str) -> tuple[bool, str]:
        """Send a chat message in the game."""
        try:
            keyboard.press('t')
            keyboard.write(message)
            keyboard.press('enter')
            return True, f"Sent message in {game}"
        except Exception as e:
            return False, f"Error sending message: {str(e)}"
    
    def _send_game_command(self, game: str, command: str) -> tuple[bool, str]:
        """Send a command in the game."""
        try:
            keyboard.press('t')
            keyboard.write(f"/{command}")
            keyboard.press('enter')
            return True, f"Sent command in {game}"
        except Exception as e:
            return False, f"Error sending command: {str(e)}"
    
    def _process_natural_command(self, game: str, command: str) -> tuple[bool, str]:
        """Process natural language commands for games with immediate execution."""
        try:
            # Advanced action mappings with immediate execution
            action_mappings = {
                'honger': {
                    'actions': [
                        ('e', 'press'),
                        ('1', 'press'),
                        ('right', 'press'),
                        ('e', 'press')
                    ],
                    'message': 'Eten gegeten'
                },
                'eten': {
                    'actions': [
                        ('e', 'press'),
                        ('1', 'press'),
                        ('right', 'press'),
                        ('e', 'press')
                    ],
                    'message': 'Eten gegeten'
                },
                'plaats': {
                    'actions': [
                        ('right', 'press')
                    ],
                    'message': 'Blok geplaatst'
                },
                'breek': {
                    'actions': [
                        ('left', 'press')
                    ],
                    'message': 'Blok gebroken'
                },
                'spring': {
                    'actions': [
                        ('space', 'press')
                    ],
                    'message': 'Gesprongen'
                },
                'loop': {
                    'actions': [
                        ('w', 'hold')
                    ],
                    'message': 'Aan het lopen'
                },
                'ren': {
                    'actions': [
                        ('w', 'hold'),
                        ('ctrl', 'hold')
                    ],
                    'message': 'Aan het rennen'
                },
                'sprint': {
                    'actions': [
                        ('w', 'hold'),
                        ('ctrl', 'hold')
                    ],
                    'message': 'Aan het sprinten'
                },
                'stop': {
                    'actions': [
                        ('w', 'release'),
                        ('a', 'release'),
                        ('s', 'release'),
                        ('d', 'release'),
                        ('ctrl', 'release'),
                        ('shift', 'release')
                    ],
                    'message': 'Gestopt'
                },
                'links': {
                    'actions': [
                        ('a', 'hold')
                    ],
                    'message': 'Naar links'
                },
                'rechts': {
                    'actions': [
                        ('d', 'hold')
                    ],
                    'message': 'Naar rechts'
                },
                'achteruit': {
                    'actions': [
                        ('s', 'hold')
                    ],
                    'message': 'Achteruit'
                },
                'spring_loop': {
                    'actions': [
                        ('w', 'hold'),
                        ('space', 'hold')
                    ],
                    'message': 'Springend lopen'
                },
                'sprint_spring': {
                    'actions': [
                        ('w', 'hold'),
                        ('ctrl', 'hold'),
                        ('space', 'hold')
                    ],
                    'message': 'Springend sprinten'
                },
                'sneak': {
                    'actions': [
                        ('shift', 'hold')
                    ],
                    'message': 'Sluipen'
                },
                'sneak_loop': {
                    'actions': [
                        ('w', 'hold'),
                        ('shift', 'hold')
                    ],
                    'message': 'Sluipend lopen'
                },
                'sprint_links': {
                    'actions': [
                        ('w', 'hold'),
                        ('a', 'hold'),
                        ('ctrl', 'hold')
                    ],
                    'message': 'Sprinten naar links'
                },
                'sprint_rechts': {
                    'actions': [
                        ('w', 'hold'),
                        ('d', 'hold'),
                        ('ctrl', 'hold')
                    ],
                    'message': 'Sprinten naar rechts'
                },
                'spring_links': {
                    'actions': [
                        ('a', 'hold'),
                        ('space', 'hold')
                    ],
                    'message': 'Springen naar links'
                },
                'spring_rechts': {
                    'actions': [
                        ('d', 'hold'),
                        ('space', 'hold')
                    ],
                    'message': 'Springen naar rechts'
                },
                'sneak_links': {
                    'actions': [
                        ('a', 'hold'),
                        ('shift', 'hold')
                    ],
                    'message': 'Sluipen naar links'
                },
                'sneak_rechts': {
                    'actions': [
                        ('d', 'hold'),
                        ('shift', 'hold')
                    ],
                    'message': 'Sluipen naar rechts'
                }
            }
            
            # Convert command to lowercase
            command = command.lower()
            
            # Check for known actions
            for action, mapping in action_mappings.items():
                if action in command:
                    # Execute the sequence of actions immediately
                    for key, action_type in mapping['actions']:
                        if action_type == 'release':
                            keyboard.release(key)
                        elif action_type == 'hold':
                            keyboard.press(key)
                        elif action_type == 'press':
                            keyboard.press(key)
                            keyboard.release(key)
                    
                    return True, mapping['message']
            
            # Handle complex commands with immediate execution
            if 'doe' in command or 'voer uit' in command:
                # Extract the action after "doe" or "voer uit"
                action = command.split('doe')[-1].split('voer uit')[-1].strip()
                
                # Advanced movement combinations
                if 'spring' in action and 'loop' in action:
                    keyboard.press('w')
                    keyboard.press('space')
                    return True, 'Springend lopen'
                
                elif 'sprint' in action and 'spring' in action:
                    keyboard.press('w')
                    keyboard.press('ctrl')
                    keyboard.press('space')
                    return True, 'Springend sprinten'
                
                elif 'sneak' in action and 'loop' in action:
                    keyboard.press('w')
                    keyboard.press('shift')
                    return True, 'Sluipend lopen'
                
                elif 'sprint' in action and 'links' in action:
                    keyboard.press('w')
                    keyboard.press('a')
                    keyboard.press('ctrl')
                    return True, 'Sprinten naar links'
                
                elif 'sprint' in action and 'rechts' in action:
                    keyboard.press('w')
                    keyboard.press('d')
                    keyboard.press('ctrl')
                    return True, 'Sprinten naar rechts'
                
                elif 'spring' in action and 'links' in action:
                    keyboard.press('a')
                    keyboard.press('space')
                    return True, 'Springen naar links'
                
                elif 'spring' in action and 'rechts' in action:
                    keyboard.press('d')
                    keyboard.press('space')
                    return True, 'Springen naar rechts'
                
                elif 'sneak' in action and 'links' in action:
                    keyboard.press('a')
                    keyboard.press('shift')
                    return True, 'Sluipen naar links'
                
                elif 'sneak' in action and 'rechts' in action:
                    keyboard.press('d')
                    keyboard.press('shift')
                    return True, 'Sluipen naar rechts'
                
                # Basic movements
                elif 'spring' in action:
                    keyboard.press('space')
                    keyboard.release('space')
                    return True, 'Gesprongen'
                
                elif 'ren' in action or 'hardlopen' in action:
                    keyboard.press('w')
                    keyboard.press('ctrl')
                    return True, 'Aan het rennen'
                
                elif 'loop' in action or 'lopen' in action:
                    keyboard.press('w')
                    return True, 'Aan het lopen'
                
                elif 'stop' in action or 'halt' in action:
                    keyboard.release('w')
                    keyboard.release('a')
                    keyboard.release('s')
                    keyboard.release('d')
                    keyboard.release('ctrl')
                    keyboard.release('shift')
                    return True, 'Gestopt'
                
                elif 'plaats' in action or 'zet' in action:
                    keyboard.press('right')
                    keyboard.release('right')
                    return True, 'Blok geplaatst'
                
                elif 'breek' in action or 'vernietig' in action:
                    keyboard.press('left')
                    keyboard.release('left')
                    return True, 'Blok gebroken'
            
            return False, f"Commando niet begrepen: {command}"
            
        except Exception as e:
            self.console.print(f"[red]Error processing natural command: {e}[/red]")
            return False, f"Error: {str(e)}"
    
    def _control_game(self, game: str, action: str) -> tuple[bool, str]:
        """Control game actions using keyboard and mouse."""
        try:
            # First try to process as a natural language command
            success, message = self._process_natural_command(game, action)
            if success:
                return True, message
            
            # If not a natural language command, proceed with regular control
            # Game control mappings
            game_controls = {
                'minecraft': {
                    'walk': 'w',
                    'run': 'w',
                    'sprint': 'w',
                    'stop': 's',
                    'jump': 'space',
                    'attack': 'left',
                    'use': 'right',
                    'inventory': 'e',
                    'chat': 't',
                    'command': 't',
                    'forward': 'w',
                    'backward': 's',
                    'left': 'a',
                    'right': 'd',
                    'sneak': 'shift',
                    'drop': 'q',
                    'switch': '1-9',
                    'look_up': 'up',
                    'look_down': 'down',
                    'look_left': 'left',
                    'look_right': 'right'
                },
                'fortnite': {
                    'build': 'b',
                    'shoot': 'left',
                    'jump': 'space',
                    'ability1': 'q',
                    'ability2': 'e',
                    'ability3': 'c',
                    'forward': 'w',
                    'backward': 's',
                    'left': 'a',
                    'right': 'd',
                    'sprint': 'shift',
                    'crouch': 'ctrl',
                    'reload': 'r',
                    'use': 'e',
                    'inventory': 'i',
                    'map': 'm',
                    'emote': 'b'
                },
                'valorant': {
                    'shoot': 'left',
                    'ability1': 'q',
                    'ability2': 'e',
                    'ability3': 'c',
                    'ultimate': 'x',
                    'forward': 'w',
                    'backward': 's',
                    'left': 'a',
                    'right': 'd',
                    'sprint': 'shift',
                    'crouch': 'ctrl',
                    'reload': 'r',
                    'use': 'f',
                    'map': 'm'
                }
            }
            
            # Convert action to lowercase
            action = action.lower()
            
            # Check if game is supported
            if game not in game_controls:
                return False, f"Game {game} not supported"
            
            # Get game controls
            controls = game_controls[game]
            
            # Handle continuous actions (movement)
            if action in ['walk', 'run', 'sprint', 'forward', 'loop']:
                key = controls['forward']
                keyboard.press(key)
                if action in ['sprint', 'run']:
                    keyboard.press('ctrl')
                return True, f"Started {action} in {game}"
            
            # Handle stop actions
            elif action == 'stop':
                keyboard.release('w')
                keyboard.release('a')
                keyboard.release('s')
                keyboard.release('d')
                keyboard.release('ctrl')
                keyboard.release('shift')
                return True, f"Stopped all movement in {game}"
            
            # Handle jump actions
            elif action == 'jump':
                keyboard.press(controls['jump'])
                keyboard.release(controls['jump'])
                return True, f"Jumped in {game}"
            
            # Handle attack actions
            elif action == 'attack':
                pyautogui.mouseDown(button='left')
                pyautogui.mouseUp(button='left')
                return True, f"Attacked in {game}"
            
            # Handle inventory actions
            elif action == 'inventory':
                keyboard.press(controls['inventory'])
                keyboard.release(controls['inventory'])
                return True, f"Toggled inventory in {game}"
            
            # Handle chat and command actions
            elif action in ['chat', 'command']:
                keyboard.press(controls['chat'])
                keyboard.release(controls['chat'])
                return True, f"Opened {action} in {game}"
            
            # Handle mouse movement
            elif action.startswith('look_'):
                direction = action.split('_')[1]
                if direction in ['up', 'down', 'left', 'right']:
                    key = controls[f'look_{direction}']
                    keyboard.press(key)
                    keyboard.release(key)
                    return True, f"Looking {direction} in {game}"
            
            # Handle strafing
            elif action in ['left', 'right']:
                key = controls[action]
                keyboard.press(key)
                keyboard.release(key)
                return True, f"Strafed {action} in {game}"
            
            # Handle game-specific actions
            elif action in controls:
                keyboard.press(controls[action])
                keyboard.release(controls[action])
                return True, f"Executed {action} in {game}"
            
            # Handle improvisation - try to interpret the action as a key
            else:
                # Check if action is a single key
                if len(action) == 1 and action.isalnum():
                    keyboard.press(action)
                    keyboard.release(action)
                    return True, f"Pressed {action} in {game}"
                
                # Check if action is a mouse button
                elif action in ['left_click', 'right_click', 'middle_click']:
                    button = action.split('_')[0]
                    pyautogui.click(button=button)
                    return True, f"Clicked {button} mouse button in {game}"
                
                # Check if action is a mouse movement
                elif action.startswith('mouse_'):
                    try:
                        # Format: mouse_x_y (e.g., mouse_100_200)
                        _, x, y = action.split('_')
                        pyautogui.moveRel(int(x), int(y))
                        return True, f"Moved mouse by {x},{y} in {game}"
                    except:
                        pass
            
            return False, f"Action {action} not supported for {game}"
            
        except Exception as e:
            self.console.print(f"[red]Error controlling game: {e}[/red]")
            return False, f"Error controlling game: {str(e)}"

    def _start_game(self, game: str) -> tuple[bool, str]:
        """Start a game."""
        try:
            import subprocess
            
            game_paths = {
                'minecraft': r"C:\Program Files (x86)\Minecraft Launcher\MinecraftLauncher.exe",
                'fortnite': r"C:\Program Files\Epic Games\Fortnite\FortniteGame\Binaries\Win64\FortniteClient-Win64-Shipping.exe",
                'valorant': r"C:\Riot Games\VALORANT\live\VALORANT.exe"
            }
            
            if game.lower() in game_paths:
                subprocess.Popen(game_paths[game.lower()])
                return True, f"{game} started"
            return False, "game not found"
            
        except Exception as e:
            return False, "start failed"

    def _stop_game(self, game: str) -> tuple[bool, str]:
        """Stop a game."""
        try:
            import psutil
            
            game_processes = {
                'minecraft': ['javaw.exe', 'Minecraft.Windows.exe'],
                'fortnite': ['FortniteClient-Win64-Shipping.exe'],
                'valorant': ['VALORANT.exe', 'VALORANT-Win64-Shipping.exe']
            }
            
            if game.lower() in game_processes:
                for proc in psutil.process_iter(['name']):
                    if proc.info['name'] in game_processes[game.lower()]:
                        proc.terminate()
                return True, f"{game} stopped"
            return False, "game not found"
            
        except Exception as e:
            return False, "stop failed"

    def _initialize_browser(self):
        """Initialize browser with custom options."""
        try:
            # Define browser paths with Brave as first option
            browser_paths = {
                'brave': r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
                'chrome': r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                'edge': r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                'firefox': r"C:\Program Files\Mozilla Firefox\firefox.exe"
            }
            
            # Try Brave first, then fall back to others
            options = Options()
            if os.path.exists(browser_paths['brave']):
                options.binary_location = browser_paths['brave']
            else:
                # Try other browsers if Brave is not found
                for browser_name, path in browser_paths.items():
                    if browser_name != 'brave' and os.path.exists(path):
                        options.binary_location = path
                        break
                else:
                    # If no browser found, try default
                    options.binary_location = None
            
            # Common options for all browsers
            options.add_argument("--start-maximized")
            options.add_argument("--disable-notifications")
            options.add_argument("--disable-popup-blocking")
            options.add_argument("--log-level=3")  # Only show fatal errors
            options.add_experimental_option('excludeSwitches', ['enable-logging'])  # Disable logging
            
            # Initialize the browser
            self.browser = webdriver.Chrome(options=options)
            return True
        except Exception as e:
            return False

    def _open_brave(self, language: str) -> tuple[bool, str]:
        """Open Brave browser."""
        try:
            if not self.browser:
                self._initialize_browser()
            
            self.browser.get("https://www.brave.com")
            return True, "browser opened"
            
        except Exception as e:
            return False, "browser open failed"

    def _open_youtube(self, language: str) -> tuple[bool, str]:
        """Open YouTube in Brave browser."""
        try:
            if not self.browser:
                self._initialize_browser()
            
            self.browser.get("https://www.youtube.com")
            
            # Wait for YouTube to load silently
            WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located((By.ID, "search"))
            )
            
            return True, "youtube opened"
            
        except Exception as e:
            return False, "youtube open failed"

    def _search_youtube(self, query: str, language: str) -> tuple[bool, str]:
        """Search YouTube for a query."""
        try:
            if not self.browser:
                self._initialize_browser()
                self.browser.get("https://www.youtube.com")
            
            # Find search box and enter query
            search_box = WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located((By.ID, "search"))
            )
            search_box.clear()
            search_box.send_keys(query)
            search_box.submit()
            
            # Store search in learning data silently
            self.learning_data['searches'].append({
                'query': query,
                'timestamp': datetime.now()
            })
            
            return True, "search completed"
            
        except Exception as e:
            return False, "search failed"

    def _download_current_video(self, language: str) -> tuple[bool, str]:
        """Download the currently playing YouTube video."""
        try:
            if not self.browser:
                return False, "no active video"
            
            # Get current video URL
            current_url = self.browser.current_url
            if 'youtube.com/watch' not in current_url:
                return False, "no youtube video"
            
            # Configure yt-dlp options with minimal output
            ydl_opts = {
                'format': 'best',
                'outtmpl': 'downloads/%(title)s.%(ext)s',
                'quiet': True,
                'no_warnings': True,
                'logtostderr': False
            }
            
            # Download video silently
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(current_url, download=True)
                video_title = info['title']
            
            # Store download info silently
            self.downloaded_videos[video_title] = {
                'url': current_url,
                'timestamp': datetime.now(),
                'path': f"downloads/{video_title}.mp4"
            }
            
            return True, "video downloaded"
            
        except Exception as e:
            return False, "download failed"

    def _analyze_current_video(self, language: str) -> tuple[bool, str]:
        """Analyze the currently playing YouTube video."""
        try:
            if not self.browser:
                return False, "no active video"
            
            # Get current video URL
            current_url = self.browser.current_url
            if 'youtube.com/watch' not in current_url:
                return False, "no youtube video"
            
            # Get video information silently
            video_title = self.browser.find_element(By.CSS_SELECTOR, "h1.title").text
            channel_name = self.browser.find_element(By.CSS_SELECTOR, "ytd-channel-name a").text
            view_count = self.browser.find_element(By.CSS_SELECTOR, "span.view-count").text
            
            # Store video info silently
            self.current_video = {
                'url': current_url,
                'title': video_title,
                'channel': channel_name,
                'views': view_count,
                'timestamp': datetime.now()
            }
            
            # Add to watch history silently
            self.learning_data['watch_history'].append(self.current_video)
            
            return True, "video analyzed"
            
        except Exception as e:
            return False, "analysis failed"

    def _full_cleanup(self):
        """Perform a full cleanup of all resources."""
        # Implementation of _full_cleanup method
        pass 

    def _control_spotify(self, action: str, prompt: str = None) -> tuple[bool, str]:
        """Control Spotify actions."""
        try:
            if action == 'open':
                if not self.system.is_process_running("Spotify.exe"):
                    self._use_run_dialog("spotify")
                    return True, "Spotify is gestart"
                return True, "Spotify is al geopend"
            elif action == 'pause':
                keyboard.press_and_release('play/pause media')
                return True, "Spotify is gepauzeerd"
            elif action == 'resume':
                keyboard.press_and_release('play/pause media')
                return True, "Spotify is hervat"
            elif action == 'next':
                keyboard.press_and_release('next track')
                return True, "Volgend nummer"
            elif action == 'previous':
                keyboard.press_and_release('previous track')
                return True, "Vorig nummer"
            elif action == 'volume_up':
                keyboard.press_and_release('volume up')
                return True, "Volume verhoogd"
            elif action == 'volume_down':
                keyboard.press_and_release('volume down')
                return True, "Volume verlaagd"
            elif action == 'mute':
                keyboard.press_and_release('volume mute')
                return True, "Geluid gedempt"
            elif action == 'volume_set' and prompt:
                try:
                    # Extract percentage from command
                    match = re.search(r'(\d+)(?:\s*%|\s+procent)?', prompt)
                    if match:
                        target_volume = int(match.group(1))
                        current_volume = self._get_current_volume()
                        
                        # Calculate number of key presses needed
                        if target_volume > current_volume:
                            for _ in range(target_volume - current_volume):
                                keyboard.press_and_release('volume up')
                        else:
                            for _ in range(current_volume - target_volume):
                                keyboard.press_and_release('volume down')
                        return True, f"Volume ingesteld op {target_volume}%"
                except Exception as e:
                    return False, f"Fout bij volume instelling: {str(e)}"
            return False, "Onbekende Spotify actie"
        except Exception as e:
            return False, f"Fout bij Spotify controle: {str(e)}"

    def _get_current_volume(self) -> int:
        """Get current system volume percentage."""
        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            
            # Convert to percentage (0-100)
            return int(volume.GetMasterVolumeLevelScalar() * 100)
        except:
            return 50  # Default to 50% if can't get current volume

    def _speak(self, text: str) -> None:
        """Speak the given text using gTTS"""
        try:
            tts = gTTS(text=text, lang='nl')
            temp_file = "temp_speech.mp3"
            tts.save(temp_file)
            
            # Use Windows Media Player to play the audio
            os.system(f'start wmplayer "{temp_file}"')
            time.sleep(2)  # Wait for audio to finish
            
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            logging.error(f"Error in text-to-speech: {str(e)}")

    def _control_notepad(self, action: str, prompt: str = None) -> tuple[bool, str]:
        """Control Notepad actions."""
        try:
            if action == 'open':
                self._use_run_dialog("notepad")
                return True, "Notepad is geopend"
            elif action == 'new' and prompt:
                # Extract note name from command
                match = re.search(r'(?:nieuw|new|maak|create|schrijf|write|op)\s+(.+)$', prompt)
                if match:
                    note_name = match.group(1).strip()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{note_name}_{timestamp}.txt"
                    filepath = os.path.join(self.notes_dir, filename)
                    
                    # Create note with timestamp
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(f"Notitie: {note_name}\n")
                        f.write(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("-" * 50 + "\n\n")
                    
                    # Open the note
                    self._use_run_dialog(f"notepad {filepath}")
                    return True, f"Notitie '{note_name}' is aangemaakt"
            elif action == 'list':
                notes = [f for f in os.listdir(self.notes_dir) if f.endswith('.txt')]
                if notes:
                    return True, f"Beschikbare notities:\n" + "\n".join(notes)
                return True, "Geen notities gevonden"
            elif action == 'delete' and prompt:
                # Extract note name from command
                match = re.search(r'(?:verwijder|delete|wis|clear)\s+(.+)$', prompt)
                if match:
                    note_name = match.group(1).strip()
                    # Find matching note
                    for filename in os.listdir(self.notes_dir):
                        if filename.startswith(note_name) and filename.endswith('.txt'):
                            os.remove(os.path.join(self.notes_dir, filename))
                            return True, f"Notitie '{note_name}' is verwijderd"
                    return False, f"Notitie '{note_name}' niet gevonden"
            return False, "Onbekende Notepad actie"
        except Exception as e:
            return False, f"Fout bij Notepad controle: {str(e)}"

    def _handle_story(self, action: str) -> tuple[bool, str]:
        """Handle story recording and management."""
        try:
            if action == 'record':
                if not hasattr(self, 'recording') or not self.recording:
                    self.recording = True
                    self.story_text = []
                    self.recording_thread = threading.Thread(target=self._record_story)
                    self.recording_thread.start()
                    return True, "Verhaal opname gestart. Spreek nu je verhaal in..."
                return False, "Er is al een opname bezig"
            
            elif action == 'stop':
                if hasattr(self, 'recording') and self.recording:
                    self.recording = False
                    self.recording_thread.join()
                    return True, "Verhaal opname gestopt"
                return False, "Geen actieve opname"
            
            elif action == 'summarize':
                if hasattr(self, 'story_text') and self.story_text:
                    return self._summarize_story()
                return False, "Geen verhaal beschikbaar om samen te vatten"
            
            elif action == 'list':
                return self._list_stories()
            
            return False, "Ongeldige verhaal actie"
            
        except Exception as e:
            return False, f"Verhaal actie mislukt: {str(e)}"

    def _record_story(self):
        """Record a story using speech recognition."""
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                while self.recording:
                    try:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=10)
                        text = self.recognizer.recognize_google(audio, language="nl-NL")
                        self.story_text.append(text)
                        self.console.print(f"[green]Opgenomen: {text}[/green]")
                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        continue
                    except Exception as e:
                        self.console.print(f"[red]Error in opname: {e}[/red]")
                
                # Save the story
                if self.story_text:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    story_path = os.path.join(self.stories_dir, f"verhaal_{timestamp}.txt")
                    
                    with open(story_path, 'w', encoding='utf-8') as f:
                        f.write(f"Verhaal opgenomen op: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        f.write("\n".join(self.story_text))
                    
                    # Create metadata file
                    meta_path = os.path.join(self.stories_dir, f"verhaal_{timestamp}_meta.json")
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'timestamp': timestamp,
                            'length': len(self.story_text),
                            'path': story_path
                        }, f, indent=2)
                    
                    self.console.print(f"[green]Verhaal opgeslagen in: {story_path}[/green]")
                
        except Exception as e:
            self.console.print(f"[red]Error in verhaal opname: {e}[/red]")

    def _summarize_story(self) -> tuple[bool, str]:
        """Summarize the recorded story."""
        try:
            if not self.summarizer:
                return False, "Summarization model niet beschikbaar"
            
            # Combine all text
            full_text = " ".join(self.story_text)
            
            # Generate summary
            summary = self.summarizer(full_text, max_length=130, min_length=30, do_sample=False)
            
            # Format the summary
            formatted_summary = f"""
Samenvatting van je verhaal:

{summary[0]['summary_text']}

Originele lengte: {len(full_text)} woorden
Samenvatting lengte: {len(summary[0]['summary_text'].split())} woorden
"""
            
            # Save the summary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = os.path.join(self.stories_dir, f"samenvatting_{timestamp}.txt")
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(formatted_summary)
            
            return True, formatted_summary
            
        except Exception as e:
            return False, f"Samenvatting mislukt: {str(e)}"

    def _list_stories(self):
        """List all recorded stories"""
        try:
            if not os.path.exists(self.stories_dir):
                return False, "Geen verhalen gevonden"
            
            stories = []
            for file in os.listdir(self.stories_dir):
                if file.endswith('.txt'):
                    file_path = os.path.join(self.stories_dir, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        word_count = len(content.split())
                        stories.append(f"- {file} ({word_count} woorden)")
            
            if not stories:
                return False, "Geen verhalen gevonden"
            
            story_list = sorted(stories)
            return True, "Beschikbare verhalen:\n" + "\n".join(story_list)
        except Exception as e:
            return False, f"Fout bij het ophalen van verhalen: {str(e)}"

    def _control_browser(self, action: str, prompt: str = None) -> tuple[bool, str]:
        """Control browser actions."""
        try:
            if action == 'open':
                self._use_run_dialog("brave")
                return True, "Brave browser is geopend"
            elif action == 'search' and prompt:
                # Extract search term and engine
                search_match = re.match(r'zoek(?:\s+op\s+(youtube|google|wikipedia))?\s+(.+)$', prompt.lower())
                if search_match:
                    engine = search_match.group(1) or 'google'
                    query = search_match.group(2).strip()
                    
                    # Check if query is a direct website
                    if '.' in query and not query.startswith('www.'):
                        # Add www. if not present
                        if not query.startswith('http'):
                            query = 'www.' + query
                        url = f"https://{query}"
                    else:
                        # Construct search URL based on engine
                        if engine == 'youtube':
                            url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
                        elif engine == 'wikipedia':
                            url = f"https://nl.wikipedia.org/wiki/Special:Search?search={query.replace(' ', '+')}"
                        else:  # google
                            url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
                    
                    # Open URL in Brave
                    self._use_run_dialog(f"brave {url}")
                    return True, f"Openen van '{query}'"
                return False, "Geen zoekterm gevonden"
            elif action == 'download':
                if self.brave_youtube:
                    return self.brave_youtube.download_current_video()
                return False, "YouTube functionaliteit is niet beschikbaar"
            elif action == 'analyze':
                if self.brave_youtube:
                    return self.brave_youtube.analyze_current_video()
                return False, "YouTube functionaliteit is niet beschikbaar"
            elif action == 'accept_cookies':
                if self.brave_youtube:
                    return self.brave_youtube.accept_cookies()
                return False, "YouTube functionaliteit is niet beschikbaar"
            return False, "Onbekende browser actie"
        except Exception as e:
            return False, f"Fout bij browser controle: {str(e)}"

    def _show_help(self) -> tuple[bool, str]:
        """Show available commands."""
        help_text = """
Beschikbare commando's:

Browser:
- "Open Brave" of "Start browser"
- "Zoek [zoekterm]" (bijv. "Zoek het weer in Amsterdam")
- "Download video"
- "Analyseer video"
- "Cookies accepteren"

Spotify:
- "Open Spotify"
- "Spotify pauzeer" of "Spotify pauze"
- "Spotify hervat" of "Spotify doorgaan"
- "Spotify volgende" of "Spotify skip"
- "Spotify vorige" of "Spotify terug"
- "Spotify volume op 50%"
- "Spotify 15% zachter"
- "Spotify 20% luider"
- "Spotify dempen"

Notepad:
- "Open Notepad"
- "Notepad nieuw [naam]"
- "Notepad lijst"
- "Notepad verwijder [naam]"

Verhalen:
- "Verhaal start"
- "Verhaal stop"
- "Verhaal samenvatting"
- "Verhalen lijst"

Spellen:
- "Open Minecraft"
- "Open Fortnite"
- "Open Valorant"

Zeg "help" of "hulp" om deze lijst opnieuw te zien.
"""
        return True, help_text 