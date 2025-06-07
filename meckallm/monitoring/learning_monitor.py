import logging
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass
import time
from pathlib import Path
import json
from .blacklist import BlacklistMonitor, BlacklistConfig
import re

@dataclass
class MonitoringConfig:
    log_file: str = "learning_monitor.log"
    save_interval: int = 300  # 5 minutes
    max_violations: int = 10
    alert_threshold: int = 5

class LearningMonitor:
    """Monitors learning activities and enforces blacklist rules."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.blacklist_monitor = BlacklistMonitor()
        self.violations = 0
        self.monitoring_stats = {
            "total_checks": 0,
            "violations": 0,
            "blocked_content": 0,
            "violation_history": []
        }
        self._setup_logging()
        self._last_save = time.time()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            filename=self.config.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _is_suspicious_url(self, url: str) -> bool:
        """Check if a URL is suspicious using multiple detection methods."""
        # Check for common URL obfuscation techniques
        if re.search(self.blacklist_monitor.config.patterns["obfuscated_urls"], url):
            return True
        
        # Check for IP addresses
        if re.search(self.blacklist_monitor.config.patterns["ip_addresses"], url):
            return True
        
        # Check for shortened URLs
        if re.search(self.blacklist_monitor.config.patterns["shortened_urls"], url):
            return True
        
        # Check for suspicious TLDs
        if re.search(self.blacklist_monitor.config.patterns["suspicious_tlds"], url):
            return True
        
        # Check for common bypass techniques in URL
        bypass_patterns = [
            r"(?i)(unblock|bypass|proxy|vpn|tor|anonym|hide|mask|conceal|evade|circumvent|avoid|dodge|escape|elude)",
            r"(?i)([a-z0-9]+\.(?:com|net|org|io|co|me|info|biz|xyz|online|site|website|web|blog|shop|store|app|dev|tech|ai|cloud|host|server|service|solutions|systems|network|security|protection|defense|guard|shield|wall|barrier|fence|gate|door|lock|key|code|pass|secret|private|hidden|masked|concealed|obscured|veiled|cloaked|disguised|camouflaged|stealth|invisible|ghost|phantom|shadow|dark|black|deep|underground|hidden|secret|private|exclusive|premium|vip|elite|pro|advanced|expert|master|guru|ninja|hacker|cracker|exploiter|bypasser|unblocker|circumventer|evader|avoider|dodger|escaper|eluder))",
            r"(?i)([a-z0-9]+[_-]?(?:unblock|bypass|proxy|vpn|tor|anonym|hide|mask|conceal|evade|circumvent|avoid|dodge|escape|elude)[_-]?[a-z0-9]*)",
            r"(?i)([a-z0-9]+[_-]?(?:com|net|org|io|co|me|info|biz|xyz|online|site|website|web|blog|shop|store|app|dev|tech|ai|cloud|host|server|service|solutions|systems|network|security|protection|defense|guard|shield|wall|barrier|fence|gate|door|lock|key|code|pass|secret|private|hidden|masked|concealed|obscured|veiled|cloaked|disguised|camouflaged|stealth|invisible|ghost|phantom|shadow|dark|black|deep|underground|hidden|secret|private|exclusive|premium|vip|elite|pro|advanced|expert|master|guru|ninja|hacker|cracker|exploiter|bypasser|unblocker|circumventer|evader|avoider|dodger|escaper|eluder)[_-]?[a-z0-9]*)"
        ]
        
        for pattern in bypass_patterns:
            if re.search(pattern, url):
                return True
        
        return False
    
    def _check_content_obfuscation(self, content: str) -> bool:
        """Check for content obfuscation techniques."""
        # Check for common obfuscation patterns
        obfuscation_patterns = [
            r"(?i)([a-z0-9]+[_-]?(?:unblock|bypass|proxy|vpn|tor|anonym|hide|mask|conceal|evade|circumvent|avoid|dodge|escape|elude)[_-]?[a-z0-9]*)",
            r"(?i)([a-z0-9]+[_-]?(?:com|net|org|io|co|me|info|biz|xyz|online|site|website|web|blog|shop|store|app|dev|tech|ai|cloud|host|server|service|solutions|systems|network|security|protection|defense|guard|shield|wall|barrier|fence|gate|door|lock|key|code|pass|secret|private|hidden|masked|concealed|obscured|veiled|cloaked|disguised|camouflaged|stealth|invisible|ghost|phantom|shadow|dark|black|deep|underground|hidden|secret|private|exclusive|premium|vip|elite|pro|advanced|expert|master|guru|ninja|hacker|cracker|exploiter|bypasser|unblocker|circumventer|evader|avoider|dodger|escaper|eluder)[_-]?[a-z0-9]*)",
            r"(?i)([a-z0-9]+[_-]?(?:unblock|bypass|proxy|vpn|tor|anonym|hide|mask|conceal|evade|circumvent|avoid|dodge|escape|elude)[_-]?[a-z0-9]*)",
            r"(?i)([a-z0-9]+[_-]?(?:com|net|org|io|co|me|info|biz|xyz|online|site|website|web|blog|shop|store|app|dev|tech|ai|cloud|host|server|service|solutions|systems|network|security|protection|defense|guard|shield|wall|barrier|fence|gate|door|lock|key|code|pass|secret|private|hidden|masked|concealed|obscured|veiled|cloaked|disguised|camouflaged|stealth|invisible|ghost|phantom|shadow|dark|black|deep|underground|hidden|secret|private|exclusive|premium|vip|elite|pro|advanced|expert|master|guru|ninja|hacker|cracker|exploiter|bypasser|unblocker|circumventer|evader|avoider|dodger|escaper|eluder)[_-]?[a-z0-9]*)"
        ]
        
        for pattern in obfuscation_patterns:
            if re.search(pattern, content):
                return True
        
        return False
    
    def monitor_content(self, content: str, source: str) -> Dict[str, Any]:
        """Monitor content for blacklist violations with enhanced detection."""
        self.monitoring_stats["total_checks"] += 1
        
        # Check for suspicious URLs in content
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        for url in urls:
            if self._is_suspicious_url(url):
                self._log_violation("suspicious_url", f"Found suspicious URL: {url}")
                return self._create_blocked_response("suspicious_url", [f"Found suspicious URL: {url}"])
        
        # Check for content obfuscation
        if self._check_content_obfuscation(content):
            self._log_violation("content_obfuscation", "Detected content obfuscation")
            return self._create_blocked_response("content_obfuscation", ["Detected content obfuscation"])
        
        # Check against blacklist
        if not self.blacklist_monitor.is_allowed(content, source):
            violations = self.blacklist_monitor.get_violations(content, source)
            self._log_violation("blacklist", violations)
            return self._create_blocked_response("blacklist", violations)
        
        return {"allowed": True, "violations": {}}
    
    def _create_blocked_response(self, category: str, violations: List[str]) -> Dict[str, Any]:
        """Create a blocked response with violations."""
        self.monitoring_stats["violations"] += 1
        self.monitoring_stats["blocked_content"] += 1
        self.monitoring_stats["violation_history"].append({
            "timestamp": time.time(),
            "category": category,
            "violations": violations
        })
        return {"allowed": False, "violations": {category: violations}}
    
    def _log_violation(self, category: str, violations: Union[str, List[str]]):
        """Log a violation with enhanced details."""
        if isinstance(violations, str):
            violations = [violations]
        
        for violation in violations:
            self.logger.warning(f"Violation detected - Category: {category}, Details: {violation}")
            
            if self.violations >= self.config.max_violations:
                self._send_alert(category, violations)
    
    def _send_alert(self, category: str, violations: List[str]):
        """Send an alert for multiple violations."""
        alert_msg = f"ALERT: Multiple violations detected!\nCategory: {category}\nViolations: {', '.join(violations)}"
        self.logger.error(alert_msg)
        print(f"\n[bold red]{alert_msg}[/bold red]")
    
    def _save_stats(self):
        """Save monitoring statistics to file."""
        current_time = time.time()
        if current_time - self._last_save >= self.config.save_interval:
            stats_file = Path("monitoring_stats.json")
            with open(stats_file, "w") as f:
                json.dump(self.monitoring_stats, f, indent=2)
            self._last_save = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        self._save_stats()
        return self.monitoring_stats
    
    def reset_stats(self):
        """Reset monitoring statistics."""
        self.monitoring_stats = {
            "total_checks": 0,
            "violations": 0,
            "blocked_content": 0,
            "violation_history": []
        }
        self._save_stats()
    
    def add_to_blacklist(self, category: str, items: Union[str, List[str]]):
        """Add items to the blacklist."""
        self.blacklist_monitor.add_to_blacklist(category, items)
    
    def remove_from_blacklist(self, category: str, items: Union[str, List[str]]):
        """Remove items from the blacklist."""
        self.blacklist_monitor.remove_from_blacklist(category, items)
    
    def get_blacklist_summary(self) -> Dict[str, int]:
        """Get a summary of blacklist items by category."""
        return self.blacklist_monitor.get_blacklist_summary() 