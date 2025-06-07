import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import time
from pathlib import Path
import json
from .blacklist import BlacklistMonitor, BlacklistConfig

@dataclass
class MonitoringConfig:
    log_file: str = "learning_monitor.log"
    save_interval: int = 300  # 5 minutes
    max_violations: int = 10
    alert_threshold: int = 5

class LearningMonitor:
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Initialize blacklist monitor
        self.blacklist = BlacklistMonitor()
        
        # Initialize monitoring state
        self.violations_count = 0
        self.last_save = time.time()
        self.monitoring_stats = {
            "total_checks": 0,
            "violations": 0,
            "blocked_content": 0,
            "last_violation": None,
            "violation_history": []
        }

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            filename=self.config.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def monitor_content(self, content: str, source: str) -> Dict[str, Any]:
        """Monitor content for learning"""
        self.monitoring_stats["total_checks"] += 1
        
        # Check content against blacklist
        violations = self.blacklist.get_violations(content)
        has_violations = any(violations.values())
        
        if has_violations:
            self.violations_count += 1
            self.monitoring_stats["violations"] += 1
            self.monitoring_stats["last_violation"] = {
                "timestamp": time.time(),
                "source": source,
                "violations": violations
            }
            self.monitoring_stats["violation_history"].append(
                self.monitoring_stats["last_violation"]
            )
            
            # Trim violation history if too long
            if len(self.monitoring_stats["violation_history"]) > self.config.max_violations:
                self.monitoring_stats["violation_history"] = \
                    self.monitoring_stats["violation_history"][-self.config.max_violations:]
            
            # Log violation
            self.logger.warning(
                f"Content violation detected from {source}: {violations}"
            )
            
            # Check if we need to alert
            if self.violations_count >= self.config.alert_threshold:
                self._send_alert(violations, source)
        
        # Save monitoring stats periodically
        if time.time() - self.last_save > self.config.save_interval:
            self._save_stats()
        
        return {
            "allowed": not has_violations,
            "violations": violations,
            "violations_count": self.violations_count
        }

    def _send_alert(self, violations: Dict[str, List[str]], source: str):
        """Send alert about violations"""
        alert_msg = f"ALERT: Multiple violations detected!\n"
        alert_msg += f"Source: {source}\n"
        alert_msg += f"Violations: {violations}\n"
        alert_msg += f"Total violations: {self.violations_count}"
        
        self.logger.error(alert_msg)
        # TODO: Implement actual alert mechanism (email, notification, etc.)

    def _save_stats(self):
        """Save monitoring statistics"""
        try:
            stats_file = Path("monitoring_stats.json")
            with open(stats_file, 'w') as f:
                json.dump(self.monitoring_stats, f, indent=2)
            self.last_save = time.time()
            self.logger.info("Monitoring stats saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving monitoring stats: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics"""
        return {
            "monitoring_stats": self.monitoring_stats,
            "blacklist_summary": self.blacklist.get_blacklist_summary(),
            "violations_count": self.violations_count
        }

    def reset_stats(self):
        """Reset monitoring statistics"""
        self.violations_count = 0
        self.monitoring_stats = {
            "total_checks": 0,
            "violations": 0,
            "blocked_content": 0,
            "last_violation": None,
            "violation_history": []
        }
        self._save_stats()
        self.logger.info("Monitoring stats reset")

    def add_to_blacklist(self, category: str, items: List[str]):
        """Add items to blacklist"""
        self.blacklist.add_to_blacklist(category, items)

    def remove_from_blacklist(self, category: str, items: List[str]):
        """Remove items from blacklist"""
        self.blacklist.remove_from_blacklist(category, items) 