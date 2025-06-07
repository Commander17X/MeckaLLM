import json
import logging
from pathlib import Path
from typing import List, Set, Dict, Optional
from dataclasses import dataclass
import re

@dataclass
class BlacklistConfig:
    blacklist_file: str = "blacklist.json"
    categories: List[str] = None
    patterns: List[str] = None
    domains: List[str] = None
    file_types: List[str] = None

    def __post_init__(self):
        if self.categories is None:
            self.categories = [
                "malware",
                "exploits",
                "hacking",
                "cracking",
                "piracy",
                "illegal_content",
                "sensitive_data",
                "personal_info",
                "financial_data",
                "trade_secrets",
                "proprietary_code",
                "restricted_apis",
                "unauthorized_access",
                "system_bypass",
                "security_breach"
            ]
        if self.patterns is None:
            self.patterns = [
                r"password\s*=\s*['\"].*['\"]",
                r"api_key\s*=\s*['\"].*['\"]",
                r"secret\s*=\s*['\"].*['\"]",
                r"token\s*=\s*['\"].*['\"]",
                r"credit_card\s*=\s*['\"].*['\"]",
                r"ssn\s*=\s*['\"].*['\"]",
                r"private_key\s*=\s*['\"].*['\"]"
            ]
        if self.domains is None:
            self.domains = [
                "malicious.com",
                "cracked-software.com",
                "hack-tools.net",
                "pirate-bay.org"
            ]
        if self.file_types is None:
            self.file_types = [
                ".exe",
                ".dll",
                ".bat",
                ".vbs",
                ".ps1",
                ".sh",
                ".bin"
            ]

class BlacklistMonitor:
    def __init__(self, config: Optional[BlacklistConfig] = None):
        self.config = config or BlacklistConfig()
        self.logger = logging.getLogger(__name__)
        self.blacklist: Dict[str, Set[str]] = {
            "categories": set(self.config.categories),
            "patterns": set(self.config.patterns),
            "domains": set(self.config.domains),
            "file_types": set(self.config.file_types)
        }
        self._load_blacklist()
        self.compiled_patterns = [re.compile(pattern) for pattern in self.blacklist["patterns"]]

    def _load_blacklist(self):
        """Load blacklist from file if it exists"""
        try:
            blacklist_path = Path(self.config.blacklist_file)
            if blacklist_path.exists():
                with open(blacklist_path, 'r') as f:
                    saved_blacklist = json.load(f)
                    for key, values in saved_blacklist.items():
                        self.blacklist[key].update(values)
                self.logger.info("Blacklist loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading blacklist: {str(e)}")

    def _save_blacklist(self):
        """Save current blacklist to file"""
        try:
            with open(self.config.blacklist_file, 'w') as f:
                json.dump({k: list(v) for k, v in self.blacklist.items()}, f, indent=2)
            self.logger.info("Blacklist saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving blacklist: {str(e)}")

    def add_to_blacklist(self, category: str, items: List[str]):
        """Add items to the blacklist"""
        if category in self.blacklist:
            self.blacklist[category].update(items)
            self._save_blacklist()
            self.logger.info(f"Added {len(items)} items to {category} blacklist")
        else:
            self.logger.error(f"Invalid category: {category}")

    def remove_from_blacklist(self, category: str, items: List[str]):
        """Remove items from the blacklist"""
        if category in self.blacklist:
            self.blacklist[category].difference_update(items)
            self._save_blacklist()
            self.logger.info(f"Removed {len(items)} items from {category} blacklist")
        else:
            self.logger.error(f"Invalid category: {category}")

    def check_content(self, content: str) -> Dict[str, List[str]]:
        """Check content against blacklist"""
        violations = {
            "categories": [],
            "patterns": [],
            "domains": [],
            "file_types": []
        }

        # Check categories
        content_lower = content.lower()
        for category in self.blacklist["categories"]:
            if category.lower() in content_lower:
                violations["categories"].append(category)

        # Check patterns
        for pattern in self.compiled_patterns:
            if pattern.search(content):
                violations["patterns"].append(pattern.pattern)

        # Check domains
        for domain in self.blacklist["domains"]:
            if domain in content:
                violations["domains"].append(domain)

        # Check file types
        for file_type in self.blacklist["file_types"]:
            if file_type in content:
                violations["file_types"].append(file_type)

        return violations

    def is_allowed(self, content: str) -> bool:
        """Check if content is allowed"""
        violations = self.check_content(content)
        return not any(violations.values())

    def get_violations(self, content: str) -> Dict[str, List[str]]:
        """Get detailed violations in content"""
        return self.check_content(content)

    def monitor_learning(self, content: str) -> bool:
        """Monitor content for learning purposes"""
        if not self.is_allowed(content):
            self.logger.warning("Content contains blacklisted items, preventing learning")
            return False
        return True

    def get_blacklist_summary(self) -> Dict[str, int]:
        """Get summary of blacklist items"""
        return {k: len(v) for k, v in self.blacklist.items()} 