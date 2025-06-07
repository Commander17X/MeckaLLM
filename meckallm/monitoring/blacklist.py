import json
import logging
from pathlib import Path
from typing import List, Set, Dict, Optional, Union
from dataclasses import dataclass, field
import re
import os

@dataclass
class BlacklistConfig:
    """Configuration for blacklist monitoring."""
    blacklist_file: str = "blacklist.json"
    categories: List[str] = field(default_factory=lambda: [
        "malware",
        "hacking",
        "piracy",
        "sensitive_data",
        "adult_content",
        "political_content",
        "financial_content",
        "hate_speech",
        "violence",
        "illegal_activities",
        "suspicious_websites",
        "proxy_sites",
        "vpn_sites",
        "bypass_sites"
    ])
    patterns: Dict[str, str] = field(default_factory=lambda: {
        "api_key": r"(?i)(api[_-]?key|apikey)[\s]*[:=][\s]*['\"]?[a-zA-Z0-9]{32,}['\"]?",
        "password": r"(?i)(password|passwd|pwd)[\s]*[:=][\s]*['\"]?[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]{8,}['\"]?",
        "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
        "ssn": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "adult_content": r"(?i)(porn|xxx|adult|nsfw|explicit|mature|18\+|sexual|nude|naked|sex|intimate|erotic|lewd|obscene|vulgar)",
        "political_content": r"(?i)(politics|political|government|election|campaign|party|vote|democrat|republican|liberal|conservative|radical|extremist|propaganda|ideology)",
        "financial_content": r"(?i)(stock|market|trading|investment|finance|banking|currency|crypto|bitcoin|ethereum|wallet|transaction|scam|fraud|pyramid|scheme)",
        "hate_speech": r"(?i)(hate|racist|sexist|discriminatory|bigot|prejudice|bias|supremacy|extremism|intolerance)",
        "violence": r"(?i)(violence|weapon|gun|bomb|terror|attack|kill|murder|assault|threat|harm|danger|weapon|explosive)",
        "illegal_activities": r"(?i)(illegal|crime|criminal|fraud|scam|hack|exploit|crack|pirate|steal|cheat|deceive|forge|counterfeit)",
        "suspicious_websites": r"(?i)(unblock|bypass|proxy|vpn|tor|anonym|hide|mask|conceal|evade|circumvent|avoid|dodge|escape|elude)",
        "proxy_sites": r"(?i)(proxy|vpn|tor|anonym|hide|mask|conceal|evade|circumvent|avoid|dodge|escape|elude|bypass|unblock)",
        "vpn_sites": r"(?i)(vpn|virtual[_-]?private[_-]?network|tunnel|encrypt|secure|private|anonymous|hide|mask|conceal)",
        "bypass_sites": r"(?i)(bypass|unblock|circumvent|evade|avoid|dodge|escape|elude|proxy|vpn|tor|anonym|hide|mask|conceal)",
        "obfuscated_urls": r"(?i)([a-z0-9]+\.(?:com|net|org|io|co|me|info|biz|xyz|online|site|website|web|blog|shop|store|app|dev|tech|ai|cloud|host|server|service|solutions|systems|network|security|protection|defense|guard|shield|wall|barrier|fence|gate|door|lock|key|code|pass|secret|private|hidden|masked|concealed|obscured|veiled|cloaked|disguised|camouflaged|stealth|invisible|ghost|phantom|shadow|dark|black|deep|underground|hidden|secret|private|exclusive|premium|vip|elite|pro|advanced|expert|master|guru|ninja|hacker|cracker|exploiter|bypasser|unblocker|circumventer|evader|avoider|dodger|escaper|eluder))",
        "ip_addresses": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "shortened_urls": r"(?i)(bit\.ly|t\.co|goo\.gl|tinyurl\.com|is\.gd|cli\.gs|ow\.ly|yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|tr\.im|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|doiop\.com|twitthis\.com|ht\.ly|alturl\.com|tiny\.pl|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net)",
        "suspicious_tlds": r"(?i)\.(xyz|top|loan|work|click|bid|win|download|stream|online|site|website|web|blog|shop|store|app|dev|tech|ai|cloud|host|server|service|solutions|systems|network|security|protection|defense|guard|shield|wall|barrier|fence|gate|door|lock|key|code|pass|secret|private|hidden|masked|concealed|obscured|veiled|cloaked|disguised|camouflaged|stealth|invisible|ghost|phantom|shadow|dark|black|deep|underground|hidden|secret|private|exclusive|premium|vip|elite|pro|advanced|expert|master|guru|ninja|hacker|cracker|exploiter|bypasser|unblocker|circumventer|evader|avoider|dodger|escaper|eluder)$"
    })
    domains: List[str] = field(default_factory=lambda: [
        "malicious.com",
        "hackers.org",
        "piracy.net",
        "cracked.software",
        "adult-content.com",
        "porn-site.com",
        "political-extremist.org",
        "hate-speech.net",
        "illegal-market.com",
        "scam-site.net",
        "unblock-site.com",
        "bypass-proxy.net",
        "vpn-service.org",
        "tor-network.com",
        "anonymous-browser.net",
        "hide-ip.org",
        "mask-traffic.com",
        "conceal-identity.net",
        "evade-detection.org",
        "circumvent-block.com",
        "avoid-restriction.net",
        "dodge-filter.org",
        "escape-monitor.com",
        "elude-tracking.net"
    ])
    file_types: List[str] = field(default_factory=lambda: [
        ".exe", ".dll", ".bat", ".sh", ".pyc", ".bin", ".iso", ".img",
        ".torrent", ".crack", ".keygen", ".patch", ".hack", ".exploit",
        ".malware", ".virus", ".trojan", ".backdoor", ".rootkit",
        ".spyware", ".adware", ".ransomware", ".botnet", ".worm",
        ".adult", ".nsfw", ".political", ".financial", ".crypto", ".wallet",
        ".proxy", ".vpn", ".tor", ".anonym", ".hide", ".mask", ".conceal",
        ".evade", ".circumvent", ".avoid", ".dodge", ".escape", ".elude",
        ".bypass", ".unblock", ".circumvent", ".evade", ".avoid", ".dodge",
        ".escape", ".elude", ".proxy", ".vpn", ".tor", ".anonym", ".hide",
        ".mask", ".conceal", ".obscure", ".veil", ".cloak", ".disguise",
        ".camouflage", ".stealth", ".invisible", ".ghost", ".phantom",
        ".shadow", ".dark", ".black", ".deep", ".underground", ".hidden",
        ".secret", ".private", ".exclusive", ".premium", ".vip", ".elite",
        ".pro", ".advanced", ".expert", ".master", ".guru", ".ninja",
        ".hacker", ".cracker", ".exploiter", ".bypasser", ".unblocker",
        ".circumventer", ".evader", ".avoider", ".dodger", ".escaper",
        ".eluder"
    ])

class BlacklistMonitor:
    """Monitors and enforces blacklist rules."""
    
    def __init__(self, config: Optional[BlacklistConfig] = None):
        self.config = config or BlacklistConfig()
        self.logger = logging.getLogger(__name__)
        self.blacklist: Dict[str, Set[str]] = {
            "categories": set(self.config.categories),
            "patterns": set(self.config.patterns.keys()),
            "domains": set(self.config.domains),
            "file_types": set(self.config.file_types)
        }
        self._load_blacklist()
        self.compiled_patterns = [re.compile(pattern) for pattern in self.blacklist["patterns"]]

    def _load_blacklist(self):
        """Load blacklist from file if it exists."""
        try:
            if os.path.exists(self.config.blacklist_file):
                with open(self.config.blacklist_file, 'r') as f:
                    data = json.load(f)
                    for category, items in data.items():
                        if category in self.blacklist:
                            self.blacklist[category].update(items)
                self.logger.info("Blacklist loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading blacklist: {str(e)}")

    def _save_blacklist(self):
        """Save blacklist to file."""
        try:
            with open(self.config.blacklist_file, 'w') as f:
                json.dump({k: list(v) for k, v in self.blacklist.items()}, f, indent=2)
            self.logger.info("Blacklist saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving blacklist: {str(e)}")

    def add_to_blacklist(self, category: str, items: Union[str, List[str]]):
        """Add items to blacklist."""
        if isinstance(items, str):
            items = [items]
        if category in self.blacklist:
            self.blacklist[category].update(items)
            self._save_blacklist()
            self.logger.info(f"Added {len(items)} items to {category} blacklist")
        else:
            self.logger.error(f"Invalid category: {category}")

    def remove_from_blacklist(self, category: str, items: Union[str, List[str]]):
        """Remove items from blacklist."""
        if isinstance(items, str):
            items = [items]
        if category in self.blacklist:
            self.blacklist[category].difference_update(items)
            self._save_blacklist()
            self.logger.info(f"Removed {len(items)} items from {category} blacklist")
        else:
            self.logger.error(f"Invalid category: {category}")

    def check_content(self, content: str, source: str) -> Dict[str, List[str]]:
        """Check content against blacklist rules."""
        violations = {
            "categories": [],
            "patterns": [],
            "domains": [],
            "file_types": []
        }
        
        # Check categories
        for category in self.blacklist["categories"]:
            if category.lower() in content.lower():
                violations["categories"].append(category)
        
        # Check patterns
        for pattern in self.compiled_patterns:
            if pattern.search(content):
                violations["patterns"].append(pattern.pattern)
        
        # Check domains in source
        for domain in self.blacklist["domains"]:
            if domain.lower() in source.lower():
                violations["domains"].append(domain)
        
        # Check file types in source
        for file_type in self.blacklist["file_types"]:
            if source.lower().endswith(file_type.lower()):
                violations["file_types"].append(file_type)
        
        return violations

    def is_allowed(self, content: str, source: str) -> bool:
        """Check if content is allowed based on blacklist rules."""
        violations = self.check_content(content, source)
        return not any(violations.values())

    def get_violations(self, content: str, source: str) -> Dict[str, List[str]]:
        """Get detailed violations for content."""
        return self.check_content(content, source)

    def monitor_learning(self, content: str) -> bool:
        """Monitor content for learning purposes"""
        if not self.is_allowed(content, ""):
            self.logger.warning("Content contains blacklisted items, preventing learning")
            return False
        return True

    def get_blacklist_summary(self) -> Dict[str, int]:
        """Get summary of blacklist items by category."""
        return {category: len(items) for category, items in self.blacklist.items()} 