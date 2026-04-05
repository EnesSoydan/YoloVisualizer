"""
Konusma Hafizasi
================
Multi-turn konusmalarda baglam korumak icin hafiza yonetimi.
"""

import json
import os
from datetime import datetime


class ConversationMemory:
    """Konusma gecmisini yonetir."""

    def __init__(self, max_turns=20):
        self.max_turns = max_turns
        self.messages = []
        self.session_start = datetime.now().isoformat()

    def add(self, role, content):
        """Mesaj ekle (role: 'user' veya 'assistant')."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        # Max turn asildiysa eski mesajlari kirp
        if len(self.messages) > self.max_turns * 2:
            # Ilk 2 mesaji (baglam icin) ve son max_turns*2 mesaji tut
            self.messages = self.messages[:2] + self.messages[-(self.max_turns * 2 - 2):]

    def get_messages(self):
        """Ollama chat formatinda mesajlari dondur."""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def get_context_summary(self):
        """Onceki konusmanin kisa ozetini dondur."""
        if not self.messages:
            return ""
        topics = []
        for m in self.messages:
            if m["role"] == "user" and len(m["content"]) > 10:
                topics.append(m["content"][:100])
        if not topics:
            return ""
        return "Onceki konusma konulari: " + " | ".join(topics[-5:])

    def clear(self):
        """Hafizayi temizle."""
        self.messages = []
        self.session_start = datetime.now().isoformat()

    def save(self, path):
        """Oturumu JSON olarak kaydet."""
        data = {
            "session_start": self.session_start,
            "messages": self.messages
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        """Onceki oturumu yukle."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.session_start = data.get("session_start", self.session_start)
            self.messages = data.get("messages", [])
            return True
        except (json.JSONDecodeError, KeyError):
            return False

    @property
    def turn_count(self):
        return len([m for m in self.messages if m["role"] == "user"])
