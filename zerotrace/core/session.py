"""
ZeroTrace AI - Persistent Chat Session & Continuity Engine.
Stores multi-turn conversations, tracks cumulative green savings per session,
and enables resuming past chats seamlessly.
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


SESSIONS_DIR = Path.home() / ".zerotrace" / "sessions"


class SessionManager:
    """Manages persistent chat sessions stored as JSON files on disk."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or SESSIONS_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def create_session(
        self,
        provider: str = "antigravity",
        model: str = "gemini-3.8-flash-high",
        agy_conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates and persists a new chat session."""
        now = datetime.now()
        session_id = f"{now.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        session = {
            "session_id": session_id,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "provider": provider,
            "model": model,
            "agy_conversation_id": agy_conversation_id or str(uuid.uuid4()),
            "messages": [],
            "total_tokens_saved": 0,
            "total_energy_saved_j": 0.0,
            "total_carbon_saved_g": 0.0,
            "cache_hits": 0,
        }
        self.save_session(session)
        return session

    def save_session(self, session: Dict[str, Any]):
        """Persists session state to JSON file."""
        session["updated_at"] = datetime.now().isoformat()
        file_path = self.storage_dir / f"{session['session_id']}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2, ensure_ascii=False)

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Loads a session by session_id."""
        file_path = self.storage_dir / f"{session_id}.json"
        if not file_path.exists():
            # Check if user provided partial ID prefix
            matches = list(self.storage_dir.glob(f"*{session_id}*.json"))
            if matches:
                file_path = matches[0]
            else:
                return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def get_last_session(self) -> Optional[Dict[str, Any]]:
        """Returns the most recently modified chat session."""
        files = list(self.storage_dir.glob("*.json"))
        if not files:
            return None
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return self.load_session(files[0].stem)

    def list_sessions(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Returns summary list of recent sessions."""
        files = list(self.storage_dir.glob("*.json"))
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        summaries = []
        for p in files[:limit]:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    preview = ""
                    if d.get("messages"):
                        first_user = next((m["content"] for m in d["messages"] if m.get("role") == "user"), "")
                        preview = (first_user[:45] + "...") if len(first_user) > 45 else first_user
                    summaries.append({
                        "session_id": d["session_id"],
                        "created_at": d.get("created_at", ""),
                        "provider": d.get("provider", "unknown"),
                        "model": d.get("model", "unknown"),
                        "messages_count": len(d.get("messages", [])),
                        "tokens_saved": d.get("total_tokens_saved", 0),
                        "preview": preview,
                    })
            except Exception:
                continue
        return summaries

    def append_message(
        self,
        session: Dict[str, Any],
        role: str,
        content: str,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Appends a turn to the session and updates green metrics."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if metrics:
            msg["metrics"] = metrics
            session["total_tokens_saved"] += metrics.get("tokens_saved", 0)
            session["total_energy_saved_j"] += metrics.get("energy_saved_j", 0.0)
            session["total_carbon_saved_g"] += metrics.get("carbon_saved_g", 0.0)
            if metrics.get("cache_hit"):
                session["cache_hits"] += 1

        session["messages"].append(msg)
        self.save_session(session)


session_manager = SessionManager()
