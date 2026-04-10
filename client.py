"""SiliconMind OpenEnv client — wraps HTTP calls to the environment server.

Usage:
    from client import SiliconMindClient
    client = SiliconMindClient("http://localhost:7860")
    obs = client.reset(task_id="easy_pattern_scout")
    obs = client.step(SiliconMindAction(type="prefetch", target="1,2"))
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

from models import SiliconMindAction, SiliconMindObservation, SiliconMindState


DEFAULT_URL = os.getenv("SILICON_MIND_URL", "http://localhost:7860")


class SiliconMindClient:
    """HTTP client for the SiliconMind environment."""

    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def reset(
        self,
        task_id: str = "easy_pattern_scout",
        seed: Optional[int] = None,
    ) -> SiliconMindObservation:
        """Reset the environment to a new episode."""
        payload: Dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed

        resp = httpx.post(
            f"{self.base_url}/reset",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if "observation" in data and isinstance(data["observation"], dict):
            obs = data["observation"]
            obs["reward"] = float(data.get("reward") or 0.0)
            obs["done"] = data.get("done", False)
            obs["info"] = data.get("info", {})
            return SiliconMindObservation(**obs)
        return SiliconMindObservation(**data)

    def step(self, action: SiliconMindAction) -> SiliconMindObservation:
        """Take one step with the given action."""
        resp = httpx.post(
            f"{self.base_url}/step",
            json={"action": action.model_dump()},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if "observation" in data and isinstance(data["observation"], dict):
            obs = data["observation"]
            obs["reward"] = float(data.get("reward") or 0.0)
            obs["done"] = data.get("done", False)
            obs["info"] = data.get("info", {})
            return SiliconMindObservation(**obs)
        return SiliconMindObservation(**data)

    def state(self) -> SiliconMindState:
        """Get the current internal state."""
        resp = httpx.get(
            f"{self.base_url}/state",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return SiliconMindState(**resp.json())

    def health(self) -> bool:
        """Check if the server is running."""
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False
