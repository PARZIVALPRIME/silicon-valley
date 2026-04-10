"""FastAPI server for SiliconMind OpenEnv environment.

Uses openenv.core.env_server.create_app for OpenEnv compliance.
Exposes /reset, /step, /state endpoints + root and health checks.
"""

from __future__ import annotations

import uvicorn
from openenv.core.env_server import create_app

from env.environment import SiliconMindEnvironment
from models import SiliconMindAction, SiliconMindObservation

# Shared stateful environment instance
_shared_env = SiliconMindEnvironment()

app = create_app(
    lambda: _shared_env,
    SiliconMindAction,
    SiliconMindObservation,
    env_name="silicon_mind",
)


@app.get("/")
def root():
    """Root endpoint — environment info."""
    return {
        "name": "SiliconMind",
        "version": "0.1.0",
        "description": "Memory Subsystem Intelligence Gym",
        "status": "running",
        "env": "silicon_mind",
        "tasks": _shared_env.available_tasks,
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }


@app.get("/health")
def health():
    """Health check for Docker / HF Spaces."""
    return {"status": "ok"}


def main():
    """Entry point for `python -m server.app` or pyproject scripts."""
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        workers=1,
    )


if __name__ == "__main__":
    main()
