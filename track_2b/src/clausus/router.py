"""The router decides per task: local, or masked to an external model.

It is plain code reading a rules file. Nothing said in a meeting can change it.
External needs BOTH a rule that allows it AND a configured target; anything
else stays local.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DESTINATIONS = ("local", "external")


@dataclass(frozen=True)
class Routes:
    tasks: dict[str, str] = field(default_factory=dict)

    @classmethod
    def default(cls) -> Routes:
        return cls(tasks={"protocol": "local"})

    @classmethod
    def load(cls, path: str | Path) -> Routes:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        tasks = {str(k): str(v) for k, v in (data.get("tasks") or {}).items()}
        bad = {k: v for k, v in tasks.items() if v not in DESTINATIONS}
        if bad:
            raise ValueError(f"invalid destination(s) in {path}: {bad}; allowed {DESTINATIONS}")
        return cls(tasks=tasks)


@dataclass(frozen=True)
class Decision:
    destination: str
    reason: str


def route(task: str, routes: Routes, external_configured: bool) -> Decision:
    wanted = routes.tasks.get(task, "local")
    if wanted == "external" and not external_configured:
        return Decision("local", f"rule allows external for '{task}', but external target not configured")
    if wanted == "external":
        return Decision("external", f"rule allows external for '{task}'")
    return Decision("local", f"rule keeps '{task}' local")
