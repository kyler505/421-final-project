"""Run manifest for reproducibility (Methods / offline bundle)."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _optional_pkg_version(name: str) -> str | None:
    try:
        mod = __import__(name)
    except ImportError:
        return None
    return getattr(mod, "__version__", None)


@dataclass
class RunManifest:
    """Serialized next to checkpoints or under report/ for traceability."""

    schema_version: int = 1
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=lambda: platform.platform())
    sklearn_version: str | None = field(default_factory=lambda: _optional_pkg_version("sklearn"))
    torch_version: str | None = field(default_factory=lambda: _optional_pkg_version("torch"))
    transformers_version: str | None = field(default_factory=lambda: _optional_pkg_version("transformers"))
    backend: str = ""
    artifact_kind: str = ""
    pretrained_source: str = ""
    checkpoint_dir: str = ""
    train_path: str = ""
    max_length: int | None = None
    truncation_policy: str = "hf_max_length_tokens"
    random_state: int | None = None
    hyperparams: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> RunManifest:
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})


def save_run_manifest(manifest: RunManifest, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_json_dict(), indent=2), encoding="utf-8")


def load_run_manifest(path: str | Path) -> RunManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RunManifest.from_json_dict(data)
