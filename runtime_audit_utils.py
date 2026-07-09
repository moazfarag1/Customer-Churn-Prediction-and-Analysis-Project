from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


MANIFEST_PATH = Path("runtime_tests/regeneration_manifest.json")
BACKUP_ROOT = Path("runtime_tests/artifact_backups")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backup_if_overwriting(path: Path, project_root: Path, label: str = "artifact") -> None:
    path = Path(path)
    if not path.exists():
        return

    try:
        relative_path = path.relative_to(project_root)
    except ValueError:
        relative_path = Path(path.name)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = project_root / BACKUP_ROOT / relative_path.parent / f"{path.stem}.{timestamp}{path.suffix}"
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_path)
    print(f"Existing {label} backed up before overwrite: {backup_path}")


def record_regeneration_step(
    project_root: Path,
    step_name: str,
    source_notebook: str,
    produced_paths: list[Path],
) -> None:
    manifest_path = project_root / MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"version": 1, "steps": {}}

    notebook_path = project_root / source_notebook
    artifacts = {}
    for path in produced_paths:
        artifact_path = Path(path)
        try:
            relative_path = str(artifact_path.relative_to(project_root)).replace("\\", "/")
        except ValueError:
            relative_path = str(artifact_path).replace("\\", "/")

        artifacts[relative_path] = {
            "sha256": sha256_file(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
            "mtime_utc": datetime.fromtimestamp(
                artifact_path.stat().st_mtime,
                timezone.utc,
            ).isoformat(),
        }

    manifest["steps"][step_name] = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_notebook": source_notebook.replace("\\", "/"),
        "source_sha256": sha256_file(notebook_path),
        "artifacts": artifacts,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Regeneration manifest updated: {manifest_path}")
