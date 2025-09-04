from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set


# ---------------------- types ----------------------

@dataclass
class NodeInfo:
    unique_id: str
    name: str
    resource_type: str
    fqn: List[str]
    original_file_path: Optional[str]
    completed_epoch: Optional[int]
    started_epoch: Optional[int]
    status: Optional[str]
    owner: str
    execution_time: float = 0.0
    dependency_count: int = 0


@dataclass
class ParsedArtifacts:
    nodes: Dict[str, NodeInfo]
    edges: List[Tuple[str, str]]  # (parent -> child)
    dbt_version: str
    is_fusion: bool
    fusion_skipped: Set[str]


# ---------------------- helpers ----------------------

def parse_iso8601(ts: str) -> datetime:
    if ts is None:
        raise ValueError("empty timestamp")
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def to_epoch(ts: Optional[str]) -> Optional[int]:
    if not ts:
        return None
    try:
        return int(parse_iso8601(ts).timestamp())
    except Exception:
        return None


def owner_from_path(original_file_path: Optional[str], fqn: List[str]) -> str:
    path = original_file_path or "/".join(fqn)
    parts = Path(path).as_posix().split("/")
    if "models" in parts:
        i = parts.index("models")
        if i + 1 < len(parts):
            return parts[i + 1]
    if len(fqn) >= 2:
        return fqn[1]
    return "dbt"


def detect_dbt_version(run_results_path: Path) -> Tuple[str, bool]:
    with open(run_results_path, 'r', encoding='utf-8') as f:
        rr = json.load(f)
    version = rr.get("metadata", {}).get("dbt_version", "unknown")
    # Fusion if version string explicitly mentions it, or 2025.8+ (Fusion release)
    is_fusion = False
    vlow = str(version).lower()
    if "fusion" in vlow:
        is_fusion = True
    else:
        try:
            if vlow.startswith("2025."):
                # Extract minor version before '+' or extra suffix
                rest = vlow.split("+", 1)[0]  # e.g., '2025.8.26'
                parts = rest.split(".")
                minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                is_fusion = minor >= 8
        except Exception:
            is_fusion = False
    return version, is_fusion


def parse_fusion_debug_log(debug_log_path: Optional[Path]) -> Set[str]:
    reused: Set[str] = set()
    if not debug_log_path or not debug_log_path.exists():
        return reused
    try:
        for line in debug_log_path.read_text().splitlines():
            if "Reused" in line and "model " in line:
                parts = line.strip().split("model ")
                if len(parts) > 1:
                    model_name = parts[1].split(" ")[0]
                    reused.add(f"model.{model_name}")
    except Exception:
        # best effort only
        pass
    return reused


# ---------------------- parser ----------------------

def load_manifest_and_results(
    manifest_path: Path,
    run_results_path: Path,
    include_types: List[str],
    debug_log_path: Optional[Path] = None,
) -> ParsedArtifacts:
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    with open(run_results_path, 'r', encoding='utf-8') as f:
        rr = json.load(f)

    dbt_version, is_fusion = detect_dbt_version(run_results_path)
    fusion_skipped_debug = parse_fusion_debug_log(debug_log_path) if is_fusion else set()

    timings: Dict[str, Tuple[Optional[int], Optional[int], Optional[str]]] = {}
    execution_times: Dict[str, float] = {}
    executed_models = set()

    for r in rr.get("results", []):
        started = r.get("started_at")
        completed = r.get("completed_at")
        status = r.get("status")
        unique_id = r.get("unique_id")
        execution_time = r.get("execution_time", 0.0)

        executed_models.add(unique_id)
        execution_times[unique_id] = float(execution_time)

        if (not started or not completed) and r.get("timing"):
            for t in r["timing"]:
                if t.get("name") == "execute":
                    started = started or t.get("started_at")
                    completed = completed or t.get("completed_at")
        timings[unique_id] = (to_epoch(started), to_epoch(completed), status)

    nodes: Dict[str, NodeInfo] = {}

    for section in ("nodes", "sources", "exposures", "metrics", "semantic_models"):
        for uid, n in manifest.get(section, {}).items():
            rtype = (n.get("resource_type") or "").lower()
            if rtype not in include_types:
                continue

            name = n.get("name") or uid.split(".")[-1]
            fqn = n.get("fqn", [])
            ofp = n.get("original_file_path")
            owner = owner_from_path(ofp, fqn)

            depends_on = n.get("depends_on", {})
            dependency_count = len(depends_on.get("nodes", [])) + len(depends_on.get("macros", []))

            started_epoch, completed_epoch, status = None, None, None
            if uid in timings:
                started_epoch, completed_epoch, status = timings[uid]
                # if extremely fast, treat as skipped signal in fusion
                if is_fusion and execution_times.get(uid, 0.0) < 0.01:
                    status = "fusion_skipped"
            elif is_fusion:
                status = "fusion_skipped"

            nodes[uid] = NodeInfo(
                unique_id=uid,
                name=name,
                resource_type=rtype,
                fqn=fqn,
                original_file_path=ofp,
                completed_epoch=completed_epoch,
                started_epoch=started_epoch,
                status=status,
                owner=owner,
                execution_time=execution_times.get(uid, 0.0),
                dependency_count=dependency_count,
            )

    edges: List[Tuple[str, str]] = []
    child_map = manifest.get("child_map") or {}
    if child_map:
        for parent_uid, children in child_map.items():
            if parent_uid not in nodes:
                continue
            for c in children:
                if c in nodes:
                    edges.append((parent_uid, c))
    else:
        parent_map = manifest.get("parent_map") or {}
        for uid, parents in parent_map.items():
            if uid not in nodes:
                continue
            for p in parents:
                if p in nodes:
                    edges.append((p, uid))

    # Prefer debug-based list if present
    fusion_skipped = fusion_skipped_debug

    return ParsedArtifacts(
        nodes=nodes,
        edges=edges,
        dbt_version=dbt_version,
        is_fusion=is_fusion,
        fusion_skipped=fusion_skipped,
    )
