"""Centralized utilities for cross-encoders experiments.

Provides:
- Centralized logging configuration (`setup_logging`).
- Path and identity anonymization for public Hugging Face exports (`anonymize_experimaestro_json`).
- Predecessor task hydration via Experimaestro's content-addressing (`hydrate_task_stubs`).
- Export sanitizer pipeline (`sanitize_and_hydrate_model_export`).
- Standalone DAG rendering and inspection CLI (`render_model_dag`).
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Centralized Logging
# ==============================================================================


def setup_logging(level=logging.INFO, format: Optional[str] = None) -> None:
    """Sets up centralized logging for the project and filters out noisy library logs."""
    if format:
        logging.basicConfig(level=level, format=format)
    else:
        logging.basicConfig(level=level)

    # Filter out noisy third-party library logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)


# ==============================================================================
# 2. Path & Identity Anonymization
# ==============================================================================


def anonymize_path_str(path_str: str, user: Optional[str] = None) -> str:
    """Anonymizes cluster user paths, home directories, and scratch paths in a string.

    Redaction rules:
    - Datamaestro storage roots: `.../datamaestro/data/` -> `<datamaestro>/data/`
    - Cluster lustre jobs: `.../jobs/<task>/<hash>` -> `<workspace>/jobs/<task>/<hash>`
    - Jean Zay cluster homes: `/(linkhome|gpfshome|gpfswork)/rech/<project>/<user>` -> `<user_home>`
    - Standard user homes: `/(home|Users)/<user>` -> `<user_home>`
    - Explicit username redaction: `$USER`, `$LOGNAME`, and default JZ user `uqo57sk` -> `<user>`
    """
    s = path_str

    # 1. Datamaestro storage
    s = re.sub(r".*?/datamaestro/data/", "<datamaestro>/data/", s)

    # 2. Cluster lustre jobs paths (preserve relative task and identifier)
    s = re.sub(r".*?/jobs/([^/]+/[0-9a-f]{64})", r"<workspace>/jobs/\1", s)

    # 3. Jean Zay cluster home directories
    s = re.sub(r"/(?:linkhome|gpfshome|gpfswork)/rech/[^/]+/[^/]+", "<user_home>", s)

    # 4. Generic user homes (/home/username or /Users/username)
    s = re.sub(r"/(?:home|Users)/[^/]+", "<user_home>", s)

    # 5. Redact explicit usernames if present anywhere in the string
    usernames: Set[str] = {
        u
        for u in [
            user,
            os.environ.get("USER"),
            os.environ.get("LOGNAME"),
            "uqo57sk",
        ]
        if u and len(u) > 1
    }
    for u in usernames:
        s = s.replace(u, "<user>")

    return s


def anonymize_serialized_data(data: Any, user: Optional[str] = None) -> Any:
    """Recursively walks serialized JSON data to anonymize all paths and user strings."""
    if isinstance(data, dict):
        # Handle experimaestro path objects
        if data.get("type") in ("path", "path.serialized") and "value" in data:
            d = dict(data)
            d["value"] = anonymize_path_str(str(data["value"]), user=user)
            return d
        return {k: anonymize_serialized_data(v, user=user) for k, v in data.items()}
    elif isinstance(data, list):
        return [anonymize_serialized_data(el, user=user) for el in data]
    elif isinstance(data, str):
        return anonymize_path_str(data, user=user)
    return data


def anonymize_experimaestro_json(
    file_or_dir_path: Path,
    output_path: Optional[Path] = None,
    user: Optional[str] = None,
) -> Path:
    """Sanitizes an experimaestro.json file in place or writes to output_path.

    Returns:
        The Path to the sanitized file.
    """
    target = (
        file_or_dir_path
        if file_or_dir_path.is_file()
        else file_or_dir_path / "experimaestro.json"
    )
    if not target.exists():
        raise FileNotFoundError(f"experimaestro.json not found at {target}")

    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)

    sanitized = anonymize_serialized_data(data, user=user)
    out = output_path or target
    with open(out, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2)

    logger.info(f"Anonymized paths in {out}")
    return out


# ==============================================================================
# 3. Predecessor Task Hydration via Content-Addressing
# ==============================================================================


def find_job_params_json(
    identifier: str,
    reference_path: Optional[Path] = None,
    workspace_path: Optional[Path] = None,
) -> Optional[Path]:
    """Locates the params.json corresponding to a given job identifier.

    Searches:
    1. Upward from reference_path (looking for an ancestor that contains a `jobs/` directory)
    2. Provided `workspace_path / jobs/*/<identifier>/params.json`
    3. Experimaestro workspaces configured in settings
    4. Environment variables (EXPERIMENTS_DIR)
    """
    candidates: List[Path] = []
    if workspace_path and workspace_path.exists():
        candidates.append(workspace_path)

    # Ascend from reference_path to locate enclosing workspace with a jobs/ directory
    if reference_path and reference_path.exists():
        ref_resolved = reference_path.resolve()
        for parent in [ref_resolved, *ref_resolved.parents]:
            if (parent / "jobs").is_dir():
                candidates.append(parent / "jobs")
                break

    # Try looking in default experimaestro workspace paths
    try:
        from experimaestro.cli import root_config

        for ws in root_config().workspaces.values():
            if hasattr(ws, "path") and ws.path:
                candidates.append(Path(ws.path))
    except Exception as e:
        logger.debug(f"Could not load root_config workspaces: {e}")

    # Standard cluster / local experiment directories
    env_xp = os.environ.get("EXPERIMENTS_DIR")
    if env_xp:
        candidates.append(Path(env_xp))

    for base in candidates:
        jobs_dir = base / "jobs" if (base / "jobs").is_dir() else base
        matches = list(jobs_dir.glob(f"*/{identifier}*/params.json"))
        if matches:
            return matches[0]

    return None


def hydrate_task_stubs(
    experimaestro_json_path: Path,
    workspace_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Hydrates hollow TaskStub records in experimaestro.json from their job params.json.

    Respects Experimaestro's content-addressing:
    - Objects with identical SHA-256 identifiers are deduplicated.
    - Ephemeral integer `id`s from predecessor runs are mapped to non-colliding IDs.
    - Replaces hollow stub records (fields: []) with full definitions.
    """
    target = (
        experimaestro_json_path
        if experimaestro_json_path.is_file()
        else experimaestro_json_path / "experimaestro.json"
    )
    if not target.exists():
        raise FileNotFoundError(f"experimaestro.json not found at {target}")

    with open(target, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list of object dicts, got {type(data)}")

    # Map existing objects by SHA-256 identifier and integer id
    existing_by_ident: Dict[str, Dict[str, Any]] = {
        o["identifier"]: o for o in data if "identifier" in o
    }
    existing_by_id: Dict[int, Dict[str, Any]] = {
        int(o["id"]): o for o in data if "id" in o
    }

    # Find tasks referenced by configurations (e.g. SimpleModuleLoader.task)
    task_refs: Set[int] = {int(o["task"]) for o in data if o.get("task") is not None}

    # Find stubs:
    # 1) Explicit stub: obj.get("stub") is True
    # 2) An object referenced as a task by another config, having empty fields and task is None
    stubs_to_hydrate: List[Tuple[Dict[str, Any], str]] = []
    for obj in data:
        ident = obj.get("identifier")
        fields = obj.get("fields")
        obj_id = int(obj["id"]) if "id" in obj else None

        is_stub = obj.get("stub", False) or (
            fields == {}
            and obj.get("task") is None
            and (
                obj_id in task_refs
                or any(
                    k in str(obj.get("typename", "")).lower()
                    for k in ("learner", "task", "trainer")
                )
                or obj.get("type", "") in ("Learner", "Task")
            )
        )
        if is_stub and ident:
            stubs_to_hydrate.append((obj, ident))

    if not stubs_to_hydrate:
        logger.info("No TaskStubs detected to hydrate.")
        return target

    max_id = max(existing_by_id.keys(), default=1000)
    hydrated_count = 0

    for stub_obj, ident in stubs_to_hydrate:
        params_file = find_job_params_json(
            ident, reference_path=target, workspace_path=workspace_path
        )
        if not params_file or not params_file.exists():
            logger.warning(
                f"Could not locate params.json for TaskStub {ident[:8]}... leaving stub as-is."
            )
            continue

        try:
            with open(params_file, "r", encoding="utf-8") as pf:
                params_data = json.load(pf)

            upstream_objs: List[Dict[str, Any]] = (
                params_data.get("objects", [])
                if isinstance(params_data, dict)
                else params_data
            )

            # Map predecessor integer IDs to new non-colliding IDs (or existing deduplicated IDs)
            id_map: Dict[int, int] = {}
            for u_obj in upstream_objs:
                u_id = int(u_obj["id"])
                u_ident = u_obj.get("identifier")
                if u_ident and u_ident in existing_by_ident:
                    # Content-addressed deduplication: reuse existing object's integer ID
                    id_map[u_id] = int(existing_by_ident[u_ident]["id"])
                else:
                    max_id += 1
                    id_map[u_id] = max_id

            def remap_refs(val: Any) -> Any:
                if isinstance(val, dict):
                    if val.get("type") == "python" and "value" in val:
                        old_val = int(val["value"])
                        return {"type": "python", "value": id_map.get(old_val, old_val)}
                    return {k: remap_refs(v) for k, v in val.items()}
                elif isinstance(val, list):
                    return [remap_refs(item) for item in val]
                return val

            # Update and merge upstream objects
            for u_obj in upstream_objs:
                u_ident = u_obj.get("identifier")
                old_u_id = int(u_obj["id"])
                new_u_id = id_map[old_u_id]

                if u_ident and u_ident in existing_by_ident:
                    # If this is the stub itself, hydrate its fields and metadata!
                    if u_ident == ident:
                        stub_obj["fields"] = remap_refs(u_obj.get("fields", {}))
                        if "task" in u_obj and u_obj["task"] is not None:
                            stub_obj["task"] = id_map.get(
                                int(u_obj["task"]), int(u_obj["task"])
                            )
                        if "init-tasks" in u_obj:
                            stub_obj["init-tasks"] = [
                                id_map.get(int(t), int(t))
                                for t in u_obj.get("init-tasks", [])
                            ]
                        # Remove stub marker if present
                        stub_obj.pop("stub", None)
                        hydrated_count += 1
                    # Already exists in graph, skip adding duplicate
                    continue

                # New upstream dependency: remap its ID and references and add to objects list
                new_obj = dict(u_obj)
                new_obj["id"] = new_u_id
                if "task" in new_obj and new_obj["task"] is not None:
                    new_obj["task"] = id_map.get(
                        int(new_obj["task"]), int(new_obj["task"])
                    )
                if "init-tasks" in new_obj:
                    new_obj["init-tasks"] = [
                        id_map.get(int(t), int(t))
                        for t in new_obj.get("init-tasks", [])
                    ]
                new_obj["fields"] = remap_refs(new_obj.get("fields", {}))

                data.append(new_obj)
                if u_ident:
                    existing_by_ident[u_ident] = new_obj
                existing_by_id[new_u_id] = new_obj

        except Exception as e:
            logger.error(
                f"Failed to hydrate task stub {ident[:8]} from {params_file}: {e}",
                exc_info=True,
            )

    out = output_path or target
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info(
        f"Hydrated {hydrated_count} TaskStub(s) into {out}. Total objects: {len(data)}"
    )
    return out


def sanitize_and_hydrate_model_export(
    model_dir: Path, workspace_path: Optional[Path] = None
) -> Path:
    """One-stop export hook: hydrates predecessor stubs and anonymizes paths.

    Called automatically upon model export to ensure public artifacts are complete
    and contain zero private cluster paths or usernames.
    """
    model_dir = Path(model_dir)
    exp_json = model_dir / "experimaestro.json"
    if not exp_json.exists():
        logger.warning(
            f"No experimaestro.json found in {model_dir}; skipping sanitization."
        )
        return model_dir

    # 1. Hydrate any predecessor TaskStubs (if params.json is accessible)
    hydrate_task_stubs(exp_json, workspace_path=workspace_path)

    # 2. Anonymize all paths and usernames
    anonymize_experimaestro_json(exp_json)

    return exp_json


# ==============================================================================
# 4. Standalone DAG Rendering & CLI
# ==============================================================================


def render_model_dag(
    model_path: Path,
    html_path: Optional[Path] = None,
    open_browser: bool = False,
    tree: bool = False,
) -> None:
    """Builds and renders a standalone DAG from an exported experimaestro.json.

    - Automatically defaults all task statuses to 'done' (green).
    - Preserves terminal deliverable configs (e.g. SimpleModuleLoader) at the final stage.
    - Extracts scalar parameters into node tags.
    """
    from experimaestro.ui.graph import (
        NodeKind,
        build_experiment_graph,
        standalone_html,
    )

    target = model_path if model_path.is_file() else model_path / "experimaestro.json"
    if not target.exists():
        raise FileNotFoundError(f"experimaestro.json not found at {target}")

    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_name = model_path.parent.name if model_path.is_file() else model_path.name
    graph = build_experiment_graph(model_name, "snapshot", data, {}, set())

    # 1. Find sink configs (deliverables that no other object depends on)
    dependents: Dict[str, List[str]] = {n: [] for n in graph.nodes}
    for n, node in graph.nodes.items():
        for d in node.depends_on:
            if d in dependents:
                dependents[d].append(n)

    sink_configs = [
        n
        for n, deps in dependents.items()
        if not deps and graph.nodes[n].kind == NodeKind.CONFIG
    ]
    for sink_id in sink_configs:
        graph.nodes[sink_id].kind = NodeKind.ACTION

    # 2. Extract scalar task parameters into tags
    for obj in data:
        ident = obj.get("identifier")
        if ident and ident in graph.nodes:
            node = graph.nodes[ident]
            if not node.tags:
                fields = obj.get("fields", {})
                scalars = {
                    k: v
                    for k, v in fields.items()
                    if isinstance(v, (int, float, str, bool)) and not k.startswith("_")
                }
                node.tags = scalars

    # 3. Mark all tasks as 'done'
    statuses = {nid: "done" for nid in graph.nodes}

    if tree:
        view = graph.view(tasks_only=True, stack=True)
        for depth, stage in enumerate(graph.stages(view)):
            print(f"\033[96mStage {depth}\033[0m")
            for vn in stage:
                node = graph.nodes[vn.id]
                count = f" ×{len(vn.members)}" if len(vn.members) > 1 else ""
                tags = ", ".join(f"{k}={v}" for k, v in (vn.tags or node.tags).items())
                print(
                    f"  {node.name}{count} \033[92m[done]\033[0m"
                    + (f"  ({tags})" if tags else "")
                )
        return

    html_content = standalone_html(graph, statuses)
    out = html_path or Path("/tmp") / f"dag_{model_name}.html"
    out.write_text(html_content, encoding="utf-8")
    print(f"\033[92mGraph saved to {out}\033[0m")

    if open_browser:
        import webbrowser

        webbrowser.open(out.as_uri())


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimaestro project utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # dag command
    dag_parser = subparsers.add_parser(
        "dag", help="Render dependency DAG for a model export"
    )
    dag_parser.add_argument(
        "path", type=Path, help="Path to model directory or experimaestro.json"
    )
    dag_parser.add_argument(
        "--html", type=Path, default=None, help="Save standalone HTML to path"
    )
    dag_parser.add_argument(
        "--open", action="store_true", help="Open HTML in default web browser"
    )
    dag_parser.add_argument(
        "--tree", action="store_true", help="Print stage tree to terminal"
    )

    # anonymize command
    anon_parser = subparsers.add_parser(
        "anonymize", help="Anonymize paths and usernames in experimaestro.json"
    )
    anon_parser.add_argument(
        "path", type=Path, help="Path to model directory or experimaestro.json"
    )
    anon_parser.add_argument(
        "--out", type=Path, default=None, help="Output path (default: in-place)"
    )

    args = parser.parse_args()

    if args.command == "dag":
        render_model_dag(
            args.path, html_path=args.html, open_browser=args.open, tree=args.tree
        )
    elif args.command == "anonymize":
        anonymize_experimaestro_json(args.path, output_path=args.out)


if __name__ == "__main__":
    main()
