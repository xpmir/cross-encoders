"""Unit tests for xp_utils (logging, anonymization, hydration, and DAG helpers)."""

import json
from pathlib import Path
from xp_utils import (
    setup_logging,
    anonymize_path_str,
    anonymize_serialized_data,
    anonymize_experimaestro_json,
    hydrate_task_stubs,
    render_model_dag,
)


def test_setup_logging():
    """Verify that setup_logging configures without raising exceptions."""
    setup_logging()


def test_anonymize_path_str():
    """Verify path anonymization across Jean Zay, Lustre, Datamaestro, and home paths."""
    # Datamaestro storage
    dm_path = "/linkhome/rech/genisi01/uqo57sk/datamaestro/data/ir/co/huggingface/nano-beir/documents/STORE"
    anon_dm = anonymize_path_str(dm_path)
    assert "<datamaestro>/data/ir/co/huggingface/nano-beir/documents/STORE" in anon_dm
    assert "uqo57sk" not in anon_dm
    assert "linkhome" not in anon_dm

    # Cluster lustre jobs
    lustre_job = "/lustre/fsn1/projects/rech/npl/commun/cross_encoders/jobs/xpm_torch.learner.learner/f41f415aeba64bcb1e755c835afb4fec0daea919d5194cade4cfe83d691e34a2/checkpoints"
    anon_job = anonymize_path_str(lustre_job)
    assert anon_job == "<workspace>/jobs/xpm_torch.learner.learner/f41f415aeba64bcb1e755c835afb4fec0daea919d5194cade4cfe83d691e34a2/checkpoints"
    assert "lustre" not in anon_job
    assert "rech/npl" not in anon_job

    # User homes
    jz_home = "/linkhome/rech/genisi01/uqo57sk/scripts/test.sh"
    assert anonymize_path_str(jz_home) == "<user_home>/scripts/test.sh"

    mac_home = "/Users/victor/code/projects/cross-encoders/model.safetensors"
    assert anonymize_path_str(mac_home) == "<user_home>/code/projects/cross-encoders/model.safetensors"

    # Explicit user string
    assert "uqo57sk" not in anonymize_path_str("Hello uqo57sk world")


def test_anonymize_serialized_data(tmp_path: Path):
    """Verify recursive anonymization on serialized experimaestro dictionary."""
    data = [
        {
            "id": 1,
            "type": "BeirDocumentStore",
            "fields": {
                "path": {
                    "type": "path",
                    "value": "/linkhome/rech/genisi01/uqo57sk/datamaestro/data/ir/store",
                },
                "job_dir": "/lustre/fsn1/projects/rech/npl/commun/jobs/task/f41f415aeba64bcb1e755c835afb4fec0daea919d5194cade4cfe83d691e34a2/run.txt",
            },
        }
    ]
    anonymized = anonymize_serialized_data(data)
    assert anonymized[0]["fields"]["path"]["value"] == "<datamaestro>/data/ir/store"
    assert anonymized[0]["fields"]["job_dir"] == "<workspace>/jobs/task/f41f415aeba64bcb1e755c835afb4fec0daea919d5194cade4cfe83d691e34a2/run.txt"

    # Test file-based anonymization
    json_file = tmp_path / "experimaestro.json"
    with open(json_file, "w") as f:
        json.dump(data, f)

    out_file = anonymize_experimaestro_json(json_file)
    with open(out_file, "r") as f:
        loaded = json.load(f)
    assert loaded[0]["fields"]["path"]["value"] == "<datamaestro>/data/ir/store"


def test_hydrate_task_stubs_and_deduplication(tmp_path: Path):
    """Verify that TaskStubs are hydrated from params.json and shared configs deduplicate."""
    workspace = tmp_path / "workspace"
    jobs_dir = workspace / "jobs" / "xpm_torch.learner.learner" / "f5ff16ef6f80a4dae9ea220692e77722810e8b6625e80bbcb2d93827fd39585c"
    jobs_dir.mkdir(parents=True)

    # 1. Create predecessor params.json (Phase 1)
    shared_tokenizer_ident = "5c2fbcde15373011fe8b80e7d998d95f10319ec3953e9294ae15a24f12136a38"
    phase1_learner_ident = "f5ff16ef6f80a4dae9ea220692e77722810e8b6625e80bbcb2d93827fd39585c"

    phase1_params = {
        "version": 1,
        "objects": [
            {
                "id": 10,
                "type": "HFTokenizer",
                "identifier": shared_tokenizer_ident,
                "fields": {"model_id": "bert-base-uncased"},
            },
            {
                "id": 20,
                "type": "Learner",
                "identifier": phase1_learner_ident,
                "task": 20,
                "fields": {
                    "lr": 5e-5,
                    "max_epochs": 1000,
                    "tokenizer": {"type": "python", "value": 10},
                },
            },
        ],
    }
    with open(jobs_dir / "params.json", "w") as f:
        json.dump(phase1_params, f)

    # 2. Create downstream experimaestro.json (Phase 2) containing a stub of Phase 1 Learner
    phase2_model_dir = tmp_path / "phase2_model"
    phase2_model_dir.mkdir()
    phase2_json = phase2_model_dir / "experimaestro.json"

    phase2_data = [
        # Shared tokenizer already present in Phase 2 with id 1
        {
            "id": 1,
            "type": "HFTokenizer",
            "identifier": shared_tokenizer_ident,
            "fields": {"model_id": "bert-base-uncased"},
        },
        # Hollow TaskStub of Phase 1 Learner
        {
            "id": 2,
            "type": "Learner",
            "identifier": phase1_learner_ident,
            "fields": {},
            "task": None,
        },
        # Phase 2 Learner referencing the stub
        {
            "id": 3,
            "type": "Learner",
            "identifier": "phase2_learner_ident_1234567890abcdef1234567890abcdef1234567890abcdef",
            "task": 3,
            "fields": {
                "base_learner": {"type": "python", "value": 2},
                "max_epochs": 500,
            },
        },
    ]
    with open(phase2_json, "w") as f:
        json.dump(phase2_data, f)

    # 3. Run hydration
    hydrate_task_stubs(phase2_json, workspace_path=workspace)

    with open(phase2_json, "r") as f:
        hydrated_data = json.load(f)

    # 4. Verify results
    by_ident = {o["identifier"]: o for o in hydrated_data}
    assert phase1_learner_ident in by_ident

    hydrated_l1 = by_ident[phase1_learner_ident]
    # Stub should now have real fields!
    assert hydrated_l1["fields"]["lr"] == 5e-5
    assert hydrated_l1["fields"]["max_epochs"] == 1000

    # Tokenizer should be deduplicated: appears only ONCE in the graph!
    tokenizers = [o for o in hydrated_data if o.get("identifier") == shared_tokenizer_ident]
    assert len(tokenizers) == 1
    # Pointer in hydrated L1 should point to the deduplicated tokenizer (id 1)
    assert hydrated_l1["fields"]["tokenizer"]["value"] == 1

    # No synthetic provenance tags
    assert "provenance" not in hydrated_l1


def test_render_model_dag_tree(tmp_path: Path):
    """Verify render_model_dag tree mode executes without error."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    exp_json = model_dir / "experimaestro.json"
    dummy_data = [
        {
            "id": 1,
            "type": "Learner",
            "identifier": "abc1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            "task": 1,
            "fields": {"epochs": 10},
        },
        {
            "id": 2,
            "type": "SimpleModuleLoader",
            "identifier": "def1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            "task": 1,
            "fields": {},
        },
    ]
    with open(exp_json, "w") as f:
        json.dump(dummy_data, f)

    # Call tree output
    render_model_dag(model_dir, tree=True)
