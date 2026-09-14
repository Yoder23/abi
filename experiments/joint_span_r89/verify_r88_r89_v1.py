"""Strict raw-row recomputation for the R88 and R89 bounded results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from abi.conditional_choice_artifact_v2 import _canonical_sha
from abi.english_generalization_evaluation import _collapse_metrics
from abi.hf_extraction import evaluate_output, load_probe_catalog
from abi.layercake_host import _sha256_file
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once
from experiments.generic_prompt_identity_r83.screen_host_v1 import _bootstrap, _jsonl
from experiments.joint_span_r88.core_v1 import validate_metadata


EXPECTED = {
    "r88": {
        "result": "99ec04026397510656500762f8c57a29dfe605065a49d4245c26fd6c73d8b853",
        "raw": "758643dd8eb2384f7aa8db757f0da82f69bbeea62816a34425cd4be0bc326dbb",
        "catalog": "5dba762eec849f2e6531f5f1283417d38e9470a9b37ca69cf3576bb25286a177",
        "source_result": "facac7a81732805ad7c2dece8160ce22e74532805d5b5da4fe2f8a8aec1baf4e",
        "source_raw": "08245820235fe4c553be5c7c961dda2ec8c21938374f241b84fc0a94a9123868",
        "source_evidence": "de446e86df020b29732c011185c48f6e0120bac2b47b9299ea95336dfacd8e29",
        "binding_file": "f669929e26cb1e2800df4063d1a0260f84406111cf2b11d7bad56f46c5b07226",
        "binding_claim": "129f5023187f2d4f1d82b6e30f83297ee8d52de44f9e1184dfd20c9be3892112",
        "seed": 88_000,
    },
    "r89": {
        "result": "63b8b632184a021c33fe5fb72b7af322d20400eda8fc356e6589936316df5616",
        "raw": "23dd0fd2b02a2526498255d72a2639eb2242a3d99a52816777a2c6550a9e7708",
        "catalog": "e21bf7e3a2f3c72d5b82307009fb05c3e1e0623208d9023a13baeaf1e265a118",
        "source_result": "8384d590e5d66a12fdab938f776b8fa0fa5cf091fe40778b17a5e0d20b47fdbb",
        "source_raw": "70850638029b6e238a25ba63543ad8111b586fc0a69926863fdf7831b0628994",
        "source_evidence": "a1e7c2c3887bb3cf900d010a572955392a7f51a4a68988541f422b4a81731e48",
        "binding_file": "a5f869fe51262f7c2b5b5e8f7aa7a6605e267bc98d03a86dbd9fac0d7c8acc67",
        "binding_claim": "71e7066a54428200bb6448c87a6e1bc40c909089574aa7872313ec1f8a3dc1d7",
        "seed": 89_000,
    },
}
CANDIDATE_SHA256 = "15c28bdceea49e61e771092ce74be1cc30233e8bef53d87171d36f5a2487372f"
CANDIDATE_METADATA_SHA256 = "d98c53ff713ead136b9a659743553b40fed2fa709d0127b93550f8fef7bfaabe"
PARENT_SHA256 = "b6977f087ac42e6e4234d026b4cd83827b720d8973cca049daf18bcc8b96a64e"
PARENT_METADATA_SHA256 = "1c91e94abc3f2faa9a6f7d68689451dc94098dd2330652a4713a0116c3080e0e"
ARTIFACT_SHA256 = "292ba40ced84db5a28ef3c8214ac7645623db5e0f047218f5f7bf7c2ce0b10cc"
PARENT_TOKENIZER_FILES = {
    "merges.txt": (456_318, "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"),
    "special_tokens_map.json": (494, "fee4e5d22631d2d4598132f059304af341ff9c5ed7a5bde94be6a7f47a1a3817"),
    "tokenizer.json": (3_557_680, "1fe93b6152957cf9cfd6d89002467f789ce8b3f3e000b3a2edf27c808ddd0b9e"),
    "tokenizer_config.json": (534, "c8e5a90723b23e61d70174aeaa0d3688716f6a4b5a9ddf4cc1027b978e187899"),
    "vocab.json": (798_156, "3ba3c3109ff33976c4bd966589c11ee14fcaa1f4c9e5e154c2ed7f99d80709e7"),
}


class VerificationError(RuntimeError):
    pass


def _verify_evidence_result(path: Path, expected_hash: str, expected_evidence: str) -> dict[str, Any]:
    if not path.is_file() or _sha256_file(path) != expected_hash:
        raise VerificationError(f"result file missing or changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.get("evidence_sha256")
    unsigned = dict(value)
    unsigned.pop("evidence_sha256", None)
    if claimed != expected_evidence or evidence_hash(unsigned) != claimed:
        raise VerificationError(f"result evidence is stale or unrecomputable: {path}")
    return value


def _verify_candidate(root: Path) -> dict[str, Any]:
    candidate = root / "results/joint_span_r88/candidate_v1"
    parent = root / "results/role_invariant_choice_r76/candidate_v1"
    artifact = root / "results/role_invariant_choice_r76/reasoning-role-invariant-search-v2.abix"
    checkpoint = candidate / "bridge.safetensors"
    metadata_path = candidate / "metadata.json"
    if (
        _sha256_file(checkpoint) != CANDIDATE_SHA256
        or _sha256_file(metadata_path) != CANDIDATE_METADATA_SHA256
        or _sha256_file(parent / "model.safetensors") != PARENT_SHA256
        or _sha256_file(parent / "metadata.json") != PARENT_METADATA_SHA256
        or _sha256_file(artifact) != ARTIFACT_SHA256
    ):
        raise VerificationError("frozen R88 candidate bytes changed")
    for name, (expected_bytes, expected_sha256) in PARENT_TOKENIZER_FILES.items():
        path = parent / name
        if (
            not path.is_file()
            or path.stat().st_size != expected_bytes
            or _sha256_file(path) != expected_sha256
        ):
            raise VerificationError(f"frozen parent tokenizer file changed: {name}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_metadata(metadata, candidate)
    if (
        metadata["source_boundary"]["teacher_present_at_inference"] is not False
        or metadata["source_boundary"]["source_parameters_copied"] != 0
        or metadata["bridge"]["parameter_count"] != 530_050
        or metadata["training"]["successful_optimizer_steps"] != 3_000
        or metadata["training"]["unique_records_seen"] != 8_129
        or metadata["normalization"]["synthetic_examples_consumed"] != 96_000
        or metadata["parent_layercake"]["checkpoint_sha256"] != PARENT_SHA256
        or metadata["parent_layercake"]["metadata_sha256"] != PARENT_METADATA_SHA256
        or metadata["imported_artifact"]["sha256"] != ARTIFACT_SHA256
    ):
        raise VerificationError("R88 package accounting changed")
    return metadata


def _verify_r88_binding(root: Path) -> dict[str, Any]:
    path = root / "results/joint_span_r88/candidate_binding_v1.json"
    if _sha256_file(path) != EXPECTED["r88"]["binding_file"]:
        raise VerificationError("R88 candidate binding file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.get("binding_sha256")
    unsigned = {key: item for key, item in value.items() if key != "binding_sha256"}
    files = {
        "screen_sha256": root / "experiments/joint_span_r88/screen_host_v1.py",
        "protocol_sha256": root / "experiments/joint_span_r88/PROTOCOL.md",
        "core_sha256": root / "experiments/joint_span_r88/core_v1.py",
        "trainer_sha256": root / "experiments/joint_span_r88/train_candidate_v1.py",
    }
    if (
        claimed != EXPECTED["r88"]["binding_claim"]
        or _canonical_sha(unsigned) != claimed
        or value.get("candidate_generation_observations_before_binding") != 0
        or value.get("candidate_checkpoint_sha256") != CANDIDATE_SHA256
        or value.get("candidate_metadata_sha256") != CANDIDATE_METADATA_SHA256
        or any(value.get(key) != _sha256_file(path) for key, path in files.items())
    ):
        raise VerificationError("R88 candidate binding is stale or unrecomputable")
    return value


def _verify_r89_binding(root: Path) -> dict[str, Any]:
    path = root / "results/joint_span_r89/holdout_binding_v1.json"
    if _sha256_file(path) != EXPECTED["r89"]["binding_file"]:
        raise VerificationError("R89 holdout binding file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.get("binding_sha256")
    unsigned = {key: item for key, item in value.items() if key != "binding_sha256"}
    files = {
        "candidate_checkpoint": root / "results/joint_span_r88/candidate_v1/bridge.safetensors",
        "candidate_metadata": root / "results/joint_span_r88/candidate_v1/metadata.json",
        "candidate_binding": root / "results/joint_span_r88/candidate_binding_v1.json",
        "candidate_screen": root / "experiments/joint_span_r88/screen_host_v1.py",
        "candidate_protocol": root / "experiments/joint_span_r88/PROTOCOL.md",
        "candidate_core": root / "experiments/joint_span_r88/core_v1.py",
        "candidate_trainer": root / "experiments/joint_span_r88/train_candidate_v1.py",
        "holdout_screen": root / "experiments/joint_span_r89/screen_frozen_v1.py",
        "holdout_protocol": root / "experiments/joint_span_r89/PROTOCOL.md",
        "catalog": root / "catalogs/joint_span_validation_r89_v1.json",
        "source_result": root / "results/joint_span_r89/source_v1/result.json",
        "source_raw": root / "results/joint_span_r89/source_v1/prior_corrected_scores.jsonl",
    }
    if (
        claimed != EXPECTED["r89"]["binding_claim"]
        or _canonical_sha(unsigned) != claimed
        or value.get("holdout_observations_before_binding") != 0
        or set(value.get("files", {})) != set(files)
    ):
        raise VerificationError("R89 holdout binding structure changed")
    for name, path in files.items():
        item = value["files"][name]
        if (
            not path.is_file()
            or item.get("sha256") != _sha256_file(path)
            or item.get("bytes") != path.stat().st_size
        ):
            raise VerificationError(f"R89 holdout binding is stale: {name}")
    return value


def _verify_campaign(root: Path, campaign: str, tokenizer: Any) -> dict[str, Any]:
    expected = EXPECTED[campaign]
    result_dir = root / f"results/joint_span_{campaign}/screen_v1"
    result = _verify_evidence_result(
        result_dir / "result.json", expected["result"], result_expected_evidence(campaign)
    )
    raw_path = result_dir / "evaluation.jsonl"
    catalog_path = root / f"catalogs/joint_span_validation_{campaign}_v1.json"
    source_result_path = root / f"results/joint_span_{campaign}/source_v1/result.json"
    source_raw_path = root / f"results/joint_span_{campaign}/source_v1/prior_corrected_scores.jsonl"
    if (
        _sha256_file(raw_path) != expected["raw"]
        or _sha256_file(catalog_path) != expected["catalog"]
        or _sha256_file(source_raw_path) != expected["source_raw"]
    ):
        raise VerificationError(f"{campaign} raw evidence or catalog changed")
    source_result = _verify_evidence_result(
        source_result_path, expected["source_result"], expected["source_evidence"]
    )
    if source_result.get("candidate_accessed") is not False:
        raise VerificationError(f"{campaign} source accessed a candidate")
    probes = list(load_probe_catalog(catalog_path)["probes"])
    rows = _jsonl(raw_path)
    source_rows = {str(row["probe_id"]): row for row in _jsonl(source_raw_path)}
    probe_by_id = {str(row["probe_id"]): row for row in probes}
    if (
        len(rows) != 1_400
        or len(probes) != 1_400
        or len({str(row["probe_id"]) for row in rows}) != 1_400
        or set(source_rows) != set(probe_by_id)
    ):
        raise VerificationError(f"{campaign} matrix coverage changed")
    for row in rows:
        probe_id = str(row["probe_id"])
        probe = probe_by_id.get(probe_id)
        source = source_rows.get(probe_id)
        if probe is None or source is None:
            raise VerificationError(f"{campaign} row is not catalog/source paired")
        prompt = str(probe["prompt"])
        evaluator = probe["evaluator"]
        candidate_passed, _ = evaluate_output(str(row["candidate_output"]), evaluator)
        parent_passed, _ = evaluate_output(str(row["parent_output"]), evaluator)
        random_passed, _ = evaluate_output(str(row["random_output"]), evaluator)
        prompt_ids = tokenizer.encode(prompt + "\n", add_special_tokens=False)
        start, end = int(row["candidate_start"]), int(row["candidate_end"])
        if not 0 <= start <= end < len(prompt_ids) or end - start >= 16:
            raise VerificationError(f"{campaign} illegal realized span")
        realized = tokenizer.decode(
            prompt_ids[start : end + 1],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        collapse = _collapse_metrics(
            list(row["candidate_token_ids"]), str(row["candidate_output"]),
            prompt_ids, prompt,
        )
        if (
            row["prompt_sha256"] != hashlib.sha256(prompt.encode()).hexdigest()
            or source["expected_output_sha256"]
            != hashlib.sha256(str(evaluator["value"]).encode()).hexdigest()
            or bool(row["source_passed"]) != bool(source["prior_corrected_passed"])
            or bool(row["candidate_passed"]) != bool(candidate_passed)
            or bool(row["parent_passed"]) != bool(parent_passed)
            or bool(row["random_passed"]) != bool(random_passed)
            or row["candidate_collapse"] != collapse
            or str(row["candidate_output"]) != realized
            or list(row["candidate_token_ids"])
            != tokenizer.encode(str(row["candidate_output"]), add_special_tokens=False)
            or row["candidate_physical_sparse"] is not True
        ):
            raise VerificationError(f"{campaign} row is stale or unrecomputable: {probe_id}")
    source_passing = sum(bool(row["source_passed"]) for row in rows)
    candidate_passing = sum(bool(row["candidate_passed"]) for row in rows)
    parent_passing = sum(bool(row["parent_passed"]) for row in rows)
    random_passing = sum(bool(row["random_passed"]) for row in rows)
    retained = sum(bool(row["source_passing_retained"]) for row in rows)
    for row in rows:
        if bool(row["source_passing_retained"]) != (
            bool(row["source_passed"]) and bool(row["candidate_passed"])
        ):
            raise VerificationError(f"{campaign} retention row is stale")
    family = {
        str(index): {
            "source_passing": sum(row["premise_family"] == index and row["source_passed"] for row in rows),
            "candidate_passing": sum(row["premise_family"] == index and row["candidate_passed"] for row in rows),
            "parent_passing": sum(row["premise_family"] == index and row["parent_passed"] for row in rows),
            "random_passing": sum(row["premise_family"] == index and row["random_passed"] for row in rows),
        }
        for index in range(7)
    }
    metrics = result["metrics"]
    recomputed = {
        "rows": len(rows), "source_passing": source_passing,
        "candidate_passing": candidate_passing, "parent_passing": parent_passing,
        "random_bridge_passing": random_passing,
        "source_passing_retained": retained,
        "source_retention": retained / source_passing,
        "candidate_collapses": sum(row["candidate_collapse"]["collapse_detected"] for row in rows),
        "candidate_physical_sparse_rows": sum(row["candidate_physical_sparse"] for row in rows),
        "by_family": family,
        "candidate_minus_source": _bootstrap(
            [row["candidate_passed"] for row in rows], [row["source_passed"] for row in rows],
            expected["seed"] + 1,
        ),
        "candidate_minus_parent": _bootstrap(
            [row["candidate_passed"] for row in rows], [row["parent_passed"] for row in rows],
            expected["seed"] + 2,
        ),
        "candidate_minus_random": _bootstrap(
            [row["candidate_passed"] for row in rows], [row["random_passed"] for row in rows],
            expected["seed"] + 3,
        ),
    }
    if any(metrics.get(key) != value for key, value in recomputed.items()):
        raise VerificationError(f"{campaign} aggregate metrics do not recompute")
    gates = {
        "matrix": len(rows) == 1_400,
        "source_quality": source_passing >= 1_330,
        "candidate_quality": candidate_passing >= 1_330,
        "candidate_family_floor": min(value["candidate_passing"] for value in family.values()) >= 180,
        "source_retention": retained / source_passing >= 0.95,
        "causal_parent_gain": (candidate_passing - parent_passing) / 1_400 >= 0.50,
        "random_bridge_fails": random_passing <= 500,
        "trained_beats_random": (candidate_passing - random_passing) / 1_400 >= 0.50,
        "zero_collapse": recomputed["candidate_collapses"] == 0,
        "physical_sparse": recomputed["candidate_physical_sparse_rows"] == 1_400,
        "teacher_absent": result.get("teacher_present_at_inference") is False,
        "artifacts_unchanged": True,
    }
    if result.get("gates") != gates or not all(gates.values()):
        raise VerificationError(f"{campaign} gate vector does not recompute")
    if not all(math.isfinite(float(metrics[key])) for key in (
        "parent_generation_seconds", "candidate_and_random_generation_seconds",
        "candidate_peak_cuda_allocated_bytes", "process_rss_bytes",
    )):
        raise VerificationError(f"{campaign} runtime accounting is non-finite")
    return {
        "rows": len(rows), "candidate_passing": candidate_passing,
        "source_passing": source_passing, "parent_passing": parent_passing,
        "random_passing": random_passing, "collapses": recomputed["candidate_collapses"],
        "result_sha256": expected["result"], "raw_sha256": expected["raw"],
    }


def result_expected_evidence(campaign: str) -> str:
    return {
        "r88": "f531c94f8dd8ed4aced03fb2a620a2e1fe6ea5355e0b4a1211826eb64d41184e",
        "r89": "cf9cf6f44b9e3fd07ad4b3c384de64c39ca92a67ac6c34b30ef032ecfea016a0",
    }[campaign]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.output.exists():
        parser.error(f"immutable verification receipt exists: {args.output}")
    _verify_candidate(root)
    _verify_r88_binding(root)
    _verify_r89_binding(root)
    tokenizer = AutoTokenizer.from_pretrained(
        root / "results/role_invariant_choice_r76/candidate_v1",
        local_files_only=True,
    )
    campaigns = {
        campaign: _verify_campaign(root, campaign, tokenizer)
        for campaign in ("r88", "r89")
    }
    receipt = {
        "format": "abi-r88-r89-strict-raw-recomputation/1",
        "verdict": "PASS_R88_R89_STRICT_RECOMPUTATION",
        "stored_scientific_booleans_trusted": False,
        "candidate_checkpoint_sha256": CANDIDATE_SHA256,
        "candidate_metadata_sha256": CANDIDATE_METADATA_SHA256,
        "campaigns": campaigns,
        "full_abi_moonshot": "OPEN",
        "claim_boundary": "Strict verification of the bounded R88/R89 joint-span results only.",
    }
    receipt["evidence_sha256"] = evidence_hash(receipt)
    write_json_once(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
