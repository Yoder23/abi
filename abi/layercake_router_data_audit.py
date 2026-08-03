"""Read-only capability counts for candidate compact-router search bundles."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from .layercake_host_v3 import load_english_training_rows


NAMES = (
    "phi3-broad-cumulative-english-training-v9.abix",
    "phi3-natural-cumulative-realization-scale-training-v8.abix",
    "phi3-natural-cumulative-english-search-training-v7.abix",
    "phi3-natural-cumulative-english-search-training-v6.abix",
    "phi3-english-realization-scale-v6-search-survey.abix",
    "phi3-natural-v2-v3-english-search-training-v3.abix",
    "phi3-realization-object-fields-search-training-v1.abix",
)


def main() -> None:
    root = Path(__file__).resolve().parents[1] / (
        "results/abi_moonshot/segregated_acquisition_v3"
    )
    results = []
    loaded = {}
    for name in NAMES:
        path = root / name
        try:
            rows, budget, _ = load_english_training_rows(
                path, budget_index=-1
            )
            loaded[name] = rows
            results.append(
                {
                    "path": str(path),
                    "records": len(rows),
                    "budget_id": budget["budget_id"],
                    "capabilities": dict(
                        sorted(Counter(row["capability"] for row in rows).items())
                    ),
                    "routes": {
                        str(key): value
                        for key, value in sorted(
                            Counter(int(row["route"]) for row in rows).items()
                        )
                    },
                }
            )
        except Exception as exc:
            results.append(
                {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
            )
    broad, cumulative = (
        loaded.get("phi3-broad-cumulative-english-training-v9.abix", []),
        load_english_training_rows(
            root / "phi3-broad-natural-conversation-complete-search-training-v3.abix",
            budget_index=3,
        )[0],
    )
    broad_ids = {row["record_id"] for row in broad}
    cumulative_ids = {row["record_id"] for row in cumulative}
    broad_prompts = {
        hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()
        for row in broad
    }
    cumulative_prompts = {
        hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()
        for row in cumulative
    }
    prompt_routes = {}
    for row in broad:
        prompt_sha = hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()
        prompt_routes.setdefault(prompt_sha, set()).add(int(row["route"]))
    conflicting_prompt_routes = {
        prompt_sha: sorted(routes)
        for prompt_sha, routes in prompt_routes.items()
        if len(routes) > 1
    }
    print(
        json.dumps(
            {
                "bundles": results,
                "v9_vs_broad_natural_v3_overlap": {
                    "record_ids": len(broad_ids & cumulative_ids),
                    "prompt_sha256": len(broad_prompts & cumulative_prompts),
                    "v9_unique_record_ids": len(broad_ids - cumulative_ids),
                    "v9_unique_prompts": len(broad_prompts - cumulative_prompts),
                    "v9_prompt_hashes_with_conflicting_routes": len(
                        conflicting_prompt_routes
                    ),
                    "v9_records_on_conflicting_prompt_hashes": sum(
                        1
                        for row in broad
                        if hashlib.sha256(
                            row["prompt"].encode("utf-8")
                        ).hexdigest()
                        in conflicting_prompt_routes
                    ),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
