"""Offline-first extraction from a frozen Hugging Face causal language model.

The source model is never trained.  Its immutable weights are hashed, prompts
are rendered through its own tokenizer/chat template, generated token IDs are
counted directly, and each result is bound to a declarative evaluator.  This is
behavioral extraction *from* frozen weights; it is not a claim that individual
neurons or tensors have uniquely identifiable English/domain meanings.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .capability_pipeline import (
    CapabilityPipelineError,
    build_probe_result,
    build_source_model_manifest,
)
from .capability_segregation import (
    DOMAIN_CONTENT_BASES,
    ENGLISH_CONTENT_BASES,
    LABEL_METHODS,
    LINGUISTIC_FORM,
    QUARANTINED,
    QUARANTINE_CONTENT_BASES,
    SEGREGATED_RECORD_SCHEMA,
    SPECIALIST_KNOWLEDGE,
    build_segregated_extraction_record,
)
from .layercake_acquisition import build_labeled_extraction_record


PROBE_CATALOG_SCHEMA = "abi-capability-probe-catalog/1"
_SEGREGATION_LABEL_FIELDS = (
    "record_schema",
    "probe_id",
    "destination_scope",
    "capability",
    "domain",
    "prompt",
    "evaluator",
    "knowledge_class",
    "content_basis",
    "domain_labels",
    "domain_claims",
    "output_introduces_unsupplied_facts",
)


def probe_label_evidence_sha256(probe: dict[str, Any]) -> str:
    """Bind a preregistered semantic label to its exact probe definition."""

    basis = {field: probe.get(field) for field in _SEGREGATION_LABEL_FIELDS}
    encoded = json.dumps(
        basis,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prompt_contract_sha256(prompt: str) -> str:
    """Bind a functional evaluator to the exact raw source prompt."""

    if not isinstance(prompt, str) or not prompt:
        raise CapabilityPipelineError("prompt contract requires a non-empty prompt")
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _immutable_revision(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _token_id_set(*values: Any) -> set[int]:
    """Normalize tokenizer/model generation stop IDs without guessing one EOS."""

    output: set[int] = set()
    pending = list(values)
    while pending:
        value = pending.pop()
        if value is None:
            continue
        if isinstance(value, bool):
            raise CapabilityPipelineError("boolean token ID is invalid")
        if isinstance(value, int):
            if value < 0:
                raise CapabilityPipelineError("negative token ID is invalid")
            output.add(value)
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            pending.extend(value)
            continue
        raise CapabilityPipelineError("source stop token IDs are invalid")
    return output


def _extract_python(text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    return fenced.group(1).strip() if fenced else text.strip()


def evaluate_output(output: str, evaluator: dict[str, Any]) -> tuple[bool, float]:
    """Evaluate output using a schema-closed, non-executable rule."""

    if not isinstance(output, str):
        raise CapabilityPipelineError("probe output must be a string")
    if not isinstance(evaluator, dict):
        raise CapabilityPipelineError("probe evaluator must be an object")
    kind = evaluator.get("kind")
    case_sensitive = bool(evaluator.get("case_sensitive", False))
    candidate = output if case_sensitive else output.casefold()

    if kind in {"all_of", "any_of"}:
        rules = evaluator.get("rules")
        if not isinstance(rules, list) or not rules or any(
            not isinstance(rule, dict) for rule in rules
        ):
            raise CapabilityPipelineError(f"{kind} evaluator needs rule objects")
        results = [evaluate_output(output, rule) for rule in rules]
        passed = (
            all(result[0] for result in results)
            if kind == "all_of"
            else any(result[0] for result in results)
        )
        score = sum(result[1] for result in results) / len(results)
        return passed, score

    if kind == "nonempty":
        minimum_characters = int(evaluator.get("minimum_characters", 1))
        score = min(1.0, len(output.strip()) / max(1, minimum_characters))
        return len(output.strip()) >= minimum_characters, score

    if kind in {
        "contains_all",
        "contains_any",
        "contains_none",
        "ordered_contains",
    }:
        values = evaluator.get("values")
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise CapabilityPipelineError(f"{kind} evaluator needs string values")
        needles = values if case_sensitive else [value.casefold() for value in values]
        hits = [needle in candidate for needle in needles]
        score = sum(hits) / len(hits)
        if kind == "contains_all":
            return all(hits), score
        if kind == "contains_any":
            return any(hits), score
        if kind == "contains_none":
            flags = 0 if case_sensitive else re.IGNORECASE
            absent = [
                re.search(
                    r"(?<!\w)" + re.escape(value) + r"(?!\w)",
                    output,
                    flags=flags,
                )
                is None
                for value in values
            ]
            return all(absent), sum(absent) / len(absent)
        cursor = 0
        ordered_hits = 0
        for needle in needles:
            location = candidate.find(needle, cursor)
            if location < 0:
                break
            ordered_hits += 1
            cursor = location + len(needle)
        ordered_score = ordered_hits / len(needles)
        return ordered_hits == len(needles), ordered_score

    if kind == "exact":
        expected = evaluator.get("value")
        if not isinstance(expected, str):
            raise CapabilityPipelineError("exact evaluator needs value")
        expected_value = expected if case_sensitive else expected.casefold()
        passed = candidate.strip() == expected_value.strip()
        return passed, 1.0 if passed else 0.0

    if kind == "regex":
        pattern = evaluator.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise CapabilityPipelineError("regex evaluator needs pattern")
        if len(pattern) > 500:
            raise CapabilityPipelineError("regex evaluator pattern is too long")
        flags = 0 if case_sensitive else re.IGNORECASE
        passed = re.search(pattern, output, flags=flags) is not None
        return passed, 1.0 if passed else 0.0

    if kind in {"json_object", "json_code_block"}:
        json_text = output.strip()
        if kind == "json_code_block":
            fenced = re.fullmatch(
                r"```(?:json)?\s*(.*?)\s*```",
                json_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if fenced is None:
                return False, 0.0
            json_text = fenced.group(1).strip()
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            return False, 0.0
        required_keys = evaluator.get("required_keys", [])
        if (
            not isinstance(required_keys, list)
            or any(not isinstance(key, str) or not key for key in required_keys)
        ):
            raise CapabilityPipelineError(f"{kind} required_keys are invalid")
        if not isinstance(parsed, dict):
            return False, 0.0
        hits = [key in parsed for key in required_keys]
        expected_values = evaluator.get("expected_values", {})
        if not isinstance(expected_values, dict):
            raise CapabilityPipelineError(f"{kind} expected_values are invalid")
        value_hits = [
            key in parsed and parsed[key] == value
            for key, value in expected_values.items()
        ]
        all_hits = hits + value_hits
        score = sum(all_hits) / max(1, len(all_hits))
        return all(all_hits), score

    if kind == "maximum_characters":
        maximum = evaluator.get("value")
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            raise CapabilityPipelineError("maximum_characters needs a positive value")
        passed = len(output.strip()) <= maximum
        score = min(1.0, maximum / max(maximum, len(output.strip())))
        return passed, score

    if kind == "python_compiles":
        try:
            ast.parse(_extract_python(output))
        except (SyntaxError, ValueError):
            return False, 0.0
        required = evaluator.get("contains", [])
        if (
            not isinstance(required, list)
            or any(not isinstance(value, str) or not value for value in required)
        ):
            raise CapabilityPipelineError("python_compiles contains is invalid")
        hits = [value in output for value in required]
        score = 1.0 if not hits else sum(hits) / len(hits)
        return all(hits), score

    if kind == "python_function_expression":
        function_name = evaluator.get("function_name")
        arguments = evaluator.get("arguments")
        expected_expression = evaluator.get("expression")
        if (
            not isinstance(function_name, str)
            or not function_name.isidentifier()
            or not isinstance(arguments, list)
            or any(
                not isinstance(argument, str) or not argument.isidentifier()
                for argument in arguments
            )
            or not isinstance(expected_expression, str)
            or not expected_expression
        ):
            raise CapabilityPipelineError(
                "python_function_expression parameters are invalid"
            )
        try:
            module = ast.parse(_extract_python(output))
            expected = ast.parse(expected_expression, mode="eval").body
        except (SyntaxError, ValueError):
            return False, 0.0
        functions = [
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ]
        if len(functions) != 1:
            return False, 0.0
        function = functions[0]
        actual_arguments = [argument.arg for argument in function.args.args]
        returns = [
            node.value
            for node in function.body
            if isinstance(node, ast.Return) and node.value is not None
        ]
        argument_score = 1.0 if actual_arguments == arguments else 0.0
        expression_match = any(
            ast.dump(value, include_attributes=False)
            == ast.dump(expected, include_attributes=False)
            for value in returns
        )
        score = (argument_score + float(expression_match)) / 2
        return argument_score == 1.0 and expression_match, score

    if kind == "numeric_equal":
        expected = evaluator.get("value")
        tolerance = evaluator.get("absolute_tolerance", 0.0)
        if (
            isinstance(expected, bool)
            or not isinstance(expected, (int, float))
            or isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or float(tolerance) < 0
        ):
            raise CapabilityPipelineError("numeric_equal parameters are invalid")
        matches = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", output.replace(",", ""))
        if not matches:
            return False, 0.0
        distances = [abs(float(value) - float(expected)) for value in matches]
        passed = min(distances) <= float(tolerance)
        return passed, 1.0 if passed else 0.0

    raise CapabilityPipelineError(f"unsupported evaluator kind: {kind!r}")


def load_probe_catalog(path: str | Path) -> dict[str, Any]:
    """Load and validate a declarative probe catalog."""

    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    if catalog.get("schema_version") != PROBE_CATALOG_SCHEMA:
        raise CapabilityPipelineError("unsupported probe catalog schema")
    catalog_id = catalog.get("catalog_id")
    if not isinstance(catalog_id, str) or not catalog_id.strip():
        raise CapabilityPipelineError("probe catalog_id is missing")
    probes = catalog.get("probes")
    if not isinstance(probes, list) or not probes:
        raise CapabilityPipelineError("probe catalog is empty")
    probe_ids: set[str] = set()
    for probe in probes:
        if not isinstance(probe, dict):
            raise CapabilityPipelineError("probe must be an object")
        probe_id = probe.get("probe_id")
        if not isinstance(probe_id, str) or not probe_id.strip():
            raise CapabilityPipelineError("probe_id is missing")
        if probe_id in probe_ids:
            raise CapabilityPipelineError(f"duplicate probe_id: {probe_id}")
        probe_ids.add(probe_id)
        scope = probe.get("destination_scope")
        domain = probe.get("domain")
        capability = probe.get("capability")
        if scope not in {"english_core", "domain_cake"}:
            raise CapabilityPipelineError("invalid probe destination_scope")
        if not isinstance(domain, str) or not domain:
            raise CapabilityPipelineError("probe domain is missing")
        if not isinstance(capability, str) or not capability:
            raise CapabilityPipelineError("probe capability is missing")
        if scope == "english_core" and domain != "domain_independent":
            raise CapabilityPipelineError("English probes must be domain-independent")
        if scope == "domain_cake" and domain == "domain_independent":
            raise CapabilityPipelineError("domain probes must name a specialist domain")
        if probe.get("split") not in {"search", "validation", "final_test"}:
            raise CapabilityPipelineError("invalid probe split")
        if not isinstance(probe.get("prompt"), str) or not probe["prompt"]:
            raise CapabilityPipelineError("probe prompt is missing")
        evaluator = probe.get("evaluator")
        if isinstance(evaluator, dict) and "prompt_contract_sha256" in evaluator:
            if evaluator.get("prompt_contract_sha256") != prompt_contract_sha256(
                probe["prompt"]
            ):
                raise CapabilityPipelineError(
                    "functional evaluator prompt contract is stale"
                )
        maximum = probe.get("max_new_tokens", 64)
        if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 2048:
            raise CapabilityPipelineError("max_new_tokens must be in [1, 2048]")
        seed = probe.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise CapabilityPipelineError("probe seed must be non-negative")
        segregation_fields = {
            "knowledge_class",
            "content_basis",
            "domain_labels",
            "domain_claims",
            "label_method",
            "label_evidence_sha256",
            "output_introduces_unsupplied_facts",
        }
        present = segregation_fields & set(probe)
        if present and present != segregation_fields:
            missing = sorted(segregation_fields - present)
            raise CapabilityPipelineError(
                f"segregated probe metadata is incomplete: {missing}"
            )
        record_schema = probe.get("record_schema")
        if present and record_schema != SEGREGATED_RECORD_SCHEMA:
            raise CapabilityPipelineError(
                "segregated probes must declare record schema v2"
            )
        if record_schema == SEGREGATED_RECORD_SCHEMA and not present:
            raise CapabilityPipelineError(
                "record schema v2 requires segregation metadata"
            )
        if record_schema not in (None, SEGREGATED_RECORD_SCHEMA):
            raise CapabilityPipelineError(
                "unsupported extraction record schema"
            )
        if (
            record_schema == SEGREGATED_RECORD_SCHEMA
            and probe.get("label_method") not in LABEL_METHODS
        ):
            raise CapabilityPipelineError(
                "invalid segregated probe label_method"
            )
        if record_schema == SEGREGATED_RECORD_SCHEMA:
            knowledge_class = probe.get("knowledge_class")
            content_basis = probe.get("content_basis")
            domain_labels = probe.get("domain_labels")
            domain_claims = probe.get("domain_claims")
            introduces_facts = probe.get(
                "output_introduces_unsupplied_facts"
            )
            if knowledge_class not in {
                LINGUISTIC_FORM,
                SPECIALIST_KNOWLEDGE,
                QUARANTINED,
            }:
                raise CapabilityPipelineError(
                    "invalid segregated probe knowledge_class"
                )
            if not isinstance(domain_labels, list) or not isinstance(
                domain_claims, list
            ):
                raise CapabilityPipelineError(
                    "segregated probe domain metadata must be lists"
                )
            if not isinstance(introduces_facts, bool):
                raise CapabilityPipelineError(
                    "segregated probe fact-introduction flag must be boolean"
                )
            if scope == "english_core":
                if knowledge_class != LINGUISTIC_FORM:
                    raise CapabilityPipelineError(
                        "English probe must carry linguistic form"
                    )
                if content_basis not in ENGLISH_CONTENT_BASES:
                    raise CapabilityPipelineError(
                        "English probe content_basis is not knowledge-minimized"
                    )
                if domain_labels or domain_claims:
                    raise CapabilityPipelineError(
                        "English probe cannot carry domain labels or claims"
                    )
                if introduces_facts:
                    raise CapabilityPipelineError(
                        "English probe cannot allow unsupplied facts"
                    )
            elif knowledge_class == QUARANTINED:
                if content_basis not in QUARANTINE_CONTENT_BASES:
                    raise CapabilityPipelineError(
                        "invalid quarantined probe content_basis"
                    )
            elif content_basis not in DOMAIN_CONTENT_BASES:
                raise CapabilityPipelineError(
                    "invalid domain probe content_basis"
                )
        if (
            record_schema == SEGREGATED_RECORD_SCHEMA
            and probe.get("label_method") == "preregistered_catalog"
            and probe.get("label_evidence_sha256")
            != probe_label_evidence_sha256(probe)
        ):
            raise CapabilityPipelineError(
                "preregistered semantic label evidence is stale"
            )
        evaluate_output("", evaluator)
    return catalog


class HuggingFaceCausalSource:
    """Frozen, offline-first Hugging Face causal-LM source adapter."""

    def __init__(
        self,
        model_id_or_path: str,
        *,
        revision: str | None = None,
        license_id: str,
        device: str = "auto",
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        use_chat_template: bool = True,
        load_in_8bit: bool = False,
    ) -> None:
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise CapabilityPipelineError(
                "torch, transformers, and huggingface_hub are required"
            ) from exc

        self._torch = torch
        self.model_id_or_path = model_id_or_path
        self.license_id = license_id
        self.local_files_only = bool(local_files_only)
        self.trust_remote_code = bool(trust_remote_code)
        self.use_chat_template = bool(use_chat_template)
        if not isinstance(load_in_8bit, bool):
            raise CapabilityPipelineError("load_in_8bit must be boolean")
        self.load_in_8bit = load_in_8bit
        supplied_path = Path(model_id_or_path)
        if supplied_path.exists():
            self.snapshot_path = supplied_path.resolve()
            resolved_revision = revision or f"local-{_sha256_file(self.snapshot_path / 'config.json')[:32]}"
            revision_is_immutable = bool(revision and _immutable_revision(revision))
            load_reference = str(self.snapshot_path)
        else:
            try:
                snapshot = snapshot_download(
                    repo_id=model_id_or_path,
                    revision=revision,
                    local_files_only=self.local_files_only,
                )
            except Exception as exc:
                raise CapabilityPipelineError(
                    f"unable to resolve frozen source model {model_id_or_path!r}"
                ) from exc
            self.snapshot_path = Path(snapshot).resolve()
            resolved_revision = self.snapshot_path.name
            revision_is_immutable = _immutable_revision(resolved_revision)
            load_reference = str(self.snapshot_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            load_reference,
            local_files_only=True,
            trust_remote_code=self.trust_remote_code,
        )
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device in {"cpu", "cuda"}:
            if device == "cuda" and not torch.cuda.is_available():
                raise CapabilityPipelineError("CUDA requested but unavailable")
            self.device = device
        else:
            raise CapabilityPipelineError("device must be auto, cpu, or cuda")
        if self.load_in_8bit and self.device != "cuda":
            raise CapabilityPipelineError(
                "8-bit source inference requires an explicitly available CUDA device"
            )
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": self.trust_remote_code,
            "dtype": dtype,
        }
        if self.load_in_8bit:
            model_kwargs.update(
                {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=False,
                    ),
                    "device_map": {"": torch.cuda.current_device()},
                }
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            load_reference,
            **model_kwargs,
        )
        if not self.load_in_8bit:
            self.model.to(self.device)
        self.model.eval()
        self.revision = resolved_revision
        self.revision_is_immutable = revision_is_immutable
        if self.load_in_8bit:
            try:
                import bitsandbytes

                quantization_library_version = str(bitsandbytes.__version__)
            except (ImportError, AttributeError):
                quantization_library_version = "unknown"
            weight_execution_precision = "bitsandbytes_int8"
            quantization_library = "bitsandbytes"
        else:
            quantization_library_version = None
            weight_execution_precision = (
                "torch_float16" if self.device == "cuda" else "torch_float32"
            )
            quantization_library = None
        self.source_inference_runtime = {
            "device": self.device,
            "weight_execution_precision": weight_execution_precision,
            "non_quantized_compute_dtype": str(dtype).replace("torch.", ""),
            "quantization_library": quantization_library,
            "quantization_library_version": quantization_library_version,
            "cpu_offload_enabled": False,
        }

        weight_paths = sorted(
            {
                path
                for suffix in ("*.safetensors", "pytorch_model*.bin")
                for path in self.snapshot_path.glob(suffix)
                if path.is_file()
            }
        )
        if not weight_paths:
            raise CapabilityPipelineError("resolved source snapshot has no weight files")
        weight_files = [
            {
                "relative_path": path.relative_to(self.snapshot_path).as_posix(),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in weight_paths
        ]
        architecture = (
            self.model.config.architectures[0]
            if getattr(self.model.config, "architectures", None)
            else self.model.__class__.__name__
        )
        self.source_manifest = build_source_model_manifest(
            model_id=model_id_or_path,
            revision=self.revision,
            revision_is_immutable=self.revision_is_immutable,
            architecture=architecture,
            parameter_count=sum(parameter.numel() for parameter in self.model.parameters()),
            tokenizer_id=model_id_or_path,
            tokenizer_revision=self.revision,
            license_id=license_id,
            weight_files=weight_files,
            trust_remote_code=self.trust_remote_code,
        )

    def rendered_prompt(self, prompt: str) -> str:
        if self.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def generation_eos_token_ids(self) -> set[int]:
        """Return the exact union of runtime-configured source EOS token IDs."""

        generation_config = getattr(self.model, "generation_config", None)
        model_config = getattr(self.model, "config", None)
        eos_ids = _token_id_set(
            getattr(generation_config, "eos_token_id", None),
            getattr(model_config, "eos_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
        )
        if not eos_ids:
            raise CapabilityPipelineError(
                "source runtime declares no EOS token IDs"
            )
        return eos_ids

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        seed: int,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        torch = self._torch
        rendered = self.rendered_prompt(prompt)
        encoded = self.tokenizer(rendered, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_tokens = int(encoded["input_ids"].shape[-1])
        do_sample = float(temperature) > 0.0
        devices = [torch.cuda.current_device()] if self.device == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if self.device == "cuda":
                torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                sequences = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=float(temperature) if do_sample else None,
                    pad_token_id=(
                        self.tokenizer.pad_token_id
                        if self.tokenizer.pad_token_id is not None
                        else self.tokenizer.eos_token_id
                    ),
                )
        generated_ids = sequences[0, input_tokens:]
        raw_ids = [int(value) for value in generated_ids.tolist()]
        eos_ids = self.generation_eos_token_ids()
        eos_position = next(
            (index for index, token_id in enumerate(raw_ids) if token_id in eos_ids),
            None,
        )
        if eos_position is None:
            if len(raw_ids) != max_new_tokens:
                raise CapabilityPipelineError(
                    "source generation stopped without EOS or the declared length ceiling"
                )
            authoritative_ids = raw_ids
            finish_reason = "length"
        else:
            authoritative_ids = raw_ids[: eos_position + 1]
            finish_reason = "eos_token"
        output = self.tokenizer.decode(
            authoritative_ids, skip_special_tokens=True
        )
        return {
            "rendered_prompt": rendered,
            "output": output,
            "input_tokens": input_tokens,
            "teacher_tokens": len(authoritative_ids),
            "teacher_token_counter": "authoritative_generated_token_ids",
            "authoritative_generated_token_ids": authoritative_ids,
            "finish_reason": finish_reason,
            "generation_max_new_tokens": max_new_tokens,
        }

    def generate_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Greedily generate a same-budget batch with per-row token counts."""

        if not requests:
            return []
        maximums = {int(request["max_new_tokens"]) for request in requests}
        temperatures = {float(request.get("temperature", 0.0)) for request in requests}
        if len(maximums) != 1 or temperatures != {0.0}:
            return [
                self.generate(
                    request["prompt"],
                    max_new_tokens=int(request["max_new_tokens"]),
                    seed=int(request["seed"]),
                    temperature=float(request.get("temperature", 0.0)),
                )
                for request in requests
            ]
        torch = self._torch
        rendered = [self.rendered_prompt(str(request["prompt"])) for request in requests]
        original_padding_side = self.tokenizer.padding_side
        original_pad_token_id = self.tokenizer.pad_token_id
        try:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            encoded = self.tokenizer(rendered, return_tensors="pt", padding=True)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            padded_input_tokens = int(encoded["input_ids"].shape[-1])
            input_counts = encoded["attention_mask"].sum(dim=1).tolist()
            with torch.inference_mode():
                sequences = self.model.generate(
                    **encoded,
                    max_new_tokens=next(iter(maximums)),
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated = sequences[:, padded_input_tokens:]
            eos_ids = self.generation_eos_token_ids()
            outputs: list[dict[str, Any]] = []
            for index, token_row in enumerate(generated):
                ids = token_row.tolist()
                counted = len(ids)
                finish_reason = "length"
                for position, token_id in enumerate(ids):
                    if token_id in eos_ids:
                        counted = position + 1
                        finish_reason = "eos_token"
                        break
                authoritative = token_row[:counted]
                authoritative_ids = [
                    int(value) for value in authoritative.tolist()
                ]
                outputs.append(
                    {
                        "rendered_prompt": rendered[index],
                        "output": self.tokenizer.decode(
                            authoritative, skip_special_tokens=True
                        ),
                        "input_tokens": int(input_counts[index]),
                        "teacher_tokens": counted,
                        "teacher_token_counter": (
                            "authoritative_generated_token_ids"
                        ),
                        "authoritative_generated_token_ids": authoritative_ids,
                        "finish_reason": finish_reason,
                        "generation_max_new_tokens": next(iter(maximums)),
                    }
                )
            return outputs
        finally:
            self.tokenizer.padding_side = original_padding_side
            self.tokenizer.pad_token_id = original_pad_token_id


def run_probe_catalog(
    source: HuggingFaceCausalSource,
    catalog: dict[str, Any],
    *,
    split: str | None = None,
    batch_size: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run catalog probes and return labeled records plus bound results."""

    if catalog.get("schema_version") != PROBE_CATALOG_SCHEMA:
        raise CapabilityPipelineError("probe catalog must be validated first")
    if split is not None and split not in {"search", "validation", "final_test"}:
        raise CapabilityPipelineError("invalid probe split")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise CapabilityPipelineError("batch_size must be a positive integer")
    records: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    selected_probes = [
        probe
        for probe in catalog["probes"]
        if split is None or probe["split"] == split
    ]
    samples_by_id: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for probe in selected_probes:
        key = (
            int(probe.get("max_new_tokens", 64)),
            float(probe.get("temperature", 0.0)),
        )
        grouped.setdefault(key, []).append(probe)
    for (maximum, temperature), probes in grouped.items():
        for start in range(0, len(probes), batch_size):
            chunk = probes[start : start + batch_size]
            requests = [
                {
                    "prompt": probe["prompt"],
                    "max_new_tokens": maximum,
                    "temperature": temperature,
                    "seed": int(probe.get("seed", 0)),
                }
                for probe in chunk
            ]
            if batch_size > 1 and hasattr(source, "generate_batch"):
                samples = source.generate_batch(requests)
            else:
                samples = [
                    source.generate(
                        request["prompt"],
                        max_new_tokens=request["max_new_tokens"],
                        seed=request["seed"],
                        temperature=request["temperature"],
                    )
                    for request in requests
                ]
            if len(samples) != len(chunk):
                raise CapabilityPipelineError("source batch returned the wrong row count")
            for probe, sample in zip(chunk, samples, strict=True):
                samples_by_id[str(probe["probe_id"])] = sample

    for probe in selected_probes:
        sample = samples_by_id[str(probe["probe_id"])]
        passed, score = evaluate_output(sample["output"], probe["evaluator"])
        if sample.get("finish_reason") == "length":
            passed = False
        common = {
            "destination_scope": probe["destination_scope"],
            "capability": probe["capability"],
            "domain": probe["domain"],
            "provenance": f"{catalog['catalog_id']}:{probe['probe_id']}",
            "split": probe["split"],
            "source_model": source.source_manifest["model_id"],
            "source_model_revision": source.source_manifest["revision"],
            "prompt": sample["rendered_prompt"],
            "output": sample["output"],
            "teacher_tokens": sample["teacher_tokens"],
            "teacher_token_counter": sample["teacher_token_counter"],
            "authoritative_generated_token_ids": sample.get(
                "authoritative_generated_token_ids"
            ),
            "finish_reason": sample.get("finish_reason"),
            "generation_max_new_tokens": sample.get(
                "generation_max_new_tokens"
            ),
            "teacher_input_tokens": sample.get("input_tokens"),
        }
        if probe.get("record_schema") == SEGREGATED_RECORD_SCHEMA:
            record = build_segregated_extraction_record(
                **common,
                knowledge_class=probe["knowledge_class"],
                content_basis=probe["content_basis"],
                domain_labels=probe["domain_labels"],
                domain_claims=probe["domain_claims"],
                label_method=probe["label_method"],
                label_evidence_sha256=probe["label_evidence_sha256"],
                output_introduces_unsupplied_facts=probe[
                    "output_introduces_unsupplied_facts"
                ],
            )
        else:
            record = build_labeled_extraction_record(**common)
        result = build_probe_result(
            record=record,
            source_manifest_sha256=source.source_manifest["source_manifest_sha256"],
            probe_id=probe["probe_id"],
            evaluator=probe["evaluator"],
            passed=passed,
            score=score,
            seed=int(probe.get("seed", 0)),
        )
        records.append(record)
        results.append(result)
    if not records:
        raise CapabilityPipelineError("no probes matched the requested split")
    return records, results
