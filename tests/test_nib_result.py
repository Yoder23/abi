"""
Tests for the ABI package.

Run with: pytest tests/
"""
import json
import os

import pytest


RESULT_FILE = os.path.join(
    os.path.dirname(__file__), "..", "cross_arch_t5_nib_v53_results.json"
)


def load_result():
    with open(RESULT_FILE) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# NIB threshold constants (immutable — these are the published values)
# --------------------------------------------------------------------------- #
TOP5_THRESHOLD = 0.860
TOP1_THRESHOLD = 0.680
JS_THRESHOLD = 0.100
ENT_THRESHOLD = 0.350

EXPECTED_TOP5 = 0.8725
EXPECTED_TOP1 = 0.8508
EXPECTED_JS = 0.01391
EXPECTED_ENT = 0.2256


class TestResultFileExists:
    def test_result_file_present(self):
        assert os.path.isfile(RESULT_FILE), (
            f"Result file not found: {RESULT_FILE}"
        )

    def test_result_file_is_valid_json(self):
        result = load_result()
        assert isinstance(result, dict)


class TestNIBThresholds:
    """Verify that the published values pass the NIB thresholds."""

    def setup_method(self):
        self.result = load_result()

    def _get(self, key):
        # Support both flat and nested result formats
        if key in self.result:
            return self.result[key]
        for section in self.result.values():
            if isinstance(section, dict) and key in section:
                return section[key]
        raise KeyError(f"Key '{key}' not found in result file")

    def test_top5_passes_threshold(self):
        top5 = self._get("top5_agreement")
        assert top5 >= TOP5_THRESHOLD, (
            f"top5={top5:.4f} is below threshold {TOP5_THRESHOLD}"
        )

    def test_top1_passes_threshold(self):
        top1 = self._get("top1_agreement")
        assert top1 >= TOP1_THRESHOLD, (
            f"top1={top1:.4f} is below threshold {TOP1_THRESHOLD}"
        )

    def test_js_passes_threshold(self):
        js = self._get("js_divergence")
        assert js < JS_THRESHOLD, (
            f"js={js:.5f} is above threshold {JS_THRESHOLD}"
        )

    def test_ent_passes_threshold(self):
        ent = self._get("entropy_diff")
        assert ent < ENT_THRESHOLD, (
            f"ent={ent:.4f} is above threshold {ENT_THRESHOLD}"
        )


class TestPublishedValues:
    """Verify that the published constants match the result file (regression guard)."""

    def setup_method(self):
        self.result = load_result()

    def _get(self, key):
        if key in self.result:
            return self.result[key]
        for section in self.result.values():
            if isinstance(section, dict) and key in section:
                return section[key]
        raise KeyError(f"Key '{key}' not found in result file")

    def test_top5_matches_published(self):
        assert abs(self._get("top5_agreement") - EXPECTED_TOP5) < 1e-4

    def test_top1_matches_published(self):
        assert abs(self._get("top1_agreement") - EXPECTED_TOP1) < 1e-4

    def test_js_matches_published(self):
        assert abs(self._get("js_divergence") - EXPECTED_JS) < 1e-5

    def test_ent_matches_published(self):
        assert abs(self._get("entropy_diff") - EXPECTED_ENT) < 1e-4
