"""Run frozen R96 on disclosed R95 as development evidence only."""

from __future__ import annotations

import experiments.source_qualified_span_r93.screen_v1 as screen
from experiments.length_invariant_span_r96.core_v1 import load_package


screen.CAMPAIGN = "r96-r95-development"
screen.BINDING_FORMAT = "abi-r96-r95-development-binding/1"
screen.SOURCE_VERDICT = "PASS_R95_SOURCE_CAPTURE"
screen.SOURCE_PASS_FIELD = "raw_passed"
screen.SOURCE_QUALITY_FLOOR = 0
screen.REQUIRE_SOURCE_POINT_NONINFERIOR = True
screen.SOURCE_NONINFERIOR_CI95_FLOOR = -0.02
screen.FAMILIES = 4; screen.FAMILY_FLOOR = 315; screen.SEED_BASE = 96_000
screen.RESULT_FORMAT = "abi-r96-r95-disclosed-development-screen/1"
screen.PASS_VERDICT = "PASS_R96_R95_DEVELOPMENT_TARGETS"
screen.FAIL_VERDICT = "FAIL_R96_R95_DEVELOPMENT_TARGETS"
screen.CLAIM_BOUNDARY = "Disclosed R95 development evidence only; not prospective promotion evidence."
screen.PACKAGE_LOADER = load_package
screen.EXPECTED_RETRAINED_AFTER_R91 = True


if __name__ == "__main__": raise SystemExit(screen.main())
