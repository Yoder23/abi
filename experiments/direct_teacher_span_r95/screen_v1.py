"""Configure the shared prospective screen for direct paired R95 comparison."""

from __future__ import annotations

import experiments.source_qualified_span_r93.screen_v1 as screen


screen.CAMPAIGN = "r95"
screen.BINDING_FORMAT = "abi-r95-direct-teacher-candidate-binding/1"
screen.SOURCE_VERDICT = "PASS_R95_SOURCE_CAPTURE"
screen.SOURCE_PASS_FIELD = "raw_passed"
screen.SOURCE_QUALITY_FLOOR = 0
screen.REQUIRE_SOURCE_POINT_NONINFERIOR = True
screen.SOURCE_NONINFERIOR_CI95_FLOOR = -0.02
screen.FAMILIES = 4
screen.FAMILY_FLOOR = 315
screen.SEED_BASE = 95_000
screen.RESULT_FORMAT = "abi-r95-direct-teacher-prospective-screen/1"
screen.PASS_VERDICT = "PASS_R95_BOUNDED_DIRECT_TEACHER_TRANSFER"
screen.FAIL_VERDICT = "FAIL_R95_DIRECT_TEACHER_TRANSFER"
screen.CLAIM_BOUNDARY = "Bounded prospective direct paired teacher-derived two-hop reasoning transfer only."


if __name__ == "__main__":
    raise SystemExit(screen.main())
