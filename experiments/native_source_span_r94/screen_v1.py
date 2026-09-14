"""Configure the bound shared screen for R94 native source evidence."""

from __future__ import annotations

import experiments.source_qualified_span_r93.screen_v1 as screen


screen.CAMPAIGN = "r94"
screen.BINDING_FORMAT = "abi-r94-native-source-candidate-binding/1"
screen.SOURCE_VERDICT = "PASS_R94_SOURCE"
screen.SOURCE_PASS_FIELD = "raw_passed"
screen.FAMILIES = 4
screen.FAMILY_FLOOR = 315
screen.SEED_BASE = 94_000
screen.RESULT_FORMAT = "abi-r94-native-source-prospective-screen/1"
screen.PASS_VERDICT = "PASS_R94_BOUNDED_PROSPECTIVE_TRANSFER"
screen.FAIL_VERDICT = "FAIL_R94_PROSPECTIVE_TRANSFER"


if __name__ == "__main__":
    raise SystemExit(screen.main())
