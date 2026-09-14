"""Execute frozen R96 on prospective R97."""

from __future__ import annotations
import experiments.source_qualified_span_r93.screen_v1 as screen
from experiments.length_invariant_span_r96.core_v1 import load_package

screen.CAMPAIGN="r97"; screen.BINDING_FORMAT="abi-r97-prospective-candidate-binding/1"; screen.SOURCE_VERDICT="PASS_R97_SOURCE_CAPTURE"; screen.SOURCE_PASS_FIELD="raw_passed"; screen.SOURCE_QUALITY_FLOOR=0; screen.REQUIRE_SOURCE_POINT_NONINFERIOR=True; screen.SOURCE_NONINFERIOR_CI95_FLOOR=-0.02; screen.FAMILIES=4; screen.FAMILY_FLOOR=315; screen.SEED_BASE=97_000; screen.RESULT_FORMAT="abi-r97-prospective-length-span-screen/1"; screen.PASS_VERDICT="PASS_R97_BOUNDED_PROSPECTIVE_TRANSFER"; screen.FAIL_VERDICT="FAIL_R97_PROSPECTIVE_TRANSFER"; screen.CLAIM_BOUNDARY="Bounded prospective length/structure-invariant two-hop reasoning transfer only."; screen.PACKAGE_LOADER=load_package; screen.EXPECTED_RETRAINED_AFTER_R91=True

if __name__ == "__main__": raise SystemExit(screen.main())
