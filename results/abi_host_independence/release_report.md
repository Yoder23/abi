# ABI host-independence release report

**ABI MOONSHOT STATUS: TECHNICALLY PROVEN — EXTERNAL VALIDATION PENDING**

Technical declaration: **ABI TECHNICAL MOONSHOT: PROVEN**

1. **Winning family:** Family A, observable canonical state.
2. **Canonical semantics:** typed instruction/task/topic/entity/relation/constraint/conversation/uncertainty/output state, capability lifecycle, and exact UTF-8 anchors.
3. **LayerCake adapter:** one capability-blind, frozen, 1,556-byte lifecycle/codec binding.
4. **Qwen adapter:** one capability-blind, frozen, 1,637-byte Qwen2 tokenizer/lifecycle binding.
5. **Pythia adapter:** one capability-blind, frozen, 1,646-byte GPT-NeoX tokenizer/lifecycle binding.
6. **Adapter parameters:** 0 for every host; 0 optimizer steps; 0 capability-specific parameters.
7. **Certification cost:** LayerCake 0.0397925 s, Qwen 9.1626161 s, Pythia 6.1848264 s; exposure was 128 domain-neutral examples and 5,953 unique UTF-8 bytes per host.
8. **Capability blindness:** package paths were not supplied, package open attempts were zero, domain examples/outputs/success IDs seen were zero, and package suffixes were denied.
9. **Adapter identity:** before/after hashes are identical: LayerCake `d1f3a9d...17f04`, Qwen `b13a75b8...291f`, Pythia `df3598b6...eafa`.
10. **English:** 1,381/1,381 frozen successes retained per host; 4,143/4,143 receiver successes total, byte exact.
11. **Python:** 100/100 per host; 300/300 total, byte exact.
12. **Chemistry:** 100/100 per host; 300/300 total, byte exact.
13. **Civics:** 100/100 per host; 300/300 total, byte exact.
14. **Mathematical equality:** 1,681/1,681 canonical contexts, output intents, and output bytes match across hosts; 300 specialist action sequences match.
15. **Semantic retention:** 5,043/5,043 receiver source successes retained; invalid-output increase 0; aggregate non-inferiority passes by exact source-output identity.
16. **Isolation:** English specialist leakage 0/900; wrong capability 0/1,200; 12/12 removals failed closed and exact reinstall restored; 24/24 random/shuffled packages were rejected; 3/3 adapter removals failed closed.
17. **Teacher absence:** teacher and source model were false in all three final host results; installation trained and calibrated nothing.
18. **Runtime:** 20 paired observations per host; adapter overhead LayerCake 3.9615%, Qwen -0.5511%, Pythia 3.1223%, all below 10%. TTFT is conservatively equal to non-streaming total capability latency.
19. **Reuse economics:** certification is one-time and installation is verify plus load. Exact 1/2/4-package costs are recorded. No matched LoRA/distillation/fine-tuning experiment exists under this protocol, so no quantitative superiority is claimed.
20. **Hostile verifier:** all 14 registered mutation classes pass, including evidence/package/adapter hash mutation, forbidden fitting, corrupt packages, removal, and wrong-capability controls.
21. **Human status:** `EXTERNAL_HUMAN_VALIDATION_PENDING`; 0/3 raters and 0/21,000 judgments.
22. **Hardware status:** `EXTERNAL_HARDWARE_VALIDATION_PENDING`; the clean-room archive is locally verified but has no independent different-hardware execution.
23. **Minimum-information status:** deferred and uncertified; no global minimum claim.
24. **Seal:** V1 failure tag `abi-v1-host-independence-failure-2026-08-24`; technical seal tag `abi-host-independence-technical-proof-2026-08-24`.
25. **Reproduction:** `python -m abi_v2.verify_release --check-existing`, `python -m abi_v2.verify_host_independence --check-existing`, and `pytest -q`.

## Boundary

This proves a representation-neutral extension/runtime ABI for the three named
hosts and four named packages. Qwen and Pythia base checkpoints participate in
native conformance probes and exact native-tokenizer realization, but their
hidden states do not create or alter capability semantics. This is not a claim
of tensor transplantation, arbitrary-model support, human preference, external
reproduction, or a global information minimum.
