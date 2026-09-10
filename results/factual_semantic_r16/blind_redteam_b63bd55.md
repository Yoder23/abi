# Blind R16 red-team report

Reviewed commit:
`b63bd55301d41f4bc352d250cef065f1450f0ef9`

Verdict: `PASS` for only
`BOUNDED_FACTUAL_EXTRACTION_AND_SEMANTIC_SEGREGATION`.

Severity counts: Critical 0, High 0, Medium 3, Low 3.

The full ABI moonshot remains open.

## Independently reproduced

- Linear chronology was confirmed: implementation freeze `b1a3904`,
  preregistration `b8e97ba`, reveal `bd5e571`, evidence seal `32b0595`, and
  documentation `b63bd55`.
- All 14 frozen core hashes, the secret commitment, and the reveal hash
  matched.
- The pinned Qwen snapshot rehashed exactly: 10 files and 15,242,778,262
  bytes, evidence SHA-256
  `bdb164548ca29e580e465f520df01fce77524eeec01e807c413443751b4f004b`.
- Strict-v2, hostile-v2, and seal-v3 independently recomputed; hostile
  verification rejected 15/15 mutations; 9/9 focused tests passed.
- A fresh pinned-Qwen in-memory replay completed in 65.005 seconds and matched
  all 96 source rows, all 48 shuffled compiler records, the complete
  `[48, 3584]` residual tensor, 576 candidate scores, 276 generated tokens,
  and 372 output bytes exactly.
- Independent row/package recomputation found 0/96 question-answer text leaks,
  96/96 exact source completions, one correct candidate in each of 48/48
  closed candidate sets, 48/48 correct score argmax selections, 48/48 residual
  hashes, 48/48 source-bundle links, and zero explicit forbidden fields.
- Both ordinary and rotated-score packages reconstructed exactly. All 7/7
  actual live files, 4/4 mount/result/launcher chains, package/runtime metrics,
  namespace isolation, removal, and causal controls were independently
  confirmed.

## Findings

### Medium

1. The v1 accounting field `unique_prompt_utf8_bytes=4,126` counted bare user
   questions, not the rendered chat prompts. The actual 96 rendered prompts
   total 18,334 UTF-8 bytes and 3,206 input tokens.
2. The historical public-v2 receipt declares protocol SHA-256 `4e5a60e1...`,
   but no committed `PUBLIC_PROTOCOL.md` has that digest. The current file is
   `4b62aba1...`. Public rows, residuals, artifacts, fact registry, and metrics
   independently recomputed exactly, so this is a provenance deficiency, not
   a blocker for the bounded held-out result.
3. “Answers absent” needs precise wording. There is no labeled answer/oracle
   field, completion, fact ID, secret, or reveal in the compiler capsule, but
   the correct answer string is present exactly once in every registered
   12-candidate set and teacher argmax scores identify it.

### Low

1. Strict-v2 validated the stored live receipt but did not reopen all actual
   live files or validate the physical live trees. The reviewer independently
   closed this locally.
2. R16 bound the source snapshot indirectly through R15B rather than directly
   rehashing it in the R16 strict verifier. The reviewer independently rehashed
   it exactly.
3. Strict-v2, hostile-v2, and seal-v3 were retrospective additions after the
   reveal. The scientific runner, base verifier, isolation mechanism, gates,
   and inputs remained frozen.

## Claim boundary

R16 proves factual selection and structured packaging only for the
preregistered 16-fact, two-domain, closed-candidate workload with a registered
keyword ontology. It does not prove autonomous factual/domain discovery,
fluent English transfer, arbitrary-domain extraction, native neural
transplantation, production LayerCake ingestion, minimality, or superiority to
LoRA/distillation.

The repository subsequently addressed the findings additively; it did not
rewrite this reviewed commit or the original evidence.
