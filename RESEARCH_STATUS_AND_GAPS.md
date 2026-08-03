# ABI research status and open questions

Status date: 2026-08-03

## What the research has resolved

1. **The repositories have distinct responsibilities.** ABI extraction and
   LayerCake execution are separate systems and evidence lineages.
2. **The bounded pipeline is real.** The historical v47 release demonstrates
   extraction, packaging, teacher removal, bounded domain installation, and a
   complete locked-suite execution path.
3. **The old bounded catalog did not generalize.** The exact v47 candidate
   scored 0/28 on novel prompt forms while frozen Phi-3 scored 19/28.
4. **Teacher data can causally affect LayerCake.** V87's real mapping beat the
   parent and shuffled control, although the formal quality gate failed.
5. **Pre-transfer segregation can work.** V89 passed its locked, disjoint
   English/domain/quarantine labeling benchmark with zero specialist leakage
   into English.
6. **Transport is not the present blocker.** Exact package transport and
   receiver-path identity are established in LayerCake. The open ABI problem
   is acquiring and conditioning enough correct, diverse information for
   natural generalization.

## Immediate blocking questions

### 1. What is the normalized acquisition unit?

We have labeled teacher prompt/response records, but not a final contract for
turning them into training-ready examples. The next protocol must decide which
information is preserved verbatim, transformed, paired contrastively,
decomposed, or rejected. Every transformation must remain reversible to raw
provenance and must not silently alter the destination label.

### 2. What belongs in the English substrate?

Grammar, coherence, grounding, instructions, conversation, supplied-text
summarization, rewriting, email drafting, tone, formatting, clarification,
abstention, and fluent realization are intended English capabilities.

“Domain-independent reasoning” remains underspecified. Historical examples
mixed linguistic reasoning with arithmetic, which would contaminate the
English-only definition. The next contract must define fact-free reasoning
tasks or assign them to a separate capability/domain.

### 3. How much breadth is adequate before training?

Record count alone is insufficient. Adequacy must cover natural surface forms,
behavioral families, response structures, lengths, interaction styles,
negative cases, clarification, abstention, and adversarial variants. It must
also require teacher-output correctness and completion, not merely nonempty
text.

### 4. How do we avoid teaching templates instead of English?

The 1,700-case v47 pass followed by a 0/28 novel-form result is direct evidence
of template overfitting. The successor needs paraphrase families, compositional
variation, unseen lexical combinations, varied discourse lengths, and a fixed
natural holdout that cannot influence corpus construction.

### 5. What is the sufficient-information frontier?

The smallest broadly fluent imported-information budget is unknown. Data and
model size are coupled: a small corpus may fail because it is insufficient,
while a small host may fail despite adequate data. Nested data budgets and
nested host-size controls must separate these factors.

### 6. Can LayerCake express the expanded target behavior?

Earlier same-topology and matched-transformer controls did not establish broad
quality at the available information budget. Future failures must be separated
among artifact inadequacy, acquisition, host capacity, and decoding. A new ABI
candidate cannot be blamed on LayerCake unless an appropriate native or oracle
payload passes the identical expanded suite.

### 7. What quality comparison closes the English goal?

Exact template checks are necessary but insufficient. Promotion needs paired
teacher-versus-candidate evaluation on untouched natural prompts, deterministic
functional checks where possible, blinded rubric or human evaluation for open
generation, repetition/collapse metrics, and prompt-level uncertainty.

### 8. How is domain discovery bounded?

V89 classifies a declared ontology; it does not discover every latent domain.
The project still needs a user-governed process for proposing domains,
qualifying their boundaries, detecting multi-domain records, and deciding when
unknown material remains quarantined.

### 9. How are multiple teachers reconciled?

Multi-source acquisition needs record-level provenance, source licenses,
confidence, contradiction detection, preference rules, and a fail-closed
conflict policy. Agreement between teachers is evidence, not an automatic
semantic label.

### 10. What is the release-quality evidence depth?

V89 has 20 disjoint observations per class. Its point estimates pass, but the
per-class Wilson lower bounds remain wide. Broader labeling validation, three
independent runs where stochasticity matters, and external reproduction are
needed before claiming population-level reliability.

### 11. How do licensing and deletion propagate?

Every normalized record and derived artifact needs source/license identity and
a deletion lineage. The project must be able to determine which artifacts are
affected when a source, record, or permission is withdrawn.

### 12. How is the current research ledger released cleanly?

The working tree contains a large preserved experiment history. Before a new
public ABI release, code, small evidence, large local assets, ignored artifacts,
and historical failures need an explicit manifest and versioned publication
policy. A passing local experiment is not automatically a clean release.

## Research priority

The next useful work is not another transfer recipe or domain sweep. It is
Phase 0 of `ABI_CAPABILITY_COMPILER_CAMPAIGN_V1.md`:

1. freeze strong matched LoRA and distillation baselines;
2. freeze equal-information, equal-deployment, and matched-quality comparisons;
3. preregister numeric margins, statistics, splits, seeds, and stop rules; and
4. define which complete product advantages would justify ABI over conventional
   methods.

Only then proceed to normalization and adequacy:

1. preregister the schema and transformations;
2. build immutable English and per-domain inventories;
3. qualify correctness, completion, diversity, provenance, and leakage;
4. freeze untouched natural generalization data; and
5. only then begin nested information-budget training.

See `CURRENT_PROJECT_STATUS.md` for the claim ledger and phase boundary.
