# R94 preregistration: native source-likelihood prospective transfer

R93 failed before candidate access because neutral-prior subtraction lowered
the source score from 1,358 native-correct rows to 1,309 corrected rows. R94
tests the measured scoring bottleneck on a new catalog; it does not rescore or
promote R93.

The source gate is now the frozen teacher's native mean conditional
log-likelihood over the three registered candidate strings. This is the direct
source behavior used to choose a completion. Same-list neutral-prior-corrected
scores remain recorded as a diagnostic but cannot override a correct native
ranking. The native source must pass at least 1,330/1,400 with zero native ties
and every display position represented.

R94 contains 1,400 new prompts over the four source-selected relation forms,
with five fresh outer interfaces, fresh triples, `PIVOT` subjects, seven-digit
identities beginning at 6,000,000, independent role/display rotations, and
balanced candidate-list placement. Exact prompt overlap is prohibited. R91
has never seen an R92, R93, or R94 candidate row.

Code is committed before the catalog; the catalog and scorer hash are sealed
before fresh source execution; source must pass before a binding can be
created; and the binding must be committed before the candidate executes.
Candidate gates remain 1,330/1,400, at least 315/350 per family, 95% retention
of native-source passes, +50 points over parent and random, random no higher
than 500, zero collapse, all-row sparse traces, teacher absence, and unchanged
artifacts.

A pass is bounded prospective teacher-derived two-hop reasoning transfer. It
is not broad English, arbitrary-domain, minimality, LoRA/distillation, human,
independent-hardware, or full-moonshot certification.
