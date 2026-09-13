# R30 v5 global copy-eligibility repair

V4 stopped after seven immutable packages when the token `2`, excluded from a
package vocabulary because it was copied in one row, appeared in another target
where it was not uniquely copyable. No LayerCake evaluation occurred.

V5 changes only pointer annotation. A candidate nonce or numeric lexeme is
excluded from the fixed vocabulary only when every target row containing that
lexeme also contains it exactly once in the source. Every row additionally
declares the source-only `INSTRUCTION` sentinel, satisfying the generic
tokenizer contract without affecting any target. Architecture, source rows,
clusters, validators, training schedule, scorer, and gates are unchanged.

