# R30 v7 grounded-pointer architecture

V6 executed all twelve source-discovered packages but failed quality at 79/144.
Routing was 144/144, removal was 12/12, adherence was generally high, and the
failures were concentrated in copied content: fixed-vocabulary decoding chose
training nouns or small numbers instead of the supplied evaluation values.

V7 makes one representation change. Every alphanumeric lexeme from SUPPLIED
MATERIAL that occurs in the target and exactly once in the normalized source is
encoded as a pointer action. Unlike v5, the lexeme remains available in the
fixed vocabulary, so a common digit can be fixed in one row and copied in
another. Repeated identifier declaration lines are removed from the student
view when the identifier already appears in another supplied field. Source
rows, clusters, tensor widths, steps, seed, validators, and routes are fixed.

The scorer also repairs two semantic defects discovered in v6: `Unsupported`
is a valid abstention, and clarification must retain the requested item rather
than an optional team reference. V6 remains failed under its original scorer.
V7 must match or exceed the live teacher's score under the v7 scorer, reach at
least 132/144 overall and 10/12 per task, route 144/144, and preserve all twelve
removal controls. It remains a disclosed public pilot.

