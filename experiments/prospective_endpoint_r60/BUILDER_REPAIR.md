# R60 pre-materialization builder repair

After the protocol and first builder implementation were committed, a dry
in-memory shape audit reported 1,400 rows but only 1,078 unique prompt texts.
The catalog had not been written, the teacher had not been loaded, the
candidate had not been loaded, and no model output or aggregate had been
observed.

The repair adds a capability-neutral ordinal (`Evaluation item N`) to each
prompt and makes the builder fail closed unless all 1,400 prompt texts are
unique. The ordinal does not expose the capability route, expected answer, or
evaluator. No endpoint, evaluator, threshold, or pass gate changes.

