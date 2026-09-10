# R17 public compositional English-realization prerequisite

## Question

Can ABI infer a compact English surface-realization package from free-form
outputs of an unchanged open-weight teacher, then apply that package to unseen
lexical combinations with the teacher absent?

## Scientific boundary

This public prerequisite covers one registered sentence family: transitive
declaratives and questions across present, past, and future tense; positive and
negative polarity; and singular and plural subjects. The semantic frame
supplies subject, object, and the base, third-person-singular, and past verb
forms. The output sentence is never supplied in the prompt.

The source bundle presented to the compiler contains only anonymous record
IDs, feature signatures, lexical slots, and the teacher's generated text. It
contains no prompts, expected outputs, evaluators, evaluation records, package
templates, success IDs, or source model. The compiler discovers constant text
and slot placement from repeated source outputs. Its worker runs inside the
same Linux pivot-root, no-network physical isolation boundary used by R16.

Evaluation uses new subjects, verbs, and objects that do not occur in the
compiler bundle. The source evaluation outputs are captured before the source
is unloaded and before package execution. A signature-permutation control
keeps the same teacher text and slots but assigns every learned pattern to the
wrong grammatical signature.

## Public gates

- all 24 registered feature signatures have three extraction observations;
- 72/72 teacher extraction outputs match the deterministic functional oracle;
- one physically compiled English package contains all 24 consistent learned
  templates;
- 48/48 teacher evaluation outputs pass the functional oracle;
- 48/48 package outputs pass on unseen lexical combinations;
- 48/48 package outputs equal the frozen teacher evaluation outputs;
- package removal abstains on 48/48 rows;
- the permuted-signature package scores at most 10% exact;
- source training and host training are both zero;
- teacher output, prompt, token, package, runtime, and GPU-memory accounting is
  complete; and
- verification fails closed on missing, stale, swapped, or corrupt evidence.

## Claim ceiling

A public pass authorizes a preregistered hidden replication only. Even a
held-out pass would establish bounded compositional surface realization for
this registered grammar family. It would not establish unrestricted English
fluency, autonomous capability discovery, a globally minimal English core,
native neural transplantation, production LayerCake ingestion, or superiority
to LoRA or distillation.
