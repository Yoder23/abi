# R21 registered-ontology label control

R21's free-generation label interface scored 583/600 and its one authorized
restricted-logit repair scored 534/600. The second failure was 66/100 prose
rows classified as summary or bullets while every other task was 100/100.
Both failures are preserved and the teacher-side label branch is closed.

This control separates semantic labeling from generative transfer. It reuses
the original 600 teacher responses byte-for-byte and labels their extraction
instructions with ABI's deterministic word classifier, fitted from the 36
unique registered public instruction/label pairs. The classifier may read only
the instruction portion of an extraction prompt. It may not read the response,
slots, public evaluation rows, or teacher logits/activations.

The control passes only at 600/600 registered training labels and the already
frozen 120/120 public instruction-label test. Passing does not prove autonomous
ontology discovery or arbitrary-teacher labeling. It only authorizes the
unchanged frozen R21 three-method generative bakeoff, allowing that bakeoff to
falsify or support the LayerCake-native representation independently of the
closed teacher-side label interface.
