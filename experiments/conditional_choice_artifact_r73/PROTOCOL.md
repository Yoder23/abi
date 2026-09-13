# R73 preregistration: conditional-weight capability artifact

R72 passed its frozen teacher-weight interrogation on 2,098 of 2,100 new
nonce-reasoning prompts. R73 converts only those independently scored passing
rows into a segregated ABI training artifact. It does not describe the rows as
teacher-generated text: each target is a candidate selected by mean
conditional log probability under the pinned source weights.

The packager must fail closed unless all of these identities and conditions
recompute:

- source bundle SHA-256 `82d1ab8a3ee7b4aa351b5c74b4a229d75e845313047065780227e8e403363150`;
- source manifest SHA-256 `3bac528a1825e77dcb35963f5c78946fb14400e1cd832e0db40c2d964360c310`;
- catalog SHA-256 `d179445c92a649f5ab6587c1aa71e00e7c622c326f8ea84438497542c00c2b47`;
- R72 result-file SHA-256 `fcbdf70621bb5c5335945c6c16e7f46bc5138c077626c2734f8e1eb0925dd38c`;
- R72 raw-score SHA-256 `5c8f8a655800612f495b20b63d3bf3dcf82dbef49c5a6eb0f22659a3f47d4e4b`;
- result evidence self-hash `81f25c0725a73f719b4e242f29ec93fcd49bdc8fc212e6f6bc78e4bc64c4005f`;
- exactly 2,100 unique raw observations, 2,098 passing rows, zero ties,
  finite scores, exact catalog bindings, and positive argmax margins;
- every packaged row is search-only, linguistic-form, nonce-only,
  domain-label-free, and introduces no unsupplied real-world fact;
- authoritative post-hoc source-tokenizer counts are recorded for the selected
  outputs, while the 6,300 scalar scores and their extraction cost remain
  separately accounted;
- the v3 bundle, segregation manifest, and consumer verification all pass.

R73 is development acquisition material. A valid package authorizes a bounded
LayerCake acquisition run; it does not certify a host, general English, or the
ABI moonshot.
