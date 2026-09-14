# R93 source-prerequisite failure certificate

Verdict: `FAIL_R93_SOURCE`

R93 selected four relation families using only R92 source evidence and then
scored 1,400 fresh prompts before any candidate access. The source reached
1,358/1,400 under native conditional sequence likelihood but only 1,309/1,400
after subtracting a same-list neutral candidate prior. The locked gate used
the latter and required 1,330, so R93 failed correctly.

- Catalog SHA-256: `2547b847a2768fbfa410c1b9c49ead79d5ab906b6491c320968aa11f215680cb`
- Raw source rows: `fcde27a945e7652d42fc4aeeec90a0aebb7a266358cf2c73b82c7ddb7f634d25`
- Evidence digest: `f5fe7eb9ed9d0a5e8ab48e64211c05ad1dad1c08954155d2677567d2687e5710`

The R91 candidate and LayerCake host were not loaded, no binding was created,
and no candidate row was generated. R93 is therefore not evidence against
R91. It shows that neutral-prior subtraction can reverse otherwise correct
native source rankings on this fresh interface. Any successor must register
its source score before fresh rows and cannot rescore or promote R93.

The full ABI moonshot remains `OPEN`.
