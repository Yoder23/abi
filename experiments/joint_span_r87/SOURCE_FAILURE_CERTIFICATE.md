# R87 source failure certificate

R87 is closed before candidate training.

- Source result file: `83f44d3f3c39caa937b8d0f6d31ee74fb97bf68929bc97d07a7fe08421fa871c`
- Raw 1,400-row evidence: `5741eeae43eea382c5dbe5856c2f34296519e7627fc573a5248dbd579f4bb3d6`
- Recomputable evidence digest: `cba74fca49de28e193b2cc8ecccf043bf154fa923020a9c6f2d2c595dd1b6f0f`
- Catalog: `d956c7ab2eb6dc4e6d099cdaf79c416751c0bf1b3601da181fbbf2ef05e07926`

The pinned Phi source passed 1,352/1,400 after prior correction, with zero
corrected ties and all display positions represented. Family counts were 174,
200, 200, 180, 200, 200, and 198. Family 0 missed the preregistered 180/200
floor, so the protocol correctly forbids R87 candidate training.

R86 and R87 both show high overall teacher performance while the same
slice-level source gate prevents testing transfer. Source scoring is a teacher
ceiling diagnostic on a candidate trained from a separate immutable artifact;
it is not itself candidate behavior. A successor must preregister an overall
teacher ceiling before source observation, keep family results diagnostic, and
retain the strict family gate for the transferred candidate. No R87 candidate
was created or observed. The full ABI moonshot remains `OPEN`.
