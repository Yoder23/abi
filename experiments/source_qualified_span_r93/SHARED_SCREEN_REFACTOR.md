# R93/R94 screen reuse refactor

R93 failed at source qualification and never created a candidate binding or
executed its screen. The R93 screen implementation is therefore made reusable
through explicit module constants for campaign name, binding format, source
pass field, family count/floor, bootstrap seed, and verdict strings. Its R93
defaults remain identical. R94 binds both this shared implementation and its
small configuration wrapper before candidate execution.

This refactor does not rescore R93, alter its source failure, or authorize an
R93 candidate run.
