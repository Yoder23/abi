# R76 selective artifact v1 consumer invalidation

Archive SHA-256
`305e75d09a64c6b3234e9554db734758ea646e2b7ad2a57cba59fd2f08b7c64b`
passed the extraction-bundle verifier but is not training eligible in practice.
The actual LayerCake v3 consumer failed closed because its evaluator and record
provenance used new `source_evidence` field names rather than the canonical
`conditional_choice_evidence` names.

The archive remains local historical evidence and must not be used for host
training. The repair changes only the package field dialect; it does not
weaken or alter the consumer, source rows, selected outputs, or source gates.
