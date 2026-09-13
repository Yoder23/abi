# R30 v6 direct-host manifest repair

V5 trained all twelve packages and then failed before the first generation
because the package input contract omitted LayerCake's required
`mode=direct_selected_portable_decoder` declaration. V6 performs no training
and changes no tensor, tokenizer, architecture, source row, route, scorer, or
gate. It loads each signed v5 archive, proves its bound hash, and rebuilds the
archive with only that input-contract field added. Evaluation then starts in a
fresh host registry under a new immutable result path.

