# Architecture and ownership

```text
host-native prompt/token units
    ↓
generic frozen codec/conformance adapter
    ↓
canonical ABI typed context
    ↓
immutable standalone capability runtime
    ↓
canonical capability output
    ↓
same frozen codec adapter
    ↓
host-native token units / strict UTF-8
```

The capability owns learned semantics. The ABI owns integrity, lifecycle, and
canonical state. The named host environment owns checkpoint conformance probes
and tokenizer units. The base model does not generate the promoted answer.

Evidence: `results/abi_final_validation/host_causality.json`.
