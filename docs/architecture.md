# Architecture

ABI is a compiler and evidence system, not an inference runtime.

## Pipeline

1. **Qualify a source.** Pin a model and tokenizer revision, hash every weight
   file, record the license identifier, and reject mutable or remote-code
   sources from promotion.
2. **Survey bounded capabilities.** Execute a declared probe catalog. The
   resulting inventory describes only those probes; it is never presented as
   exhaustive teacher introspection.
3. **Label every record.** Route linguistic-form records toward the English
   core, specialist claims toward a named domain, and ambiguous material into
   quarantine.
4. **Select and budget.** The user chooses English and/or named domains. Nested
   budgets expose the marginal effect of imported teacher information.
5. **Package immutably.** Canonical serialization and SHA-256 bind sources,
   records, labels, probes, budgets, and accounting into `.abix` or `.abicir`
   acquisition artifacts.
6. **Hand off through the canonical boundary.** An external host such as
   LayerCake consumes qualified acquisition material and owns any host-side
   conformance, package installation, routing, and execution.

## Trust boundaries

- Teacher output is untrusted input until labeled and qualified.
- A classifier suggestion is not a human label unless a human attests it.
- An acquisition artifact can contain teacher material and is not a deployable
  model package.
- A LayerCake cake is a host artifact and is not owned by this repository.
- Exact byte identity, bounded functional retention, and general semantic
  equivalence are separate claims.

## English/domain separation

English-core records must use a declared linguistic capability and a
domain-minimized content basis. They cannot carry specialist labels or atomic
domain claims, and they cannot introduce facts absent from supplied context.
Specialist facts, procedures, reasoning, and code require an explicit domain
destination. Known conflicts fail closed; uncertain cases are quarantined.

These controls prove enforcement of the declared data boundary. They do not
prove that an arbitrary neural representation contains literally zero world
knowledge.

## Stable public surface

The supported import surface is exported from `abi/__init__.py`. Phase-numbered
modules are preserved research implementations and may change between alpha
releases. New production-facing behavior should enter through a small,
documented facade with focused tests rather than another versioned top-level
module.
