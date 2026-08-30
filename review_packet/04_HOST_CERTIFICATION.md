# Host certification

Each declared environment is certified in a filesystem containing only the
generic corpus, selected host, ABI specification, and adapter code. Capability
archives and success IDs are physically absent.

| Environment | Inventory rows |
| --- | ---: |
| LayerCake v25 | 100,511 |
| Qwen2.5-0.5B | 100,517 |
| Pythia-160M | 100,515 |

R7 content-scans every reachable non-virtual regular file, recursively expands
supported archives, and recognizes checksum-valid USTAR/GNU/V7 tar streams even
behind arbitrary byte prefixes. Exact raw inventories are retained under
`isolated_certification_strict_r7_tar_bound/`.
