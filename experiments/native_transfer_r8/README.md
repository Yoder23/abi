# R8 native neural capability transfer

R8 is an additive falsification campaign. It asks whether information acquired
by source-model training can become one immutable, representation-neutral
artifact that changes several frozen recipient models through their own neural
states and logits. It does not modify or broaden R7.

The initial registered family is an opaque modular micro-language. Three
operator symbols receive independently sampled hidden meanings. A source model
must learn those meanings from examples. The extractor records only the source
model's frozen atomic neural response distributions. A generic host bridge,
trained on unrelated meta-capabilities and frozen before held-out generation,
maps that model-neutral tensor to a neural prefix. It never sees the program
being answered and cannot execute the task.

The campaign is staged:

1. build and test the fail-closed evidence machinery;
2. train/freeze generic bridges on preregistered meta capabilities;
3. reveal held-out seeds only after the freeze receipt exists;
4. train the source, extract one package, and run recipients and controls;
5. run native-state interventions and matched baselines;
6. recompute every gate from raw rows.

Until all primary gates pass across the pinned open-weight recipient families,
the controlling answer is `NOT YET ESTABLISHED`.

See `PROTOCOL.md` and `configs/preregistered_v3.json` before running anything.
The preserved v1 draft requested more unique depth-1--3 source rows than exist;
v2 records that pre-reveal feasibility correction. V1/v2 are disqualified from
use because their held-out secret appeared in a test; v3 commits a fresh secret
that is not present anywhere in the repository. No gate was changed.

## Execution order

Use a new immutable revision directory. The registered first run uses
`results/native_transfer_r8/revision_002`. Revision 001 contains only the
preserved preflight from before the prompt-length truncation defect was found.

```powershell
$config = "experiments/native_transfer_r8/configs/preregistered_v3.json"
$campaign = "results/native_transfer_r8/revision_002"

python -B -m experiments.native_transfer_r8.environment_probe --config $config --output "$campaign/preflight/environment.json"
python -B -m experiments.native_transfer_r8.train_source --config $config --output "$campaign/pre_reveal/source_public"
python -B -m experiments.native_transfer_r8.extract_capability --config $config --source-dir "$campaign/pre_reveal/source_public" --output "$campaign/pre_reveal/meta_extraction"
python -B -m experiments.native_transfer_r8.train_meta_interface --config $config --extraction-dir "$campaign/pre_reveal/meta_extraction" --campaign-root $campaign --host qwen2 --output "$campaign/pre_reveal/bridges/qwen2"
python -B -m experiments.native_transfer_r8.train_meta_interface --config $config --extraction-dir "$campaign/pre_reveal/meta_extraction" --campaign-root $campaign --host pythia --output "$campaign/pre_reveal/bridges/pythia"
python -B -m experiments.native_transfer_r8.train_meta_interface --config $config --extraction-dir "$campaign/pre_reveal/meta_extraction" --campaign-root $campaign --host t5 --output "$campaign/pre_reveal/bridges/t5"
python -B -m experiments.native_transfer_r8.freeze_campaign --root . --config $config --campaign-root $campaign
```

The held-out secret is supplied only to `reveal_heldout` after that freeze.
Do not place it in source control or a shell transcript. After reveal, run
`run_transfer`, each label-free `recipient_worker`, causal interventions,
non-interference, nested budgets, composition, and—only if native transfer is
observed—the expensive matched baselines. `verify` recomputes the verdict from
raw rows; `report` only renders that verifier output.

## Current local isolation limit

The preflight records whether Docker or Podman is actually available. A normal
subprocess is not physical filesystem isolation. Therefore a run on a host with
neither runtime cannot pass the physical-isolation gate or reach Level 2, even
if its neural transfer metrics are positive. This is a deliberate fail-closed
boundary, not an exception to the protocol.
