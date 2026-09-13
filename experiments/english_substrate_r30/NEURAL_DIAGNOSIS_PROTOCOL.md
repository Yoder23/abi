# R30 v3 foreign-representation diagnosis pilot

V1 proved that isolated per-row free labels mostly describe prompt subjects.
V2 proved that a one-shot generative grouping prompt collapses the whole set.
Both immutable failures remain authoritative.

V3 is an exploratory public diagnosis of the source's own instruction
representations. It forwards each of the 36 unique instructions through the
unchanged Qwen2-7B-Instruct source without answers, labels, examples, or task
names. It stores mean-pooled and final-position residuals from preregistered
layers 4, 8, 12, 16, 20, 24, and 28. For each view, deterministic k-means forms
exactly twelve clusters. The chosen view is the one with the highest
unsupervised cosine silhouette score; task oracles cannot affect selection.

Only after selection are the hidden task identities joined for measurement.
The pilot passes its diagnosis prerequisite only if all twelve selected
clusters are pure, contain three instructions each, and cover every ID once.
This disclosed representation search is not a held-out certification. A pass
only authorizes freezing one representation for new instruction paraphrases.

