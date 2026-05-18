"""
succession_test_v2.py  —  Layercake ABI Succession Test v2
===========================================================
Same as succession_test.py with one architectural fix:
  proj_in is FROZEN during backbone updates (not just proj_out).

Rationale: when proj_in is allowed to adapt during backbone evolution,
it shifts the ABI coordinate frame so M_A's domain modules misfire even
though cosine similarity remains high.  Freezing proj_in forces the
backbone to maintain ABI compatibility in a FIXED projection space,
which is what the zero-shot portability claim actually requires.

Tests: zero-shot domain transfer across 3 successive backbone updates,
two domain types (Python + Markdown), GPT-2-medium (354M), WikiText-2.
"""

import math, time, copy, json, pathlib, random
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# ─── Hyperparams ─────────────────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_ABI        = 256
BATCH_SIZE   = 8
MAX_SEQ_LEN  = 128
DOMAIN_STEPS = 500           # unchanged from all prior tests
UPDATE_STEPS = 1000          # per backbone round
NUM_ROUNDS   = 3             # → 3000 total update steps
BACKBONE_LR  = 5e-5
ABI_LR       = 3e-4
ALPHA_STAB   = 1.0
SEED         = 42

random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

ROOT = pathlib.Path(__file__).parent.parent


# ─── Data ────────────────────────────────────────────────────────────────────
class TokenDataset(Dataset):
    def __init__(self, ids, seq_len):
        n = (len(ids) // seq_len) * seq_len
        self.data = ids[:n].reshape(-1, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def make_loader(ids, batch_size=BATCH_SIZE, seq_len=MAX_SEQ_LEN, shuffle=True):
    return DataLoader(
        TokenDataset(ids, seq_len),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )


# ─── Model ───────────────────────────────────────────────────────────────────
class DomainModule(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.LayerNorm(d * 2),
            nn.Linear(d * 2, d), nn.LayerNorm(d),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h):
        return h + torch.sigmoid(self.gate) * self.net(h)


class ABIModel(nn.Module):
    def __init__(self):
        super().__init__()
        g             = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        dm            = g.config.n_embd          # 1024
        self.backbone = g.transformer
        self.lm_head  = g.lm_head
        self.proj_in  = nn.Linear(dm, D_ABI, bias=False)
        self.abi_ln   = nn.LayerNorm(D_ABI)
        self.proj_out = nn.Linear(D_ABI, dm, bias=False)
        self.domains  = nn.ModuleDict()
        self.dm       = dm

    def add_domain(self, name):
        dev = next(self.parameters()).device
        self.domains[name] = DomainModule(D_ABI).to(dev)

    def forward(self, ids, domain=None):
        h     = self.backbone(ids).last_hidden_state
        h_abi = self.abi_ln(self.proj_in(h))
        if domain and domain in self.domains:
            h_abi = self.domains[domain](h_abi)
        return self.lm_head(h + self.proj_out(h_abi))


# ─── Loss / eval ─────────────────────────────────────────────────────────────
def lm_loss(logits, ids):
    B, T, V = logits.shape
    return nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, V),
        ids[:, 1:].reshape(-1),
    )


@torch.no_grad()
def compute_ppl(model, loader, domain=None, max_batches=30):
    model.eval()
    tot, n = 0.0, 0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        b = b.to(DEVICE)
        tot += lm_loss(model(b, domain=domain), b).item()
        n += 1
    return math.exp(tot / n) if n else float("inf")


# ─── Training ────────────────────────────────────────────────────────────────
def train_domain(model, loader, dom, steps):
    """Freeze backbone + lm_head + proj_out; train proj_in, abi_ln, domain."""
    for p in model.parameters():
        p.requires_grad_(False)
    for nm, p in model.named_parameters():
        if "proj_in" in nm or "abi_ln" in nm or nm.startswith(f"domains.{dom}"):
            p.requires_grad_(True)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=ABI_LR
    )
    model.train()
    it = iter(loader)
    for _ in range(steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        b = b.to(DEVICE)
        opt.zero_grad()
        lm_loss(model(b, domain=dom), b).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def update_backbone_frozen_proj(model_cur, model_anchor, loader, steps, rnd):
    """Fine-tune BACKBONE ONLY on loader with ABI stability anchored to model_anchor.

    KEY DIFFERENCE vs v1: proj_in is ALSO FROZEN (only backbone updated).
    This forces the backbone to maintain ABI-space compatibility in M_A's
    fixed coordinate frame, keeping pasted domain modules valid indefinitely.
    proj_out and lm_head remain frozen (same as v1).
    """
    for p in model_cur.parameters():
        p.requires_grad_(False)
    for nm, p in model_cur.named_parameters():
        if "backbone" in nm:         # ONLY backbone — NOT proj_in
            p.requires_grad_(True)

    opt = torch.optim.AdamW(
        [p for p in model_cur.parameters() if p.requires_grad],
        lr=BACKBONE_LR,
    )
    model_cur.train()
    model_anchor.eval()
    it = iter(loader)

    for step in range(1, steps + 1):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        b = b.to(DEVICE)
        opt.zero_grad()

        # Single forward pass: proj_in is frozen so no grad flows through it,
        # but gradient through backbone is still computed via chain rule:
        # loss → h_abi → (frozen proj_in) → backbone output → backbone params
        h     = model_cur.backbone(b).last_hidden_state
        h_abi = model_cur.abi_ln(model_cur.proj_in(h))   # proj_in frozen, no grad there
        h_out = h + model_cur.proj_out(h_abi)
        logits = model_cur.lm_head(h_out)

        ll = lm_loss(logits, b)

        # ABI stability: compare to original M_A representations
        with torch.no_grad():
            h_anc   = model_anchor.backbone(b).last_hidden_state
            h_abi_a = model_anchor.abi_ln(model_anchor.proj_in(h_anc))

        sl = nn.functional.mse_loss(h_abi, h_abi_a)
        (ll + ALPHA_STAB * sl).backward()
        nn.utils.clip_grad_norm_(model_cur.parameters(), 1.0)
        opt.step()

        if step % 200 == 0:
            print(
                f"    R{rnd} step {step:4d}/{steps}: "
                f"lm={ll.item():.3f}  stab={sl.item():.4f}"
            )


def paste_domain(src_model, dst_model, dom):
    dst_model.domains[dom].load_state_dict(
        copy.deepcopy(src_model.domains[dom].state_dict())
    )


# ─── Test registry ───────────────────────────────────────────────────────────
_results = []


def record(name, passed, note=""):
    status = "PASS" if passed else "FAIL"
    _results.append({"name": name, "status": status, "note": note})
    print(f"  [{status}] {name}")
    if note:
        print(f"       {note}")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    t0  = time.time()
    SEP = "═" * 68

    print(f"\n{SEP}")
    print("  LAYERCAKE ABI SUCCESSION TEST v2")
    print(f"  GPT-2-medium (354M) | WikiText-2 (3 chunks) | {NUM_ROUNDS}×{UPDATE_STEPS} steps")
    print("  FIX: proj_in FROZEN during backbone update (preserves ABI coordinate frame)")
    print(f"{SEP}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("  [Data] Loading tokenizer + corpora...")
    t1  = time.time()
    tok = GPT2TokenizerFast.from_pretrained("gpt2-medium")
    tok.pad_token = tok.eos_token

    from datasets import load_dataset
    wiki_raw = "\n".join(
        r["text"]
        for r in load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        if r["text"].strip()
    )

    py_parts, py_chars = [], 0
    for p in ROOT.rglob("*.py"):
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
            py_parts.append(t); py_chars += len(t)
            if py_chars >= 600_000:
                break
        except Exception:
            continue
    pyth_raw = "\n".join(py_parts)

    md_parts, md_chars = [], 0
    for p in ROOT.rglob("*.md"):
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
            md_parts.append(t); md_chars += len(t)
            if md_chars >= 600_000:
                break
        except Exception:
            continue
    md_raw = "\n".join(md_parts)

    wiki_ids = tok(wiki_raw, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
    pyth_ids = tok(pyth_raw, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
    md_ids   = tok(md_raw,   return_tensors="pt", truncation=False)["input_ids"].squeeze(0)

    chunk        = len(wiki_ids) // NUM_ROUNDS
    wiki_loaders = [make_loader(wiki_ids[r * chunk : (r + 1) * chunk]) for r in range(NUM_ROUNDS)]
    pyth_loader  = make_loader(pyth_ids)
    md_loader    = make_loader(md_ids)

    print(
        f"  [Data] {time.time()-t1:.1f}s | WikiText-2={len(wiki_ids):,} | "
        f"Python={len(pyth_ids):,} | Markdown={len(md_ids):,} tokens"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n  [Model] Loading pretrained gpt2-medium...")
    t1 = time.time()
    model_A = ABIModel().to(DEVICE)
    model_A.add_domain("python")
    model_A.add_domain("markdown")
    n_bb = sum(p.numel() for p in model_A.backbone.parameters())
    print(
        f"  [Model] {time.time()-t1:.1f}s | backbone={n_bb/1e6:.1f}M | "
        f"d_model={model_A.dm} | d_abi={D_ABI}"
    )

    # ── STEP 1: Train both domain modules on frozen M_A ──────────────────────
    print(f"\n  ── STEP 1: Domain module training ({DOMAIN_STEPS} steps each, backbone FROZEN) ──")
    t1 = time.time()

    ppl_py_pre = compute_ppl(model_A, pyth_loader, domain="python")
    ppl_md_pre = compute_ppl(model_A, md_loader,   domain="markdown")

    train_domain(model_A, pyth_loader, "python",   DOMAIN_STEPS)
    train_domain(model_A, md_loader,   "markdown", DOMAIN_STEPS)

    ppl_py_A = compute_ppl(model_A, pyth_loader, domain="python")
    ppl_md_A = compute_ppl(model_A, md_loader,   domain="markdown")

    py_s1 = (ppl_py_pre - ppl_py_A) / ppl_py_pre * 100
    md_s1 = (ppl_md_pre - ppl_md_A) / ppl_md_pre * 100

    print(f"  Step 1 in {time.time()-t1:.1f}s")
    print(f"  Python:   PPL {ppl_py_pre:.1f} → {ppl_py_A:.1f}  (+{py_s1:.1f}%)")
    print(f"  Markdown: PPL {ppl_md_pre:.1f} → {ppl_md_A:.1f}  (+{md_s1:.1f}%)")

    record(
        "ST1_both_domains_on_pretrained",
        py_s1 > 3.0 and md_s1 > 3.0,
        f"Python +{py_s1:.1f}% | Markdown +{md_s1:.1f}% | both >3%: {py_s1>3.0 and md_s1>3.0}",
    )

    # Freeze M_A permanently
    model_A.eval()
    for p in model_A.parameters():
        p.requires_grad_(False)

    # ── STEP 2: Successive backbone updates + zero-shot paste ─────────────────
    print(
        f"\n  ── STEP 2: {NUM_ROUNDS} successive backbone rounds ({UPDATE_STEPS} steps each) ──"
    )
    print(
        "  (proj_in + proj_out + lm_head FROZEN | backbone only | "
        "ABI stability anchored to M_A)"
    )

    curve     = []
    model_cur = copy.deepcopy(model_A)

    for rnd in range(1, NUM_ROUNDS + 1):
        cum = rnd * UPDATE_STEPS
        print(
            f"\n  ── Round {rnd}/{NUM_ROUNDS}: WikiText-2 chunk {rnd}  "
            f"({cum} cumulative steps) ──"
        )
        t1 = time.time()

        update_backbone_frozen_proj(
            model_cur, model_A, wiki_loaders[rnd - 1], UPDATE_STEPS, rnd
        )

        # Zero-shot paste from M_A
        paste_domain(model_A, model_cur, "python")
        paste_domain(model_A, model_cur, "markdown")

        ppl_py_nd = compute_ppl(model_cur, pyth_loader, domain=None)
        ppl_md_nd = compute_ppl(model_cur, md_loader,   domain=None)
        ppl_py_zs = compute_ppl(model_cur, pyth_loader, domain="python")
        ppl_md_zs = compute_ppl(model_cur, md_loader,   domain="markdown")

        py_zs = (ppl_py_nd - ppl_py_zs) / ppl_py_nd * 100
        md_zs = (ppl_md_nd - ppl_md_zs) / ppl_md_nd * 100

        # ABI cosine similarity vs M_A (through the SAME frozen proj_in)
        with torch.no_grad():
            sb    = next(iter(pyth_loader)).to(DEVICE)[:4]
            h_A   = model_A.abi_ln(model_A.proj_in(model_A.backbone(sb).last_hidden_state))
            h_B   = model_cur.abi_ln(model_cur.proj_in(model_cur.backbone(sb).last_hidden_state))
            cs    = nn.functional.cosine_similarity(
                h_A.reshape(-1, D_ABI), h_B.reshape(-1, D_ABI), dim=-1
            ).mean().item()
            cs_mult = abs(cs) / (1.0 / math.sqrt(D_ABI))

        print(
            f"  Round {rnd} in {time.time()-t1:.1f}s | "
            f"ABI cos_sim={cs:.4f} ({cs_mult:.0f}× rand)"
        )
        print(f"  Python:   no-domain {ppl_py_nd:.1f} → zero-shot {ppl_py_zs:.1f} (+{py_zs:.1f}%)")
        print(f"  Markdown: no-domain {ppl_md_nd:.1f} → zero-shot {ppl_md_zs:.1f} (+{md_zs:.1f}%)")

        curve.append(
            dict(
                round=rnd, steps=cum,
                py_zs=py_zs, md_zs=md_zs,
                py_ppl=ppl_py_zs, md_ppl=ppl_md_zs,
                cos_sim=cs, cos_mult=cs_mult,
            )
        )

        record(
            f"ST2_R{rnd}_python",
            py_zs > 3.0,
            f"After {cum} steps: Python zero-shot +{py_zs:.1f}% (>3%)",
        )
        record(
            f"ST3_R{rnd}_markdown",
            md_zs > 3.0,
            f"After {cum} steps: Markdown zero-shot +{md_zs:.1f}% (>3%)",
        )

    # ── STEP 3: Cold-start baseline at final checkpoint ───────────────────────
    print(f"\n  ── STEP 3: Cold-start baseline at round {NUM_ROUNDS} ({NUM_ROUNDS*UPDATE_STEPS} steps) ──")
    t1 = time.time()

    model_cs = copy.deepcopy(model_cur)
    model_cs.add_domain("py_cs")
    model_cs.add_domain("md_cs")

    pre_py_cs = compute_ppl(model_cs, pyth_loader, domain="py_cs")
    pre_md_cs = compute_ppl(model_cs, md_loader,   domain="md_cs")

    train_domain(model_cs, pyth_loader, "py_cs", DOMAIN_STEPS)
    train_domain(model_cs, md_loader,   "md_cs", DOMAIN_STEPS)

    ppl_py_cs = compute_ppl(model_cs, pyth_loader, domain="py_cs")
    ppl_md_cs = compute_ppl(model_cs, md_loader,   domain="md_cs")

    py_cs = (pre_py_cs - ppl_py_cs) / pre_py_cs * 100
    md_cs = (pre_md_cs - ppl_md_cs) / pre_md_cs * 100

    py_eff = curve[-1]["py_zs"] / py_cs * 100 if py_cs > 0 else 0.0
    md_eff = curve[-1]["md_zs"] / md_cs * 100 if md_cs > 0 else 0.0

    print(f"  Cold-start in {time.time()-t1:.1f}s")
    print(f"  Python native +{py_cs:.1f}% | zero-shot +{curve[-1]['py_zs']:.1f}% | efficacy {py_eff:.1f}%")
    print(f"  Markdown native +{md_cs:.1f}% | zero-shot +{curve[-1]['md_zs']:.1f}% | efficacy {md_eff:.1f}%")

    record(
        "ST4_python_efficacy_final",
        py_eff >= 40.0,
        f"Python efficacy at {NUM_ROUNDS*UPDATE_STEPS} steps: {py_eff:.1f}% (≥40%)",
    )
    record(
        "ST5_markdown_efficacy_final",
        md_eff >= 40.0,
        f"Markdown efficacy at {NUM_ROUNDS*UPDATE_STEPS} steps: {md_eff:.1f}% (≥40%)",
    )

    all_py = all(r["py_zs"] > 3.0 for r in curve)
    all_md = all(r["md_zs"] > 3.0 for r in curve)
    record(
        "ST6_succession_consistent",
        all_py and all_md,
        f"Python all {NUM_ROUNDS} rounds: {all_py} | Markdown all {NUM_ROUNDS} rounds: {all_md}",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pass  = sum(1 for r in _results if r["status"] == "PASS")
    n_total = len(_results)
    elapsed = time.time() - t0

    print(f"\n{SEP}")
    print("  SUCCESSION TEST v2 — FINAL RESULTS")
    print(SEP)
    for r in _results:
        print(f"  [{r['status']}] {r['name']}")

    print(f"\n  {n_pass}/{n_total} tests PASSED  ({elapsed:.0f}s total)")

    print()
    print("  ┌─────────────┬──────────────────┬──────────────────┬────────────┐")
    print("  │ Cum. Steps  │ Python zero-shot │ Markdown zero-shot│ ABI ×rand  │")
    print("  ├─────────────┼──────────────────┼──────────────────┼────────────┤")
    for r in curve:
        print(
            f"  │    {r['steps']:5d}    │   +{r['py_zs']:5.1f}%        │"
            f"   +{r['md_zs']:5.1f}%        │  {r['cos_mult']:6.0f}×    │"
        )
    print("  └─────────────┴──────────────────┴──────────────────┴────────────┘")

    print(f"\n  Transfer efficacy at {NUM_ROUNDS * UPDATE_STEPS} cumulative steps:")
    print(f"    Python:   {py_eff:.1f}%")
    print(f"    Markdown: {md_eff:.1f}%")

    min_eff = min(py_eff, md_eff)
    if n_pass == n_total and min_eff >= 70:
        claim = "BREAKTHROUGH-LEVEL — both domains, 3 successive rounds, ≥70% efficacy"
    elif n_pass == n_total and min_eff >= 50:
        claim = "STRONG — both domains survive 3 successive rounds, ≥50% efficacy"
    elif n_pass >= n_total - 1 and min_eff >= 40:
        claim = "STRONG — succession demonstrated with minor limitation"
    elif n_pass >= n_total - 2:
        claim = f"PARTIAL — {n_pass}/{n_total} pass, efficacy {min_eff:.1f}%"
    else:
        claim = f"INCOMPLETE — {n_pass}/{n_total} pass"

    print(f"\n  Claim level: {claim}")

    out = {
        "test": "succession_test_v2",
        "variant": "proj_in_frozen_during_backbone_update",
        "backbone": "gpt2-medium (354.8M pretrained)",
        "update_corpus": "WikiText-2 (3 equal chunks)",
        "domains": ["python", "markdown"],
        "num_rounds": NUM_ROUNDS,
        "steps_per_round": UPDATE_STEPS,
        "total_update_steps": NUM_ROUNDS * UPDATE_STEPS,
        "succession_curve": curve,
        "py_transfer_efficacy": py_eff,
        "md_transfer_efficacy": md_eff,
        "n_pass": n_pass,
        "n_total": n_total,
        "claim_level": claim,
        "elapsed_s": elapsed,
    }
    with open("succession_results_v2.json", "w") as f:
        json.dump(out, f, indent=2)
    print("  Results → succession_results_v2.json")

    if n_pass == n_total:
        print()
        print("  ╔" + "═" * 66 + "╗")
        print("  ║  SUCCESSION v2 COMPLETE" + " " * 42 + "║")
        print("  ║" + " " * 66 + "║")
        print("  ║  Domain modules persist across 3 successive backbone rounds.   ║")
        print("  ║  Both domain types confirmed simultaneously at each checkpoint  ║")
        print(f"  ║  {NUM_ROUNDS * UPDATE_STEPS} total update steps | proj_in fixed coordinate frame    ║")
        print("  ║" + " " * 66 + "║")
        print(f"  ║  Efficacy: Python {py_eff:.0f}% | Markdown {md_eff:.0f}%{' ' * (25 - len(f'{py_eff:.0f}') - len(f'{md_eff:.0f}'))}║")
        print("  ╚" + "═" * 66 + "╝")


if __name__ == "__main__":
    main()
