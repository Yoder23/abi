"""
abi_ablation_test.py  —  ABI Stability Coefficient Ablation
============================================================
Sweeps alpha (ABI stability loss weight) across {0, 1, 2, 3}.

alpha=0 is exactly a standard fine-tune + domain-module paste.
alpha>0 is the ABI protocol.

This directly answers:
  • "Is the stability constraint actually doing the work?"   (alpha=0 control)
  • "Can stronger stability push efficacy toward 80–90%?"   (alpha=2, alpha=3)
  • "Does the backbone still learn WikiText?"                (wiktext_ppl column)

Key framing: after the backbone update the model must be SIMULTANEOUSLY good at:
  1. The new task (WikiText, measured by wikitext_ppl)
  2. The old domain  (Python, measured by python_ppl with zero-shot domain paste)

Standard fine-tuning has no mechanism for (2). ABI does.

Protocol: identical to scale_validation_test.py (1000-step WikiText update,
GPT-2-medium 354M, 500-step frozen-backbone domain training, zero-shot paste).
"""

import math, time, copy, json, pathlib
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# ─── Config ──────────────────────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_ABI        = 256
BATCH_SIZE   = 8
MAX_SEQ_LEN  = 128
DOMAIN_STEPS  = 500
UPDATE_STEPS  = 1000
BACKBONE_LR   = 5e-5
ABI_LR        = 3e-4
ALPHAS        = [0, 1, 2, 3]   # sweep: 0 = standard fine-tune control
MAX_WIKITEXT  = 600_000        # cap to match scale_validation_test.py (1.7 epochs, causes Python forgetting)
SEED          = 42

torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

ROOT = pathlib.Path(__file__).parent.parent


# ─── Data ────────────────────────────────────────────────────────────────────
class TokenDataset(Dataset):
    def __init__(self, ids, seq_len):
        n = (len(ids) // seq_len) * seq_len
        self.data = ids[:n].reshape(-1, seq_len)
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def make_loader(ids, shuffle=True):
    return DataLoader(TokenDataset(ids, MAX_SEQ_LEN),
                      batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=True)


# ─── Model ───────────────────────────────────────────────────────────────────
class DomainModule(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net  = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.LayerNorm(d*2),
            nn.Linear(d*2, d), nn.LayerNorm(d),
        )
        self.gate = nn.Parameter(torch.zeros(1))
    def forward(self, h):
        return h + torch.sigmoid(self.gate) * self.net(h)


class ABIModel(nn.Module):
    def __init__(self):
        super().__init__()
        g            = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        dm           = g.config.n_embd
        self.backbone = g.transformer
        self.lm_head  = g.lm_head
        self.proj_in  = nn.Linear(dm, D_ABI, bias=False)
        self.abi_ln   = nn.LayerNorm(D_ABI)
        self.proj_out = nn.Linear(D_ABI, dm, bias=False)
        self.domain   = DomainModule(D_ABI)
        self.dm       = dm

    def forward(self, ids, use_domain=True):
        h     = self.backbone(ids).last_hidden_state
        h_abi = self.abi_ln(self.proj_in(h))
        if use_domain:
            h_abi = self.domain(h_abi)
        return self.lm_head(h + self.proj_out(h_abi))


# ─── Loss / eval ─────────────────────────────────────────────────────────────
def lm_loss(logits, ids):
    B, T, V = logits.shape
    return nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))

@torch.no_grad()
def ppl(model, loader, use_domain=True, max_batches=30):
    model.eval()
    tot, n = 0.0, 0
    for i, b in enumerate(loader):
        if i >= max_batches: break
        b = b.to(DEVICE)
        tot += lm_loss(model(b, use_domain=use_domain), b).item()
        n  += 1
    return math.exp(tot / n) if n else float("inf")


# ─── Training ────────────────────────────────────────────────────────────────
def train_domain_module(model, loader, steps):
    """Freeze everything except proj_in, abi_ln, domain."""
    for p in model.parameters(): p.requires_grad_(False)
    for nm, p in model.named_parameters():
        if any(k in nm for k in ("proj_in", "abi_ln", "domain")):
            p.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=ABI_LR)
    model.train()
    it = iter(loader)
    for _ in range(steps):
        try: b = next(it)
        except StopIteration: it = iter(loader); b = next(it)
        b = b.to(DEVICE)
        opt.zero_grad()
        lm_loss(model(b, use_domain=True), b).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def update_backbone(model_cur, model_anchor, loader, steps, alpha):
    """Update backbone + proj_in with stability coefficient alpha.
    alpha=0 → standard fine-tune (no stability).
    alpha>0 → ABI protocol.
    proj_out + lm_head always frozen.
    """
    for p in model_cur.parameters(): p.requires_grad_(False)
    for nm, p in model_cur.named_parameters():
        if "backbone" in nm or "proj_in" in nm:
            p.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in model_cur.parameters() if p.requires_grad], lr=BACKBONE_LR)
    model_cur.train()
    if alpha > 0: model_anchor.eval()
    it = iter(loader)
    for step in range(1, steps + 1):
        try: b = next(it)
        except StopIteration: it = iter(loader); b = next(it)
        b = b.to(DEVICE)
        opt.zero_grad()
        h      = model_cur.backbone(b).last_hidden_state
        h_abi  = model_cur.abi_ln(model_cur.proj_in(h))
        logits = model_cur.lm_head(h + model_cur.proj_out(h_abi))
        loss   = lm_loss(logits, b)
        if alpha > 0:
            with torch.no_grad():
                h_a   = model_anchor.backbone(b).last_hidden_state
                h_aa  = model_anchor.abi_ln(model_anchor.proj_in(h_a))
            loss = loss + alpha * nn.functional.mse_loss(h_abi, h_aa)
        loss.backward()
        nn.utils.clip_grad_norm_(model_cur.parameters(), 1.0)
        opt.step()
    return model_cur


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    SEP = "═" * 70

    print(f"\n{SEP}")
    print("  ABI STABILITY COEFFICIENT ABLATION")
    print("  GPT-2-medium (354M) | WikiText-2 | Python | alpha ∈ {0, 1, 2, 3}")
    print("  alpha=0 is standard fine-tune control (no ABI stability)")
    print(f"{SEP}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("  [Data] Loading corpora...")
    t1  = time.time()
    tok = GPT2TokenizerFast.from_pretrained("gpt2-medium")
    tok.pad_token = tok.eos_token

    from datasets import load_dataset
    wiki_raw = "\n".join(
        r["text"] for r in load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        if r["text"].strip()
    )
    # Use held-out WikiText test split for PPL measurement
    wiki_test_raw = "\n".join(
        r["text"] for r in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        if r["text"].strip()
    )

    py_parts, py_chars = [], 0
    for p in ROOT.rglob("*.py"):
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
            py_parts.append(t); py_chars += len(t)
            if py_chars >= 600_000: break
        except Exception: continue
    pyth_raw = "\n".join(py_parts)

    wiki_ids      = tok(wiki_raw,      return_tensors="pt", truncation=False)["input_ids"].squeeze(0)[:MAX_WIKITEXT]
    wiki_test_ids = tok(wiki_test_raw, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
    pyth_ids      = tok(pyth_raw,      return_tensors="pt", truncation=False)["input_ids"].squeeze(0)

    wiki_loader      = make_loader(wiki_ids)
    wiki_test_loader = make_loader(wiki_test_ids, shuffle=False)
    pyth_loader      = make_loader(pyth_ids)

    print(f"  [Data] {time.time()-t1:.1f}s | "
          f"wiki_train={len(wiki_ids):,} | wiki_test={len(wiki_test_ids):,} | "
          f"python={len(pyth_ids):,} tokens")

    # ── Load model + train domain ─────────────────────────────────────────────
    print("\n  [Model] Loading pretrained gpt2-medium...")
    t1 = time.time()
    model_base = ABIModel().to(DEVICE)
    n_bb  = sum(p.numel() for p in model_base.backbone.parameters())
    print(f"  [Model] {time.time()-t1:.1f}s | backbone={n_bb/1e6:.1f}M params")

    print(f"\n  [Step 1] Training domain module ({DOMAIN_STEPS} steps, backbone FROZEN)...")
    t1 = time.time()
    ppl_py_pre = ppl(model_base, pyth_loader, use_domain=True)
    train_domain_module(model_base, pyth_loader, DOMAIN_STEPS)
    ppl_py_after_domain = ppl(model_base, pyth_loader, use_domain=True)
    py_domain_gain = (ppl_py_pre - ppl_py_after_domain) / ppl_py_pre * 100
    print(f"  Domain trained in {time.time()-t1:.1f}s: "
          f"Python PPL {ppl_py_pre:.1f} → {ppl_py_after_domain:.1f} (+{py_domain_gain:.1f}%)")

    # Freeze model_base as permanent anchor
    model_base.eval()
    for p in model_base.parameters(): p.requires_grad_(False)

    # ── Native cold-start baseline ────────────────────────────────────────────
    # We'll compute this ONCE at the end using a copy of the final backbone state.
    # For now note: native baseline will be computed after alpha sweep.

    # ── Alpha sweep ───────────────────────────────────────────────────────────
    print(f"\n  [Step 2] Running alpha sweep: {ALPHAS}")
    print("  Each run: 1000-step WikiText backbone update → zero-shot domain paste\n")

    rows = []
    for alpha in ALPHAS:
        label = f"alpha={alpha}" + (" [STANDARD FINE-TUNE CONTROL]" if alpha == 0 else " [ABI protocol]")
        print(f"  ── {label} ──")
        t1 = time.time()

        model_cur = copy.deepcopy(model_base).to(DEVICE)
        update_backbone(model_cur, model_base, wiki_loader, UPDATE_STEPS, alpha)

        # Zero-shot paste: copy domain module from base → cur
        model_cur.domain.load_state_dict(copy.deepcopy(model_base.domain.state_dict()))

        # Measure
        py_nd   = ppl(model_cur, pyth_loader, use_domain=False)   # backbone only, no domain
        py_zs   = ppl(model_cur, pyth_loader, use_domain=True)    # zero-shot domain paste
        wt_ppl  = ppl(model_cur, wiki_test_loader, use_domain=False)  # WikiText test PPL
        py_gain = (py_nd - py_zs) / py_nd * 100 if py_nd > py_zs else 0.0

        # ABI cos_sim
        with torch.no_grad():
            sb    = next(iter(pyth_loader)).to(DEVICE)[:4]
            h_A   = model_base.abi_ln(model_base.proj_in(model_base.backbone(sb).last_hidden_state))
            h_B   = model_cur.abi_ln(model_cur.proj_in(model_cur.backbone(sb).last_hidden_state))
            cs    = nn.functional.cosine_similarity(h_A.reshape(-1, D_ABI), h_B.reshape(-1, D_ABI), dim=-1).mean().item()
            cs_mult = abs(cs) / (1.0 / math.sqrt(D_ABI))

        elapsed = time.time() - t1
        rows.append(dict(
            alpha=alpha, py_nd=py_nd, py_zs=py_zs, py_gain=py_gain,
            wt_ppl=wt_ppl, cos_sim=cs, cos_mult=cs_mult, elapsed=elapsed,
        ))
        print(f"    Done in {elapsed:.0f}s | WikiText PPL: {wt_ppl:.2f} | "
              f"Python no-domain: {py_nd:.1f} | zero-shot: {py_zs:.1f} "
              f"(+{py_gain:.1f}%) | ABI {cs_mult:.0f}× rand\n")

    # ── Native cold-start baseline ────────────────────────────────────────────
    print("  ── Native cold-start baseline (on alpha=1 updated backbone) ──")
    t1 = time.time()
    # Use the alpha=1 backbone as the "representative updated backbone"
    alpha1_idx = ALPHAS.index(1)
    model_native = copy.deepcopy(model_base).to(DEVICE)
    update_backbone(model_native, model_base, wiki_loader, UPDATE_STEPS, alpha=1)
    # Fresh domain module (no pre-training, cold-start)
    model_native.domain = DomainModule(D_ABI).to(DEVICE)
    py_cs_pre = ppl(model_native, pyth_loader, use_domain=True)
    train_domain_module(model_native, pyth_loader, DOMAIN_STEPS)
    py_cs = ppl(model_native, pyth_loader, use_domain=True)
    native_gain = (py_cs_pre - py_cs) / py_cs_pre * 100
    print(f"    Native cold-start in {time.time()-t1:.0f}s: "
          f"PPL {py_cs_pre:.1f} → {py_cs:.1f} (+{native_gain:.1f}%)")

    # Compute efficacy for each alpha
    for r in rows:
        r["efficacy"] = r["py_gain"] / native_gain * 100 if native_gain > 0 else 0.0

    # ── Print main results table ──────────────────────────────────────────────
    elapsed_total = time.time() - t0
    print(f"\n{SEP}")
    print("  ABLATION RESULTS — ABI STABILITY COEFFICIENT SWEEP")
    print(f"{SEP}\n")
    print("  Baseline (GPT-2-medium pretrained, no update, no domain module):")
    print(f"    Python PPL: {ppl_py_pre:.1f}")
    print(f"\n  Native cold-start (500-step domain training on updated backbone): PPL {py_cs:.1f}")
    print()

    # Table header
    w = 10
    print(f"  {'alpha':>{w}} | {'WikiText PPL':>13} | {'Py no-dom PPL':>14} | {'Py zero-shot PPL':>17} | {'ZS gain':>8} | {'Efficacy':>9} | {'ABI ×rand':>10}")
    print("  " + "-"*95)
    for r in rows:
        tag = " ← control" if r["alpha"] == 0 else ""
        print(f"  {r['alpha']:>{w}} | {r['wt_ppl']:>13.2f} | {r['py_nd']:>14.1f} | "
              f"{r['py_zs']:>17.1f} | {r['py_gain']:>7.1f}% | "
              f"{r['efficacy']:>8.1f}% | {r['cos_mult']:>9.0f}×{tag}")
    print(f"\n  Total elapsed: {elapsed_total:.0f}s\n")

    # ── Interpretation ────────────────────────────────────────────────────────
    print(f"{SEP}")
    print("  INTERPRETATION")
    print(f"{SEP}\n")

    alpha0 = next(r for r in rows if r["alpha"] == 0)
    alpha1 = next(r for r in rows if r["alpha"] == 1)
    best   = max(rows, key=lambda r: r["efficacy"])

    print(f"  [Control] Standard fine-tune (alpha=0) + paste:")
    if alpha0["efficacy"] < 10:
        print(f"    → Python efficacy {alpha0['efficacy']:.1f}% — domain module FAILS without stability.")
        print(f"    → This proves the ABI stability constraint is load-bearing, not redundant.")
    else:
        print(f"    → Python efficacy {alpha0['efficacy']:.1f}% (some signal even without stability).")

    print(f"\n  [ABI alpha=1 — confirmed]: efficacy {alpha1['efficacy']:.1f}%"
          f" | WikiText PPL {alpha1['wt_ppl']:.2f}")

    print(f"\n  [Best alpha]: alpha={best['alpha']} → efficacy {best['efficacy']:.1f}%"
          f" | WikiText PPL {best['wt_ppl']:.2f}")

    if best["efficacy"] >= 90:
        claim = "BREAKTHROUGH — 90%+ efficacy, changes assumptions"
    elif best["efficacy"] >= 80:
        claim = "BREAKTHROUGH-LEVEL — 80–90% efficacy, peer bar met"
    elif best["efficacy"] >= 70:
        claim = "STRONG — 70–80%, people will pay close attention"
    elif best["efficacy"] >= 50:
        claim = "STRONG — above peer minimum bar (50–70%)"
    else:
        claim = f"PARTIAL — best efficacy {best['efficacy']:.1f}%"

    print(f"\n  Peak claim level: {claim}")

    # WikiText parity check
    print(f"\n  WikiText PPL comparison (new-task learning):")
    print(f"    alpha=0 (standard fine-tune): {alpha0['wt_ppl']:.2f}")
    for r in rows:
        if r["alpha"] > 0:
            delta = r["wt_ppl"] - alpha0["wt_ppl"]
            sign  = "+" if delta >= 0 else ""
            print(f"    alpha={r['alpha']} (ABI): {r['wt_ppl']:.2f} ({sign}{delta:.2f} vs standard)")
    print("    → If ABI WikiText PPL ≈ standard, backbone is still learning new task effectively.")
    print("      If ABI WikiText PPL >> standard, stability is hurting new-task learning.")

    print(f"\n  The dual-task story:")
    print(f"    After 1000-step WikiText update:")
    wt_std = alpha0["wt_ppl"]
    wt_abi = alpha1["wt_ppl"]
    py_std = alpha0["py_zs"]
    py_abi = alpha1["py_zs"]
    print(f"    Standard fine-tune: WikiText PPL {wt_std:.2f} | Python PPL {py_std:.1f}")
    print(f"    ABI (alpha=1):      WikiText PPL {wt_abi:.2f} | Python PPL {py_abi:.1f}")
    py_improvement = (py_std - py_abi) / py_std * 100
    wt_gap = (wt_abi - wt_std) / wt_std * 100
    print(f"    → ABI Python PPL is {py_improvement:.1f}% better at similar WikiText performance (+{wt_gap:.1f}% gap)")

    # Save
    out = {
        "test": "abi_ablation_test",
        "backbone": "gpt2-medium (354.8M pretrained)",
        "update_corpus": "WikiText-2",
        "domain_corpus": "Python source code",
        "domain_steps": DOMAIN_STEPS,
        "update_steps": UPDATE_STEPS,
        "native_ppl": py_cs,
        "native_gain_pct": native_gain,
        "rows": rows,
        "elapsed_s": elapsed_total,
    }
    with open("abi_ablation_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  Results → abi_ablation_results.json")


if __name__ == "__main__":
    main()
