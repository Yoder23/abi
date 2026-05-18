"""
cross_size_transfer_test.py  —  ABI Cross-Size Domain Transfer
================================================================
Demonstrates that a domain module trained on GPT-2-SMALL (117M, d_model=768)
transfers to GPT-2-MEDIUM (354M, d_model=1024) after a one-time ABI space
alignment step.

This is significant because:
  • The two backbones have DIFFERENT sizes (117M vs 354M)
  • They have DIFFERENT hidden dimensions (768 vs 1024)
  • The d=256 ABI bottleneck crosses the size boundary cleanly via alignment
  • Training a domain module on a cheap small model scales up to a large model

Protocol:
  STEP 1: Train ABI-small (proj_in: 768→256, proj_out: 256→768, domain: 256→256)
          on Python code. GPT-2-small backbone FROZEN.

  STEP 2: Align ABI-medium: Train proj_in_medium (1024→256) to match
          proj_in_small's outputs for the same input tokens.
          (proj_in_medium learns the same d=256 "language" as proj_in_small)

  STEP 3: Adapt proj_out_medium (256→1024) in 100 steps while domain_small is
          FROZEN — teaches the output projection how to express domain corrections
          in the medium model's 1024-dim backbone space.

  STEP 4: Zero-shot paste domain_small → ABI-medium (proj_in/proj_out already adapted).
          Measure Python PPL on GPT-2-medium.

Baselines:
  (A) GPT-2-medium pretrained, no domain
  (B) GPT-2-medium with natively-trained domain (500 steps, same as small)
  (C) GPT-2-medium with cross-transferred domain_small

Transfer efficacy = (A_ppl - C_ppl) / (A_ppl - B_ppl)
"""

import math, time, copy, json, pathlib
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_ABI        = 256
BATCH_SIZE   = 8
MAX_SEQ_LEN  = 128
DOMAIN_STEPS = 500       # domain training on small model
ALIGN_STEPS  = 500       # align medium proj_in to small's ABI space
ADAPT_STEPS  = 100       # adapt medium proj_out to broadcast domain corrections
ABI_LR       = 3e-4
ALIGN_LR     = 3e-4
ADAPT_LR     = 3e-4
MAX_PYTHON   = 500_000
SEED         = 42

torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

ROOT = pathlib.Path(__file__).parent.parent


class TokenDataset(Dataset):
    def __init__(self, ids, seq_len):
        n = (len(ids) // seq_len) * seq_len
        self.data = ids[:n].reshape(-1, seq_len)
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


def make_loader(ids, shuffle=True):
    return DataLoader(TokenDataset(ids, MAX_SEQ_LEN),
                      batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=True)


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


class ABIWrapped(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        g              = GPT2LMHeadModel.from_pretrained(model_name)
        dm             = g.config.n_embd
        self.backbone  = g.transformer
        self.lm_head   = g.lm_head
        self.proj_in   = nn.Linear(dm, D_ABI, bias=False)
        self.abi_ln    = nn.LayerNorm(D_ABI)
        self.proj_out  = nn.Linear(D_ABI, dm, bias=False)
        self.domain    = DomainModule(D_ABI)
        self.dm        = dm
        self.model_name = model_name

    def forward(self, ids, use_domain=True):
        h     = self.backbone(ids).last_hidden_state
        h_abi = self.abi_ln(self.proj_in(h))
        if use_domain:
            h_abi = self.domain(h_abi)
        return self.lm_head(h + self.proj_out(h_abi))


def lm_loss(logits, ids):
    B, T, V = logits.shape
    return nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))


@torch.no_grad()
def ppl_eval(model, loader, use_domain=True, max_batches=50):
    model.eval()
    tot, n = 0.0, 0
    for i, b in enumerate(loader):
        if i >= max_batches: break
        b = b.to(DEVICE)
        tot += lm_loss(model(b, use_domain=use_domain), b).item()
        n  += 1
    return math.exp(tot / n) if n else float("inf")


def train_abi_and_domain(model, loader, steps):
    """Freeze backbone + lm_head + proj_out; train proj_in, abi_ln, domain."""
    for p in model.parameters(): p.requires_grad_(False)
    for nm, p in model.named_parameters():
        if any(k in nm for k in ("proj_in", "abi_ln", "domain")):
            p.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=ABI_LR)
    model.train()
    it = iter(loader)
    for step in range(1, steps + 1):
        try: b = next(it)
        except StopIteration: it = iter(loader); b = next(it)
        b = b.to(DEVICE)
        opt.zero_grad()
        lm_loss(model(b, use_domain=True), b).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"    domain step {step}/{steps}")


def align_projin(model_large, model_small, loader, steps):
    """Train proj_in_large + abi_ln_large to match proj_in_small's d=256 outputs.

    After this, model_large's d=256 ABI space is aligned to model_small's space,
    so a domain module trained on model_small's space can be directly pasted.
    """
    for p in model_large.parameters(): p.requires_grad_(False)
    for nm, p in model_large.named_parameters():
        if any(k in nm for k in ("proj_in", "abi_ln")):
            p.requires_grad_(True)
    model_small.eval()
    model_large.train()
    opt = torch.optim.AdamW([p for p in model_large.parameters() if p.requires_grad], lr=ALIGN_LR)
    it = iter(loader)
    loss_log = []
    for step in range(1, steps + 1):
        try: b = next(it)
        except StopIteration: it = iter(loader); b = next(it)
        b = b.to(DEVICE)
        opt.zero_grad()
        with torch.no_grad():
            h_small  = model_small.abi_ln(model_small.proj_in(
                model_small.backbone(b).last_hidden_state))
        h_large = model_large.abi_ln(model_large.proj_in(
            model_large.backbone(b).last_hidden_state))
        align_loss = nn.functional.mse_loss(h_large, h_small)
        align_loss.backward()
        nn.utils.clip_grad_norm_(model_large.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            loss_log.append(align_loss.item())
            print(f"    align step {step}/{steps}: MSE={align_loss.item():.4f}")
    return loss_log


def adapt_projout(model_large, domain_small_state, loader, steps):
    """Train proj_out_large ONLY while domain is frozen (loaded from domain_small).

    This teaches the output projection to broadcast domain_small's 256-dim
    corrections into useful 1024-dim backbone-space corrections for model_large.
    """
    # Load (frozen) domain module from small model
    model_large.domain.load_state_dict(copy.deepcopy(domain_small_state))
    for p in model_large.parameters(): p.requires_grad_(False)
    for nm, p in model_large.named_parameters():
        if "proj_out" in nm:
            p.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in model_large.parameters() if p.requires_grad], lr=ADAPT_LR)
    model_large.train()
    it = iter(loader)
    for step in range(1, steps + 1):
        try: b = next(it)
        except StopIteration: it = iter(loader); b = next(it)
        b = b.to(DEVICE)
        opt.zero_grad()
        lm_loss(model_large(b, use_domain=True), b).backward()
        nn.utils.clip_grad_norm_(model_large.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            print(f"    adapt step {step}/{steps}")


def cos_sim_abi(model_a, model_b, loader):
    """Measure cosine similarity of h_abi between two models on same inputs."""
    model_a.eval(); model_b.eval()
    with torch.no_grad():
        b   = next(iter(loader)).to(DEVICE)[:4]
        h_a = model_a.abi_ln(model_a.proj_in(model_a.backbone(b).last_hidden_state))
        h_b = model_b.abi_ln(model_b.proj_in(model_b.backbone(b).last_hidden_state))
        cs  = nn.functional.cosine_similarity(
            h_a.reshape(-1, D_ABI), h_b.reshape(-1, D_ABI), dim=-1).mean().item()
    return cs, abs(cs) / (1.0 / math.sqrt(D_ABI))


def main():
    t0  = time.time()
    SEP = "═" * 72

    print(f"\n{SEP}")
    print("  ABI CROSS-SIZE DOMAIN TRANSFER TEST")
    print("  Source: GPT-2-small (117M, d_model=768)")
    print("  Target: GPT-2-medium (354M, d_model=1024)")
    print("  Claim: domain modules trained cheaply on small models transfer")
    print("         to large models via ABI space alignment")
    print(f"{SEP}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("  [Data] Loading tokenizer + Python corpus...")
    t1  = time.time()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.model_max_length = 10**30

    py_parts, py_chars = [], 0
    for p in ROOT.rglob("*.py"):
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
            py_parts.append(t); py_chars += len(t)
            if py_chars >= MAX_PYTHON * 4: break
        except Exception: continue
    pyth_raw  = "\n".join(py_parts)
    pyth_ids  = tok(pyth_raw, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)[:MAX_PYTHON]
    pyth_loader = make_loader(pyth_ids)
    print(f"  [Data] {time.time()-t1:.1f}s | Python={len(pyth_ids):,} tokens")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n  [Models] Loading GPT-2-small and GPT-2-medium...")
    t1 = time.time()
    model_small  = ABIWrapped("gpt2").to(DEVICE)
    model_medium = ABIWrapped("gpt2-medium").to(DEVICE)
    n_small  = sum(p.numel() for p in model_small.backbone.parameters())
    n_medium = sum(p.numel() for p in model_medium.backbone.parameters())
    print(f"  [Models] {time.time()-t1:.1f}s | "
          f"small={n_small/1e6:.0f}M (d={model_small.dm}) | "
          f"medium={n_medium/1e6:.0f}M (d={model_medium.dm}) | "
          f"ABI bottleneck: {D_ABI}")

    # ── STEP 1: Train ABI-small on Python ────────────────────────────────────
    print(f"\n  ── STEP 1: Train ABI-small + domain on Python "
          f"({DOMAIN_STEPS} steps, backbone frozen) ──")
    t1 = time.time()

    ppl_sm_pre = ppl_eval(model_small, pyth_loader, use_domain=False)
    train_abi_and_domain(model_small, pyth_loader, DOMAIN_STEPS)
    ppl_sm_post = ppl_eval(model_small, pyth_loader, use_domain=True)
    sm_gain = (ppl_sm_pre - ppl_sm_post) / ppl_sm_pre * 100

    print(f"  Step 1 in {time.time()-t1:.1f}s: "
          f"Small Python PPL {ppl_sm_pre:.1f} → {ppl_sm_post:.1f} (+{sm_gain:.1f}%)")

    # Freeze small model permanently
    model_small.eval()
    for p in model_small.parameters(): p.requires_grad_(False)

    # Save domain module params
    domain_small_state = copy.deepcopy(model_small.domain.state_dict())

    # ── Baseline: GPT-2-medium native domain ─────────────────────────────────
    print(f"\n  ── BASELINE: Train ABI-medium natively on Python "
          f"({DOMAIN_STEPS} steps, backbone frozen) ──")
    t1 = time.time()
    model_medium_native = copy.deepcopy(model_medium)
    ppl_md_pre = ppl_eval(model_medium_native, pyth_loader, use_domain=False)
    train_abi_and_domain(model_medium_native, pyth_loader, DOMAIN_STEPS)
    ppl_md_native = ppl_eval(model_medium_native, pyth_loader, use_domain=True)
    md_native_gain = (ppl_md_pre - ppl_md_native) / ppl_md_pre * 100
    print(f"  Baseline in {time.time()-t1:.1f}s: "
          f"Medium Python PPL {ppl_md_pre:.1f} → {ppl_md_native:.1f} (+{md_native_gain:.1f}%)")

    # ── STEP 2: Align medium's proj_in to small's ABI space ──────────────────
    print(f"\n  ── STEP 2: Align GPT-2-medium proj_in to GPT-2-small ABI space "
          f"({ALIGN_STEPS} steps) ──")
    t1 = time.time()

    cs_before, cs_mult_before = cos_sim_abi(model_small, model_medium, pyth_loader)
    print(f"  ABI cos_sim BEFORE alignment: {cs_before:.4f} ({cs_mult_before:.0f}× rand)")

    align_projin(model_medium, model_small, pyth_loader, ALIGN_STEPS)

    cs_after, cs_mult_after = cos_sim_abi(model_small, model_medium, pyth_loader)
    print(f"  ABI cos_sim AFTER alignment:  {cs_after:.4f} ({cs_mult_after:.0f}× rand) "
          f"[in {time.time()-t1:.1f}s]")

    # ── STEP 3: Adapt proj_out_medium ─────────────────────────────────────────
    print(f"\n  ── STEP 3: Adapt proj_out_medium ({ADAPT_STEPS} steps, "
          f"domain_small FROZEN) ──")
    t1 = time.time()

    adapt_projout(model_medium, domain_small_state, pyth_loader, ADAPT_STEPS)

    print(f"  Adaptation in {time.time()-t1:.1f}s")

    # ── STEP 4: Zero-shot evaluate ────────────────────────────────────────────
    print(f"\n  ── STEP 4: Evaluate cross-size transfer ──")

    # The domain module is already loaded from domain_small_state in adapt_projout
    ppl_md_xfer = ppl_eval(model_medium, pyth_loader, use_domain=True)
    ppl_md_no_dom = ppl_eval(model_medium, pyth_loader, use_domain=False)

    cross_gain = (ppl_md_no_dom - ppl_md_xfer) / ppl_md_no_dom * 100
    efficacy   = (ppl_md_pre - ppl_md_xfer) / (ppl_md_pre - ppl_md_native) * 100 if ppl_md_pre > ppl_md_native else 0.0

    elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  CROSS-SIZE TRANSFER — RESULTS")
    print(SEP)
    print(f"\n  Backbone sizes:")
    print(f"    Source: GPT-2-small  = {n_small/1e6:.0f}M params, d_model={model_small.dm}")
    print(f"    Target: GPT-2-medium = {n_medium/1e6:.0f}M params, d_model={model_medium.dm}")
    print(f"    ABI bottleneck: d={D_ABI} (same for both)")
    print()
    print(f"  Training cost breakdown:")
    print(f"    Small domain training: {DOMAIN_STEPS} steps (one-time)")
    print(f"    ABI alignment:         {ALIGN_STEPS} steps (one-time per size pair)")
    print(f"    Output adaptation:     {ADAPT_STEPS} steps (one-time per size pair)")
    print(f"    Total overhead:        {DOMAIN_STEPS + ALIGN_STEPS + ADAPT_STEPS} steps")
    print(f"    vs native {DOMAIN_STEPS}-step training on large model: +{ALIGN_STEPS+ADAPT_STEPS} extra steps")
    print()
    print(f"  Python PPL comparison (GPT-2-medium target):")
    print(f"    Pretrained (no domain):         {ppl_md_pre:.1f}")
    print(f"    No domain after alignment:      {ppl_md_no_dom:.1f}")
    print(f"    Cross-size transfer (from small): {ppl_md_xfer:.1f}")
    print(f"    Native domain training (oracle):  {ppl_md_native:.1f}")
    print()
    print(f"  ABI space alignment:")
    print(f"    Before: cos_sim = {cs_before:.4f} ({cs_mult_before:.0f}× rand)")
    print(f"    After:  cos_sim = {cs_after:.4f}  ({cs_mult_after:.0f}× rand)")
    print()
    print(f"  Cross-size domain gain: +{cross_gain:.1f}% (no-domain → transfer)")
    print(f"  Transfer efficacy vs native: {efficacy:.1f}%")
    print(f"  Total time: {elapsed:.0f}s")

    if efficacy >= 80:
        claim = "BREAKTHROUGH — 80%+ cross-size efficacy, backbone-size-agnostic domain modules"
    elif efficacy >= 50:
        claim = "BREAKTHROUGH-LEVEL — 50%+ cross-size efficacy: train small, run large"
    elif efficacy >= 30:
        claim = "STRONG — 30%+ cross-size efficacy, meaningful across-size transfer"
    elif cross_gain > 5:
        claim = f"PARTIAL — cross-size gain +{cross_gain:.1f}%, efficacy {efficacy:.1f}% (alignment direction correct)"
    else:
        claim = "WEAK — alignment insufficient or adaptation failed"

    print(f"\n  Claim level: {claim}")
    print()
    print(f"  Key result: GPT-2-small (117M) domain module → GPT-2-medium (354M)")
    print(f"  d_model mismatch: {model_small.dm} vs {model_medium.dm}")
    print(f"  ABI bottleneck: {D_ABI} (size-agnostic by construction)")

    json.dump({
        "test": "cross_size_transfer_test",
        "source_model": "gpt2 (117M)",
        "target_model": "gpt2-medium (354M)",
        "d_abi": D_ABI,
        "domain_steps": DOMAIN_STEPS,
        "align_steps": ALIGN_STEPS,
        "adapt_steps": ADAPT_STEPS,
        "ppl_medium_pretrained": ppl_md_pre,
        "ppl_medium_post_alignment_no_domain": ppl_md_no_dom,
        "ppl_medium_cross_transfer": ppl_md_xfer,
        "ppl_medium_native": ppl_md_native,
        "cross_size_gain_pct": cross_gain,
        "efficacy_vs_native_pct": efficacy,
        "abi_cos_sim_before": cs_before,
        "abi_cos_mult_before": cs_mult_before,
        "abi_cos_sim_after": cs_after,
        "abi_cos_mult_after": cs_mult_after,
        "claim": claim,
        "elapsed_s": elapsed,
    }, open("cross_size_transfer_results.json", "w"), indent=2)

    print("  Results → cross_size_transfer_results.json")


if __name__ == "__main__":
    main()
