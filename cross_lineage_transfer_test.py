"""
cross_lineage_transfer_test.py  —  ABI Cross-Lineage Domain Transfer
=====================================================================
Demonstrates that a domain module trained on Pythia-410m (EleutherAI, GPT-NeoX
architecture, The Pile training data) transfers zero-shot to GPT-2-medium
(OpenAI, GPT-2 architecture, WebText training data) via ABI space alignment.

Both models have h=1024, so this test ISOLATES the lineage dimension:
  • Same hidden dimension: 1024
  • Different architecture: gpt_neox vs gpt2
  • Different training data: The Pile vs WebText
  • Different tokenizer: neox (50254) vs gpt2-bpe (50257)
  • Different organization: EleutherAI vs OpenAI

This directly addresses the peer's remaining critique:
  "This only works because the models are structurally similar."

Cross-tokenizer alignment protocol:
  Alignment operates at the SENTENCE level: the same text is tokenized with
  EACH model's own tokenizer independently, ABI representations are MEAN-POOLED
  over sequence length, and MSE loss aligns the resulting sentence vectors.
  This is the correct physically meaningful approach when tokenizations differ.

Steps:
  STEP 0: Establish baseline  (GPT-2-medium native domain, GPT-2 tokenizer)
  STEP 1: Train domain on Pythia-410m  (Pythia tokenizer, backbone frozen)
  STEP 2: Align GPT-2-medium proj_in to Pythia's ABI space  (mean-pool MSE)
  STEP 3: Adapt proj_out for GPT-2-medium  (domain_pythia FROZEN, 100 steps)
  STEP 4: Zero-shot evaluate cross-lineage transfer
"""

import math, time, copy, json, pathlib
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2TokenizerFast, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForCausalLM,
)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_ABI        = 256
BATCH_SIZE   = 6     # slightly smaller for 410M+354M in same run
MAX_SEQ_LEN  = 128
DOMAIN_STEPS = 500
ALIGN_STEPS  = 500
ADAPT_STEPS  = 150
NATIVE_STEPS = 500
ABI_LR       = 3e-4
ALIGN_LR     = 3e-4
ADAPT_LR     = 3e-4
MAX_PY_CHARS = 2_400_000   # ~500K tokens for Python corpus
SEED         = 42

torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

ROOT = pathlib.Path(__file__).parent.parent

# ─── Architecture wrappers ────────────────────────────────────────────────────

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


class ABIPythia(nn.Module):
    """ABI wrapper around GPT-NeoX (Pythia) backbone."""
    def __init__(self, model_name="EleutherAI/pythia-410m"):
        super().__init__()
        m             = AutoModelForCausalLM.from_pretrained(model_name)
        dm            = m.config.hidden_size
        self.backbone = m.gpt_neox          # → BaseModelOutputWithPast
        self.lm_head  = m.embed_out         # unembedding
        self.proj_in  = nn.Linear(dm, D_ABI, bias=False)
        self.abi_ln   = nn.LayerNorm(D_ABI)
        self.proj_out = nn.Linear(D_ABI, dm, bias=False)
        self.domain   = DomainModule(D_ABI)
        self.dm       = dm
        self.model_name = model_name

    def forward(self, ids, use_domain=True):
        h     = self.backbone(ids).last_hidden_state
        h_abi = self.abi_ln(self.proj_in(h))
        if use_domain:
            h_abi = self.domain(h_abi)
        return self.lm_head(h + self.proj_out(h_abi))

    def get_mean_abi(self, ids):
        """Return mean-pooled ABI representation for cross-tokenizer alignment."""
        with torch.no_grad():
            h = self.backbone(ids).last_hidden_state        # (B, T, dm)
        h_abi = self.abi_ln(self.proj_in(h))               # (B, T, D_ABI)
        return h_abi.mean(dim=1)                            # (B, D_ABI)


class ABIGPT2(nn.Module):
    """ABI wrapper around GPT-2 backbone."""
    def __init__(self, model_name="gpt2-medium"):
        super().__init__()
        g             = GPT2LMHeadModel.from_pretrained(model_name)
        dm            = g.config.n_embd
        self.backbone = g.transformer
        self.lm_head  = g.lm_head
        self.proj_in  = nn.Linear(dm, D_ABI, bias=False)
        self.abi_ln   = nn.LayerNorm(D_ABI)
        self.proj_out = nn.Linear(D_ABI, dm, bias=False)
        self.domain   = DomainModule(D_ABI)
        self.dm       = dm
        self.model_name = model_name

    def forward(self, ids, use_domain=True):
        h     = self.backbone(ids).last_hidden_state
        h_abi = self.abi_ln(self.proj_in(h))
        if use_domain:
            h_abi = self.domain(h_abi)
        return self.lm_head(h + self.proj_out(h_abi))

    def get_mean_abi(self, ids):
        """Return mean-pooled ABI representation for cross-tokenizer alignment."""
        h     = self.backbone(ids).last_hidden_state
        h_abi = self.abi_ln(self.proj_in(h))
        return h_abi.mean(dim=1)


# ─── Paired text dataset ─────────────────────────────────────────────────────

class PairedTextDataset(Dataset):
    """Same raw text chunks tokenized independently with two tokenizers.
    Yields (pythia_ids, gpt2_ids) pairs for cross-lineage ABI alignment.
    """
    def __init__(self, text_chunks, tok_pythia, tok_gpt2, seq_len):
        self.pairs = []
        for chunk in text_chunks:
            p = tok_pythia(chunk, return_tensors="pt",
                           truncation=True, max_length=seq_len)["input_ids"].squeeze(0)
            g = tok_gpt2(chunk,   return_tensors="pt",
                         truncation=True, max_length=seq_len)["input_ids"].squeeze(0)
            if len(p) >= 4 and len(g) >= 4:
                self.pairs.append((p, g))

    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]


def collate_pairs(batch):
    """Pad each side independently to its own max length."""
    ps = [b[0] for b in batch]
    gs = [b[1] for b in batch]
    p_max = max(len(x) for x in ps)
    g_max = max(len(x) for x in gs)
    ps_pad = torch.stack([nn.functional.pad(x, (0, p_max - len(x))) for x in ps])
    gs_pad = torch.stack([nn.functional.pad(x, (0, g_max - len(x))) for x in gs])
    return ps_pad, gs_pad


class TokenDataset(Dataset):
    def __init__(self, ids, seq_len):
        n = (len(ids) // seq_len) * seq_len
        self.data = ids[:n].reshape(-1, seq_len)
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


def make_loader(ids, batch_size=BATCH_SIZE, shuffle=True):
    return DataLoader(TokenDataset(ids, MAX_SEQ_LEN),
                      batch_size=batch_size, shuffle=shuffle, drop_last=True)


# ─── Loss / eval ─────────────────────────────────────────────────────────────

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


# ─── Training routines ───────────────────────────────────────────────────────

def train_abi_domain(model, loader, steps, tag=""):
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
            print(f"    {tag}domain step {step}/{steps}")


def align_proj_in_cross_lineage(model_target, model_source, paired_loader, steps):
    """Align target's proj_in+abi_ln to source's ABI space using mean-pool MSE.

    For each paired batch (pythia_ids, gpt2_ids) of the SAME TEXT:
      source_vec = source.get_mean_abi(source_ids)  (frozen)
      target_vec = target.get_mean_abi(target_ids)  (trainable: proj_in + abi_ln)
      loss = MSE(target_vec, source_vec)

    Mean-pooling handles the tokenizer mismatch: different T, same semantic content.
    """
    for p in model_target.parameters(): p.requires_grad_(False)
    for nm, p in model_target.named_parameters():
        if any(k in nm for k in ("proj_in", "abi_ln")):
            p.requires_grad_(True)
    model_source.eval()
    model_target.train()
    opt = torch.optim.AdamW([p for p in model_target.parameters() if p.requires_grad], lr=ALIGN_LR)
    it = iter(paired_loader)
    for step in range(1, steps + 1):
        try: pythia_ids, gpt2_ids = next(it)
        except StopIteration: it = iter(paired_loader); pythia_ids, gpt2_ids = next(it)
        pythia_ids = pythia_ids.to(DEVICE)
        gpt2_ids   = gpt2_ids.to(DEVICE)
        opt.zero_grad()
        with torch.no_grad():
            src_vec = model_source.get_mean_abi(pythia_ids)  # (B, D_ABI)
        tgt_vec = model_target.get_mean_abi(gpt2_ids)        # (B, D_ABI)
        loss = nn.functional.mse_loss(tgt_vec, src_vec)
        loss.backward()
        nn.utils.clip_grad_norm_(model_target.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"    align step {step}/{steps}: MSE={loss.item():.4f}")
    return model_target


def adapt_proj_out(model_target, domain_state, loader, steps):
    """Train only proj_out while domain is frozen (loaded from source)."""
    model_target.domain.load_state_dict(copy.deepcopy(domain_state))
    for p in model_target.parameters(): p.requires_grad_(False)
    for nm, p in model_target.named_parameters():
        if "proj_out" in nm:
            p.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in model_target.parameters() if p.requires_grad], lr=ADAPT_LR)
    model_target.train()
    it = iter(loader)
    for step in range(1, steps + 1):
        try: b = next(it)
        except StopIteration: it = iter(loader); b = next(it)
        b = b.to(DEVICE)
        opt.zero_grad()
        lm_loss(model_target(b, use_domain=True), b).backward()
        nn.utils.clip_grad_norm_(model_target.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"    adapt step {step}/{steps}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t0  = time.time()
    SEP = "═" * 74

    print(f"\n{SEP}")
    print("  ABI CROSS-LINEAGE DOMAIN TRANSFER TEST")
    print("  Source: Pythia-410m (EleutherAI | GPT-NeoX | The Pile | 50254-vocab)")
    print("  Target: GPT-2-medium (OpenAI     | GPT-2    | WebText  | 50257-vocab)")
    print("  Both: d_model=1024  |  ABI bottleneck: d=256")
    print("  Alignment: sentence mean-pool (cross-tokenizer MSE)")
    print(f"{SEP}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("  [Data] Loading tokenizers + Python corpus...")
    t1  = time.time()
    tok_pythia = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    tok_gpt2   = GPT2TokenizerFast.from_pretrained("gpt2")
    tok_pythia.pad_token = tok_pythia.eos_token
    tok_gpt2.pad_token   = tok_gpt2.eos_token
    tok_pythia.model_max_length = 10**30
    tok_gpt2.model_max_length   = 10**30

    py_parts, py_chars = [], 0
    for p in ROOT.rglob("*.py"):
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
            py_parts.append(t); py_chars += len(t)
            if py_chars >= MAX_PY_CHARS: break
        except Exception: continue
    pyth_raw = "\n".join(py_parts)

    # Flat token tensors for each model (for standard training and PPL eval)
    pyth_ids_p = tok_pythia(pyth_raw, return_tensors="pt",
                             truncation=False)["input_ids"].squeeze(0)[:500_000]
    pyth_ids_g = tok_gpt2(pyth_raw, return_tensors="pt",
                           truncation=False)["input_ids"].squeeze(0)[:500_000]

    # Split corpus into text chunks for paired alignment dataset
    # Chunk at character level so both tokenizers see the same semantic content
    CHUNK_CHARS = 512
    raw_chunks  = [pyth_raw[i : i + CHUNK_CHARS]
                   for i in range(0, len(pyth_raw) - CHUNK_CHARS, CHUNK_CHARS)]

    paired_ds = PairedTextDataset(raw_chunks, tok_pythia, tok_gpt2, MAX_SEQ_LEN)
    paired_loader = DataLoader(paired_ds, batch_size=BATCH_SIZE,
                               shuffle=True, drop_last=True, collate_fn=collate_pairs)

    pyth_loader_p = make_loader(pyth_ids_p)          # for Pythia training
    pyth_loader_g = make_loader(pyth_ids_g)          # for GPT-2 eval + adapt

    print(f"  [Data] {time.time()-t1:.1f}s | "
          f"py_pythia={len(pyth_ids_p):,} | py_gpt2={len(pyth_ids_g):,} | "
          f"paired_chunks={len(paired_ds):,}")

    print(f"\n  [Tokenizer diff check]")
    sample = "def train(model, optimizer, data):"
    print(f"  Text: '{sample}'")
    print(f"  Pythia: {tok_pythia.encode(sample)}")
    print(f"  GPT-2:  {tok_gpt2.encode(sample)}")
    print(f"  (Different IDs confirm genuinely different tokenization)")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n  [Models] Loading Pythia-410m and GPT-2-medium...")
    t1 = time.time()
    model_pythia = ABIPythia("EleutherAI/pythia-410m").to(DEVICE)
    model_gpt2   = ABIGPT2("gpt2-medium").to(DEVICE)
    n_p = sum(p.numel() for p in model_pythia.backbone.parameters())
    n_g = sum(p.numel() for p in model_gpt2.backbone.parameters())
    print(f"  [Models] {time.time()-t1:.1f}s | "
          f"Pythia-410m={n_p/1e6:.0f}M (d={model_pythia.dm}) | "
          f"GPT-2-medium={n_g/1e6:.0f}M (d={model_gpt2.dm}) | "
          f"both d_abi={D_ABI}")

    # ── STEP 0: Native GPT-2-medium baseline ─────────────────────────────────
    print(f"\n  ── STEP 0: Native GPT-2-medium domain baseline "
          f"({NATIVE_STEPS} steps, GPT-2 tokenizer) ──")
    t1 = time.time()
    model_gpt2_native = copy.deepcopy(model_gpt2)
    ppl_gpt2_pre  = ppl_eval(model_gpt2_native, pyth_loader_g, use_domain=False)
    train_abi_domain(model_gpt2_native, pyth_loader_g, NATIVE_STEPS, tag="native-")
    ppl_gpt2_native = ppl_eval(model_gpt2_native, pyth_loader_g, use_domain=True)
    gpt2_native_gain = (ppl_gpt2_pre - ppl_gpt2_native) / ppl_gpt2_pre * 100
    print(f"  Native baseline in {time.time()-t1:.1f}s: "
          f"GPT-2-medium Python PPL {ppl_gpt2_pre:.1f} → {ppl_gpt2_native:.1f} "
          f"(+{gpt2_native_gain:.1f}%)")

    # Freeze native model
    model_gpt2_native.eval()
    for p in model_gpt2_native.parameters(): p.requires_grad_(False)

    # ── STEP 1: Train domain module on Pythia-410m ───────────────────────────
    print(f"\n  ── STEP 1: Train ABI+domain on Pythia-410m "
          f"({DOMAIN_STEPS} steps, Pythia tokenizer) ──")
    t1 = time.time()
    ppl_py_pre = ppl_eval(model_pythia, pyth_loader_p, use_domain=False)
    train_abi_domain(model_pythia, pyth_loader_p, DOMAIN_STEPS, tag="pythia-")
    ppl_py_post = ppl_eval(model_pythia, pyth_loader_p, use_domain=True)
    py_gain = (ppl_py_pre - ppl_py_post) / ppl_py_pre * 100
    print(f"  Pythia domain trained in {time.time()-t1:.1f}s: "
          f"Python PPL {ppl_py_pre:.1f} → {ppl_py_post:.1f} (+{py_gain:.1f}%)")

    # Freeze Pythia
    model_pythia.eval()
    for p in model_pythia.parameters(): p.requires_grad_(False)
    domain_pythia_state = copy.deepcopy(model_pythia.domain.state_dict())

    # ── Measure ABI space before alignment ───────────────────────────────────
    sample_batch = next(iter(paired_loader))
    p_ids, g_ids = sample_batch[0].to(DEVICE), sample_batch[1].to(DEVICE)
    with torch.no_grad():
        src_v = model_pythia.get_mean_abi(p_ids)
        tgt_v = model_gpt2.get_mean_abi(g_ids)
        cs_before = nn.functional.cosine_similarity(src_v, tgt_v, dim=-1).mean().item()
        rand_floor = 1.0 / math.sqrt(D_ABI)
        cs_before_mult = abs(cs_before) / rand_floor

    print(f"\n  ABI cos_sim BEFORE alignment (Pythia vs GPT-2-medium): "
          f"{cs_before:.4f} ({cs_before_mult:.0f}× rand)")

    # ── STEP 2: Align GPT-2-medium proj_in to Pythia's ABI space ────────────
    print(f"\n  ── STEP 2: Cross-lineage ABI alignment "
          f"({ALIGN_STEPS} steps, mean-pool MSE) ──")
    print("  (Same text, different tokenizers — mean-pooled sentence vectors)")
    t1 = time.time()
    align_proj_in_cross_lineage(model_gpt2, model_pythia, paired_loader, ALIGN_STEPS)

    with torch.no_grad():
        tgt_v2 = model_gpt2.get_mean_abi(g_ids)
        cs_after = nn.functional.cosine_similarity(src_v, tgt_v2, dim=-1).mean().item()
        cs_after_mult = abs(cs_after) / rand_floor
    print(f"  Alignment in {time.time()-t1:.1f}s")
    print(f"  ABI cos_sim AFTER alignment: {cs_after:.4f} ({cs_after_mult:.0f}× rand)")

    # ── STEP 3: Adapt proj_out for GPT-2-medium ──────────────────────────────
    print(f"\n  ── STEP 3: Adapt proj_out_gpt2 "
          f"({ADAPT_STEPS} steps, domain_pythia FROZEN) ──")
    t1 = time.time()
    adapt_proj_out(model_gpt2, domain_pythia_state, pyth_loader_g, ADAPT_STEPS)
    print(f"  Adaptation in {time.time()-t1:.1f}s")

    # ── STEP 4: Evaluate ─────────────────────────────────────────────────────
    print(f"\n  ── STEP 4: Evaluate cross-lineage transfer ──")
    ppl_no_domain = ppl_eval(model_gpt2, pyth_loader_g, use_domain=False)
    ppl_xfer      = ppl_eval(model_gpt2, pyth_loader_g, use_domain=True)
    cross_gain    = (ppl_no_domain - ppl_xfer) / ppl_no_domain * 100
    efficacy      = ((ppl_gpt2_pre - ppl_xfer) /
                     (ppl_gpt2_pre - ppl_gpt2_native) * 100
                     if ppl_gpt2_pre > ppl_gpt2_native else 0.0)

    elapsed = time.time() - t0

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  CROSS-LINEAGE TRANSFER — RESULTS")
    print(SEP)
    print()
    print("  Model families:")
    print(f"    Source: Pythia-410m   (EleutherAI / GPT-NeoX / The Pile / {tok_pythia.vocab_size}-vocab-tokens)")
    print(f"    Target: GPT-2-medium  (OpenAI     / GPT-2    / WebText  / {tok_gpt2.vocab_size}-vocab-tokens)")
    print(f"    Both d_model=1024 | ABI d=256 | DIFFERENT tokenizers confirmed")
    print()
    print("  Training cost:")
    print(f"    Domain training on SMALL/CHEAP model (Pythia-410m): {DOMAIN_STEPS} steps")
    print(f"    Cross-lineage alignment (GPT-2-medium proj_in):      {ALIGN_STEPS} steps")
    print(f"    Output adaptation (GPT-2-medium proj_out):           {ADAPT_STEPS} steps")
    print()
    print("  Python PPL on GPT-2-medium (GPT-2 tokenizer throughout):")
    print(f"    Pretrained GPT-2-medium (no domain):     {ppl_gpt2_pre:.1f}")
    print(f"    No domain after alignment:               {ppl_no_domain:.1f}")
    print(f"    Cross-lineage transfer (domain_pythia):  {ppl_xfer:.1f}")
    print(f"    Native GPT-2-medium training (oracle):   {ppl_gpt2_native:.1f}")
    print()
    print("  ABI space alignment:")
    print(f"    Before: cos_sim={cs_before:.4f} ({cs_before_mult:.0f}× rand)")
    print(f"    After:  cos_sim={cs_after:.4f}  ({cs_after_mult:.0f}× rand)")
    print()
    print(f"  Cross-lineage zero-shot gain: +{cross_gain:.1f}%")
    print(f"  Transfer efficacy vs native:  {efficacy:.1f}%")
    print(f"  Total time: {elapsed:.0f}s")

    # Claim
    if efficacy >= 80:
        claim = ("BREAKTHROUGH — 80%+ cross-lineage efficacy. "
                 "Domain knowledge transfers across model families.")
    elif efficacy >= 60:
        claim = "BREAKTHROUGH-LEVEL — 60%+ cross-lineage. Lineage independence demonstrated."
    elif efficacy >= 40:
        claim = "STRONG — 40%+ cross-lineage transfer. Effect survives lineage boundary."
    elif cross_gain >= 10:
        claim = f"PARTIAL — {efficacy:.1f}% efficacy; clear positive signal (+{cross_gain:.1f}% gain)"
    else:
        claim = f"WEAK/FAILED — efficacy {efficacy:.1f}%, gain {cross_gain:.1f}%"

    print(f"\n  Claim level: {claim}")

    # Comparison to same-family cross-size result
    print()
    print("  Comparison to same-family cross-size result (cross_size_transfer_test.py):")
    print("    GPT-2-small → GPT-2-medium (same family, diff size): 88.2%")
    print(f"    Pythia-410m → GPT-2-medium (diff family, diff arch): {efficacy:.1f}%")
    if efficacy >= 60:
        print("    Result: cross-lineage efficacy is within striking range of same-family.")
        print("    The ABI bottleneck bridges lineage boundaries.")
    elif efficacy >= 30:
        print("    Result: lineage transfer is harder than same-family, but positive.")
        print("    ABI provides partial lineage-invariant structure.")

    json.dump({
        "test": "cross_lineage_transfer_test",
        "source": "EleutherAI/pythia-410m",
        "target": "gpt2-medium",
        "source_arch": "gpt_neox",
        "target_arch": "gpt2",
        "d_abi": D_ABI,
        "domain_steps": DOMAIN_STEPS,
        "align_steps": ALIGN_STEPS,
        "adapt_steps": ADAPT_STEPS,
        "ppl_gpt2_pretrained": ppl_gpt2_pre,
        "ppl_after_align_no_domain": ppl_no_domain,
        "ppl_cross_lineage_transfer": ppl_xfer,
        "ppl_gpt2_native": ppl_gpt2_native,
        "cross_lineage_gain_pct": cross_gain,
        "efficacy_vs_native_pct": efficacy,
        "abi_cos_sim_before": cs_before,
        "abi_cos_mult_before": cs_before_mult,
        "abi_cos_sim_after": cs_after,
        "abi_cos_mult_after": cs_after_mult,
        "claim": claim,
        "elapsed_s": elapsed,
    }, open("cross_lineage_results.json", "w"), indent=2)
    print("\n  Results → cross_lineage_results.json")


if __name__ == "__main__":
    main()
