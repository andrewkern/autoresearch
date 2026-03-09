"""
One-time data preparation for population genetics autoresearch.
Simulates haplotype matrices using msprime and prepares evaluation.

Usage:
    python prepare.py                  # generate data (default 20K train, 1K val)
    python prepare.py --n-train 50000  # custom training set size

Data is stored in ~/.cache/autoresearch-popgen/.
"""

import os
import math
import random
import argparse
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

N_SAMPLES = 128       # number of haplotypes per simulation
N_SITES = 256         # number of segregating sites per window
TIME_BUDGET = 300     # training time budget in seconds (5 minutes)
MASK_RATIO = 0.15     # fraction of columns to mask during training/eval
EVAL_EXAMPLES = 1024  # number of validation examples (must be divisible by batch_size)

# Simulation parameters
REGION_LENGTH = 200_000  # base pairs per simulation
MU = 1.25e-8             # mutation rate per bp per generation
RECOMB_RATE = 1e-8       # recombination rate per bp per generation
NE = 10_000              # effective population size

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-popgen")
TRAIN_PATH = os.path.join(CACHE_DIR, "train.pt")
VAL_PATH = os.path.join(CACHE_DIR, "val.pt")

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_one(seed):
    """Simulate one haplotype matrix of shape (N_SAMPLES, N_SITES).
    Returns numpy int8 array or None if not enough segregating sites."""
    import msprime

    ts = msprime.sim_ancestry(
        samples=N_SAMPLES // 2,  # diploid individuals -> N_SAMPLES haplotypes
        sequence_length=REGION_LENGTH,
        recombination_rate=RECOMB_RATE,
        population_size=NE,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=seed + 1_000_000)

    if ts.num_mutations < N_SITES:
        return None

    G = ts.genotype_matrix()  # (n_variants, N_SAMPLES)

    rng = random.Random(seed)
    start = rng.randint(0, G.shape[0] - N_SITES)
    window = G[start:start + N_SITES]  # (N_SITES, N_SAMPLES)

    return window.T.astype(np.int8)  # (N_SAMPLES, N_SITES)


def generate_examples(n_examples, seed_offset=0, num_workers=8):
    """Generate n_examples haplotype matrices using multiprocessing."""
    # Generate extra seeds in case some simulations don't have enough sites
    seeds = list(range(seed_offset + 1, seed_offset + n_examples * 3 + 1))

    results = []
    with Pool(processes=num_workers) as pool:
        for result in pool.imap(simulate_one, seeds, chunksize=100):
            if result is not None:
                results.append(result)
                if len(results) >= n_examples:
                    break
                if len(results) % 1000 == 0:
                    print(f"  Generated {len(results)}/{n_examples}...")

    if len(results) < n_examples:
        raise RuntimeError(f"Only generated {len(results)}/{n_examples}. Increase REGION_LENGTH.")

    return np.stack(results[:n_examples])

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

def make_dataloader(batch_size, split):
    """
    Yields (hap_matrix, epoch) where hap_matrix is (B, N_SAMPLES, N_SITES) float on GPU.
    Infinite iterator, shuffles each epoch for training.
    """
    assert split in ["train", "val"]
    path = TRAIN_PATH if split == "train" else VAL_PATH
    data = torch.load(path, map_location="cpu")  # (N, N_SAMPLES, N_SITES) int8
    N = data.shape[0]

    epoch = 1
    while True:
        perm = torch.randperm(N) if split == "train" else torch.arange(N)
        for i in range(0, N - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            batch = data[idx].float().cuda()
            yield batch, epoch
        epoch += 1

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpa(model, batch_size, mask_ratio=MASK_RATIO):
    """
    Bits per allele (BPA): evaluate masked column prediction quality.
    Applies deterministic masking, computes binary cross-entropy in bits
    over all masked alleles on the validation set.

    Lower is better. For random binary data with p=0.5, BPA = 1.0.
    For real genetic data with LD, a good model should do well below 1.0.
    """
    val_loader = make_dataloader(batch_size, "val")
    n_steps = EVAL_EXAMPLES // batch_size

    total_nats = 0.0
    total_alleles = 0

    for step_idx in range(n_steps):
        hap_matrix, _ = next(val_loader)  # (B, N_SAMPLES, N_SITES)
        B = hap_matrix.shape[0]

        # Deterministic mask for reproducibility
        g = torch.Generator(device=hap_matrix.device)
        g.manual_seed(step_idx + 999999)
        mask = torch.rand(B, N_SITES, device=hap_matrix.device, generator=g) < mask_ratio

        # Model returns logits (B, N_SITES, N_SAMPLES)
        logits = model(hap_matrix, mask)
        targets = hap_matrix.transpose(1, 2)  # (B, N_SITES, N_SAMPLES)

        # BCE on masked positions
        all_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        mask_3d = mask.unsqueeze(-1).expand_as(all_bce)
        total_nats += (all_bce * mask_3d).sum().item()
        total_alleles += mask_3d.sum().item()

    bpa = total_nats / (math.log(2) * total_alleles)
    return bpa

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for population genetics autoresearch")
    parser.add_argument("--n-train", type=int, default=20000, help="Number of training examples")
    parser.add_argument("--n-val", type=int, default=EVAL_EXAMPLES, help="Number of validation examples")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Cache directory: {CACHE_DIR}")
    print()

    if os.path.exists(VAL_PATH):
        print(f"Validation data already exists at {VAL_PATH}")
    else:
        print(f"Generating {args.n_val} validation examples...")
        val_data = generate_examples(args.n_val, seed_offset=1_000_000, num_workers=args.workers)
        torch.save(torch.from_numpy(val_data).to(torch.int8), VAL_PATH)
        print(f"Saved to {VAL_PATH}")
    print()

    if os.path.exists(TRAIN_PATH):
        print(f"Training data already exists at {TRAIN_PATH}")
    else:
        print(f"Generating {args.n_train} training examples...")
        train_data = generate_examples(args.n_train, seed_offset=0, num_workers=args.workers)
        torch.save(torch.from_numpy(train_data).to(torch.int8), TRAIN_PATH)
        print(f"Saved to {TRAIN_PATH}")
    print()

    print("Done! Ready to train.")
