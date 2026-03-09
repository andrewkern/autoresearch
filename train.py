"""
Population genetics autoresearch training script. Single-GPU, single-file.
Masked column prediction on simulated haplotype matrices.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import N_SAMPLES, N_SITES, TIME_BUDGET, MASK_RATIO, make_dataloader, evaluate_bpa

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    n_samples: int = 128     # haplotypes per simulation
    n_sites: int = 256       # segregating sites per window
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    d_ff_mult: int = 4


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_head
        assert config.d_model % config.n_head == 0
        self.c_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.c_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.c_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x):
        B, L, D = x.size()
        q = self.c_q(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        y = F.scaled_dot_product_attention(q, k, v)  # bidirectional (no causal mask)
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_ff = config.d_model * config.d_ff_mult
        self.c_fc = nn.Linear(config.d_model, d_ff, bias=False)
        self.c_proj = nn.Linear(d_ff, config.d_model, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = SelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x


class MaskedColumnModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Column embedding: binary vector + frequency -> d_model
        self.col_embed = nn.Linear(config.n_samples, config.d_model, bias=True)
        # Learned mask token
        self.mask_token = nn.Parameter(torch.randn(config.d_model))
        # Learned positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(config.n_sites, config.d_model) * 0.02)
        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # Output head: predict allele values for each sample
        self.output_head = nn.Linear(config.d_model, config.n_samples, bias=True)

    @torch.no_grad()
    def init_weights(self):
        d = self.config.d_model
        s = 3**0.5 * d**-0.5
        # Column embedding
        nn.init.uniform_(self.col_embed.weight, -s, s)
        # Mask token and positional embeddings
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        # Output head
        nn.init.normal_(self.output_head.weight, std=0.001)
        # Transformer blocks
        for block in self.blocks:
            nn.init.uniform_(block.attn.c_q.weight, -s, s)
            nn.init.uniform_(block.attn.c_k.weight, -s, s)
            nn.init.uniform_(block.attn.c_v.weight, -s, s)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            nn.init.zeros_(block.mlp.c_proj.weight)

    def estimate_flops(self):
        """Estimated FLOPs per example (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        nparams -= self.pos_embed.numel() + self.mask_token.numel()
        L = self.config.n_sites
        h = self.config.n_head
        d = self.config.d_model // self.config.n_head
        attn_flops = self.config.n_layer * 12 * h * d * L
        return 6 * nparams + attn_flops

    def setup_optimizer(self, embedding_lr=0.02, output_lr=0.004, matrix_lr=0.04,
                        scalar_lr=0.5, weight_decay=0.0, adam_betas=(0.8, 0.95)):
        d = self.config.d_model
        dmodel_lr_scale = (d / 512) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({d}/512) = {dmodel_lr_scale:.6f}")

        matrix_params = list(self.blocks.parameters())
        embedding_params = [self.col_embed.weight]
        output_params = [self.output_head.weight]
        small_params = [self.mask_token, self.pos_embed,
                        self.col_embed.bias, self.output_head.bias]

        param_groups = [
            dict(kind='adamw', params=output_params, lr=output_lr * dmodel_lr_scale,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=small_params, lr=embedding_lr * dmodel_lr_scale,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, hap_matrix, mask):
        """
        hap_matrix: (B, N_SAMPLES, N_SITES) float tensor of 0s and 1s
        mask: (B, N_SITES) bool tensor, True = masked

        Returns logits: (B, N_SITES, N_SAMPLES)
        """
        B, S, L = hap_matrix.shape

        # Embed columns directly from allele values
        cols = hap_matrix.transpose(1, 2)  # (B, L, S)
        x = self.col_embed(cols)  # (B, L, d_model)

        # Replace masked positions with mask token
        mask_emb = self.mask_token.view(1, 1, -1).expand(B, L, -1)
        x = torch.where(mask.unsqueeze(-1), mask_emb, x)

        # No positional embeddings — columns are permutation-equivariant

        # Transformer
        x = norm(x)
        for block in self.blocks:
            x = block(x)
        x = norm(x)

        # Predict allele values
        logits = self.output_head(x)  # (B, L, S)
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization (fp32 for V100 compatibility)
    X = g.float()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
D_MODEL = 512
N_HEADS = 16
D_FF_MULT = 3

# Optimization
TOTAL_BATCH_SIZE = 256   # examples per optimizer step
EMBEDDING_LR = 0.01     # learning rate for column embeddings (Adam)
OUTPUT_LR = 0.004        # learning rate for output head (Adam)
MATRIX_LR = 0.04         # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5          # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2       # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.9)
WARMUP_RATIO = 0.0       # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5     # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0      # final LR as fraction of initial

# Model size
DEPTH = 8                # number of transformer layers
DEVICE_BATCH_SIZE = 64   # per-device batch size (reduce if OOM)

# ---------------------------------------------------------------------------
# Setup: model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
V100_FP32_PEAK_FLOPS = 15.7e12

config = ModelConfig(
    n_samples=N_SAMPLES, n_sites=N_SITES,
    n_layer=DEPTH, n_head=N_HEADS, d_model=D_MODEL, d_ff_mult=D_FF_MULT,
)
print(f"Model config: {asdict(config)}")

model = MaskedColumnModel(config).to(device)
model.init_weights()

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")
num_flops_per_example = model.estimate_flops()
print(f"Estimated FLOPs per example: {num_flops_per_example:e}")

assert TOTAL_BATCH_SIZE % DEVICE_BATCH_SIZE == 0
grad_accum_steps = TOTAL_BATCH_SIZE // DEVICE_BATCH_SIZE

optimizer = model.setup_optimizer(
    embedding_lr=EMBEDDING_LR, output_lr=OUTPUT_LR,
    scalar_lr=SCALAR_LR, adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR, weight_decay=WEIGHT_DECAY,
)

model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(DEVICE_BATCH_SIZE, "train")
hap_matrix, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        # Create random mask for this micro-batch
        B = hap_matrix.shape[0]
        TRAIN_MASK_RATIO = 0.50
        mask = torch.rand(B, N_SITES, device=device) < TRAIN_MASK_RATIO

        # Forward pass (no autocast — fp32 for V100 stability)
        logits = model(hap_matrix, mask)
        targets = hap_matrix.transpose(1, 2)  # (B, N_SITES, N_SAMPLES)
        all_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        mask_3d = mask.unsqueeze(-1).expand_as(all_loss)
        loss = (all_loss * mask_3d).sum() / mask_3d.sum().clamp_min(1)

        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        hap_matrix, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding
    if train_loss_f > 10 or train_loss_f != train_loss_f:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    examples_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_example * TOTAL_BATCH_SIZE / dt / V100_FP32_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | ex/sec: {examples_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_examples = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
val_bpa = evaluate_bpa(model, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
steady_state_mfu = 100 * num_flops_per_example * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / V100_FP32_PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpa:          {val_bpa:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_examples:   {total_examples}")
print(f"num_steps:        {step}")
print(f"num_params:       {num_params:,}")
print(f"depth:            {DEPTH}")
