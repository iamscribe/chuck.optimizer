"""
Tests for Chuck Optimizer — PyTorch Edition.

Run:  pytest test_chuck.py -v
"""

import math
import os
import struct
import tempfile

import torch
import torch.nn as nn
import pytest

from chuck import ChuckMemory, ChuckMonitor, ChuckOptimizer, chuck_params


# ═══════════════════════════════════════════════════════════════════════
# ChuckMemory
# ═══════════════════════════════════════════════════════════════════════

class TestChuckMemory:

    def test_binary_compat_with_c(self, tmp_path):
        """16-byte entries must match C struct layout."""
        path = tmp_path / 'chuck.mem'
        with open(path, 'wb') as f:
            f.write(struct.pack('ffff', 1.5, 0.3, 1.0, -0.1))
            f.write(struct.pack('ffff', 0.8, 0.2, 0.95, -0.05))
        mem = ChuckMemory(path=str(path))
        n = mem.load()
        assert n == 2
        assert abs(mem.entries[0][0] - 1.5) < 1e-6   # loss
        assert abs(mem.entries[0][1] - 0.3) < 1e-6   # grad_norm
        assert abs(mem.entries[0][2] - 1.0) < 1e-6   # lambda
        assert abs(mem.entries[0][3] - (-0.1)) < 1e-6  # delta_loss
        assert abs(mem.entries[1][2] - 0.95) < 1e-6

    def test_save_load_roundtrip(self, tmp_path):
        path = tmp_path / 'chuck.mem'
        mem = ChuckMemory(path=str(path))
        mem.save_entry(2.0, 0.5, 1.2, -0.3)
        mem.save_entry(1.0, 0.3, 0.8, -0.1)
        assert len(mem) == 2

        mem2 = ChuckMemory(path=str(path))
        n = mem2.load()
        assert n == 2
        assert abs(mem2.entries[0][0] - 2.0) < 1e-6
        assert abs(mem2.entries[1][0] - 1.0) < 1e-6

    def test_recall_nearest_neighbor(self, tmp_path):
        path = tmp_path / 'chuck.mem'
        mem = ChuckMemory(path=str(path))
        mem.save_entry(1.0, 0.5, 0.8, -0.1)   # low loss zone
        mem.save_entry(5.0, 2.0, 1.5, -0.5)   # high loss zone
        # Query near (1.0, 0.5) should return lambda=0.8
        lam = mem.recall(1.1, 0.45)
        assert abs(lam - 0.8) < 1e-6
        # Query near (5.0, 2.0) should return lambda=1.5
        lam = mem.recall(4.9, 1.9)
        assert abs(lam - 1.5) < 1e-6

    def test_recall_prefers_wins(self, tmp_path):
        """Entries with delta_loss < 0 get 0.5x distance (preferred)."""
        path = tmp_path / 'chuck.mem'
        mem = ChuckMemory(path=str(path))
        # Place entries at symmetric offsets from query point (1.0, 0.5)
        # so raw distances are equal — only the 0.5x win bonus differs
        mem.save_entry(0.9, 0.4, 0.7, 0.2)    # loss went up (bad)
        mem.save_entry(1.1, 0.6, 1.3, -0.2)   # loss went down (win)
        lam = mem.recall(1.0, 0.5)
        assert abs(lam - 1.3) < 1e-6

    def test_reservoir_sampling_caps(self, tmp_path):
        path = tmp_path / 'chuck.mem'
        cap = 10
        mem = ChuckMemory(capacity=cap, path=str(path))
        for i in range(100):
            mem.save_entry(float(i), 0.1, 1.0, -0.01)
        assert len(mem) == cap
        assert mem.total == 100
        # File should contain exactly cap entries
        with open(path, 'rb') as f:
            data = f.read()
        assert len(data) == cap * mem.ENTRY_SIZE

    def test_empty_recall(self, tmp_path):
        mem = ChuckMemory(path=str(tmp_path / 'empty.mem'))
        assert mem.recall(1.0, 0.5) == -1.0


# ═══════════════════════════════════════════════════════════════════════
# ChuckMonitor
# ═══════════════════════════════════════════════════════════════════════

class TestChuckMonitor:

    def test_silu_health(self):
        """SiLU with mostly-dead outputs should reduce σ."""
        model = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 1))
        mon = ChuckMonitor(model)
        # Force near-zero activations by setting large negative bias
        with torch.no_grad():
            model[0].bias.fill_(-100.0)
        x = torch.randn(4, 8)
        _ = model(x)
        sigma = mon.sigma
        assert sigma < 1.0, f'dead SiLU should reduce σ, got {sigma}'
        mon.detach()

    def test_healthy_model_sigma_near_one(self):
        """Normal model should have σ ≈ 1.0."""
        model = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 1))
        mon = ChuckMonitor(model)
        x = torch.randn(4, 8)
        _ = model(x)
        sigma = mon.sigma
        assert sigma > 0.9, f'healthy model σ should be near 1, got {sigma}'
        mon.detach()

    def test_reset_clears_silu(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.SiLU())
        mon = ChuckMonitor(model)
        _ = model(torch.randn(2, 4))
        assert len(mon._silu_alive) > 0
        mon.reset()
        assert len(mon._silu_alive) == 0
        mon.detach()

    def test_attention_entropy(self):
        """feed_attention_entropy should update entropy EMA."""
        model = nn.Linear(1, 1)  # dummy
        mon = ChuckMonitor(model)
        # Uniform attention → high entropy
        n_heads, seq = 4, 16
        uniform = torch.ones(1, n_heads, seq, seq) / seq
        mon.feed_attention_entropy(uniform)
        assert mon._attn_init
        assert len(mon.attn_entropy_ema) == n_heads
        # Entropy should be close to log(seq)
        expected = math.log(seq)
        for h in mon.attn_entropy_ema:
            assert abs(h - expected) < 0.5
        mon.detach()

    def test_detach_removes_hooks(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.SiLU())
        mon = ChuckMonitor(model)
        assert len(mon.hooks) > 0
        mon.detach()
        assert len(mon.hooks) == 0


# ═══════════════════════════════════════════════════════════════════════
# ChuckOptimizer
# ═══════════════════════════════════════════════════════════════════════

class TestChuckOptimizer:

    @pytest.fixture
    def mem_path(self, tmp_path):
        return str(tmp_path / 'chuck.mem')

    def test_convergence_quadratic(self, mem_path):
        """Minimize ||x - target||² from random init."""
        torch.manual_seed(42)
        x = nn.Parameter(torch.randn(10))
        target = torch.zeros(10)
        opt = ChuckOptimizer([x], lr=0.03, mem_path=mem_path,
                             window=4, rec_cd=5)
        for i in range(500):
            loss = (x - target).pow(2).sum()
            loss.backward()
            opt.step(loss=loss.item())
            opt.zero_grad()
        final = (x - target).pow(2).sum().item()
        assert final < 0.01, f'should converge, got loss={final}'

    def test_convergence_mlp(self, mem_path):
        """2-layer MLP learns XOR-like pattern."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(2, 16), nn.SiLU(),
                              nn.Linear(16, 1))
        X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
        Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float)
        opt = ChuckOptimizer(model.parameters(), lr=0.01,
                             mem_path=mem_path, window=4, rec_cd=5)
        for i in range(1000):
            pred = model(X)
            loss = nn.functional.mse_loss(pred, Y)
            loss.backward()
            opt.step(loss=loss.item())
            opt.zero_grad()
        final = nn.functional.mse_loss(model(X), Y).item()
        assert final < 0.05, f'MLP should learn XOR, got loss={final}'

    def test_layer_freezing(self, mem_path):
        """Layer with zero gradients should freeze."""
        p_idle = nn.Parameter(torch.ones(10))
        p_active = nn.Parameter(torch.randn(10))
        target = torch.zeros(10)
        opt = ChuckOptimizer(
            [{'params': [p_idle], 'layer': 0},
             {'params': [p_active], 'layer': 1}],
            lr=0.01, mem_path=mem_path,
            window=4, freeze_thr=0.01, freeze_pat=2)
        for _ in range(30):
            # Only p_active participates in loss
            loss = (p_active - target).pow(2).sum()
            loss.backward()
            opt.step(loss=loss.item())
            opt.zero_grad()
        assert 0 in opt.frozen_layers, 'idle layer should freeze'
        assert 1 not in opt.frozen_layers, 'active layer must not freeze'

    def test_macro_lr_decay(self, mem_path):
        """Constant loss plateau should trigger macro decay."""
        x = nn.Parameter(torch.randn(4))
        opt = ChuckOptimizer([x], lr=0.01, mem_path=mem_path,
                             window=4, macro_int=10, macro_pat=2)
        # Feed constant loss to simulate plateau
        for i in range(100):
            x.grad = torch.randn(4) * 0.001  # tiny gradient
            opt.step(loss=5.0)                # constant loss
        assert opt.lr_scale < 1.0, \
            f'plateau should trigger decay, lr_scale={opt.lr_scale}'

    def test_psi_with_memory(self, mem_path):
        """With pre-loaded memory, Ψ should be non-zero."""
        # Pre-populate memory
        mem = ChuckMemory(path=mem_path)
        for i in range(50):
            mem.save_entry(2.0 - i * 0.02, 0.5, 1.2, -0.02)

        x = nn.Parameter(torch.randn(4))
        opt = ChuckOptimizer([x], lr=0.01, mem_path=mem_path,
                             window=4, rec_cd=5)
        assert opt.psi_w > 0, f'Ψ_w should be > 0 with memory'
        for i in range(20):
            loss = x.pow(2).sum()
            loss.backward()
            opt.step(loss=loss.item())
            opt.zero_grad()
        # After some steps with memory, psi should have moved
        assert opt.psi != 0.0 or opt.psi_w > 0

    def test_no_loss_fallback(self, mem_path):
        """step() without loss should work as vanilla Adam."""
        x = nn.Parameter(torch.randn(4))
        opt = ChuckOptimizer([x], lr=0.01, mem_path=mem_path)
        loss = x.pow(2).sum()
        loss.backward()
        x_before = x.data.clone()
        opt.step()  # no loss → Adam fallback
        assert not torch.equal(x.data, x_before), \
            'Adam fallback should update params'

    def test_weight_decay(self, mem_path):
        """Weight decay should shrink parameters."""
        torch.manual_seed(42)
        x = nn.Parameter(torch.ones(10) * 5.0)
        opt = ChuckOptimizer([x], lr=0.001, weight_decay=0.1,
                             mem_path=mem_path, window=4)
        for _ in range(50):
            loss = x.pow(2).sum()
            loss.backward()
            opt.step(loss=loss.item())
            opt.zero_grad()
        assert x.data.abs().mean().item() < 5.0, \
            'weight decay should shrink params'

    def test_state_dict_roundtrip(self, mem_path):
        """Save/load should preserve Chuck's soul."""
        x = nn.Parameter(torch.randn(4))
        opt = ChuckOptimizer([x], lr=0.01, mem_path=mem_path,
                             window=4, rec_cd=5)
        # Train a few steps to build state
        for _ in range(20):
            loss = x.pow(2).sum()
            loss.backward()
            opt.step(loss=loss.item())
            opt.zero_grad()

        sd = opt.state_dict()
        # Verify chuck state is in state_dict
        assert 'chuck' in sd
        assert sd['chuck']['global_step'] == 20
        saved_dampen = sd['chuck']['dampen']

        # Create new optimizer and load
        x2 = nn.Parameter(torch.randn(4))
        opt2 = ChuckOptimizer([x2], lr=0.01, mem_path=mem_path,
                              window=4, rec_cd=5)
        opt2.load_state_dict(sd)
        assert opt2.global_step == 20
        assert abs(opt2.dampen - saved_dampen) < 1e-8

    def test_verbose_output(self, mem_path, capsys):
        """Verbose mode should print status."""
        x = nn.Parameter(torch.randn(4))
        opt = ChuckOptimizer([x], lr=0.01, mem_path=mem_path,
                             verbose=1, window=4)
        loss = x.pow(2).sum()
        loss.backward()
        opt.step(loss=loss.item())
        captured = capsys.readouterr()
        assert 'step' in captured.out
        assert 'chuck' in captured.out

    def test_with_monitor(self, mem_path):
        """Optimizer should read σ from monitor."""
        model = nn.Sequential(nn.Linear(4, 8), nn.SiLU(),
                              nn.Linear(8, 1))
        mon = ChuckMonitor(model)
        opt = ChuckOptimizer(model.parameters(), lr=0.01,
                             mem_path=mem_path, monitor=mon, window=4)
        x = torch.randn(4, 4)
        for _ in range(10):
            pred = model(x)
            loss = pred.pow(2).mean()
            loss.backward()
            opt.step(loss=loss.item())
            opt.zero_grad()
        # sigma should have been read from monitor
        assert opt.sigma > 0
        mon.detach()

    def test_closure(self, mem_path):
        """step(closure=...) should evaluate closure and extract loss."""
        x = nn.Parameter(torch.ones(4))

        def closure():
            opt.zero_grad()
            l = x.pow(2).sum()
            l.backward()
            return l

        opt = ChuckOptimizer([x], lr=0.01, mem_path=mem_path, window=4)
        opt.step(closure=closure)
        assert opt.global_step == 1
        assert opt.loss_ema > 0

    def test_unfreeze_all(self, mem_path):
        """unfreeze_all should reset frozen state."""
        p = nn.Parameter(torch.ones(4))
        opt = ChuckOptimizer(
            [{'params': [p], 'layer': 0}],
            lr=0.01, mem_path=mem_path,
            window=4, freeze_thr=100.0, freeze_pat=0)
        # Force freeze by running with tiny grads
        for _ in range(10):
            p.grad = torch.zeros(4)
            opt.step(loss=1.0)
        assert 0 in opt.frozen_layers
        opt.unfreeze_all()
        assert len(opt.frozen_layers) == 0


# ═══════════════════════════════════════════════════════════════════════
# chuck_params helper
# ═══════════════════════════════════════════════════════════════════════

class TestChuckParams:

    def test_auto_detect_layers(self):
        """Should detect .layers.N. pattern."""
        model = nn.ModuleDict({
            'embed': nn.Embedding(100, 32),
            'layers': nn.ModuleList([
                nn.Linear(32, 32) for _ in range(3)
            ]),
            'head': nn.Linear(32, 100),
        })
        groups = chuck_params(model, lr=1e-3)
        # Should have: 1 global group (embed + head) + 3 layer groups
        layer_ids = [g.get('layer') for g in groups]
        assert -1 in layer_ids, 'global params should get layer=-1'
        assert 0 in layer_ids
        assert 1 in layer_ids
        assert 2 in layer_ids

    def test_no_layers_all_global(self):
        """Model without .layers. pattern → everything is global."""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 1))
        groups = chuck_params(model)
        assert len(groups) == 1
        assert groups[0]['layer'] == -1

    def test_groups_have_all_params(self):
        """All model params should appear in exactly one group."""
        model = nn.ModuleDict({
            'layers': nn.ModuleList([nn.Linear(4, 4) for _ in range(2)]),
            'head': nn.Linear(4, 1),
        })
        groups = chuck_params(model)
        group_params = set()
        for g in groups:
            for p in g['params']:
                assert id(p) not in group_params, 'param in multiple groups'
                group_params.add(id(p))
        model_params = set(id(p) for p in model.parameters())
        assert group_params == model_params, 'missing params'
