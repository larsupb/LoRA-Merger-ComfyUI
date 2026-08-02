# Semantic Adapter System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Living document.** Phases 1–3 are already built; this plan tracks what remains. Update the status table and task checkboxes in place as work proceeds — do not create separate status files.

**Goal:** Replace static heuristic LoRA merging (weighted average, TIES, SLERP, …) with a *learned* composition — a small trainable adapter per layer that decides how much of LoRA A vs LoRA B to use, conditioned on semantic importance maps from Integrated Gradients. At inference there is no merge step: base model + frozen LoRAs + a trained adapter package combine at runtime via forward hooks.

**Architecture:** Two adapter families routed per layer by `AdapterRegistry`. **Rank-space** intercepts between the LoRA's A and B matrices and gates the low-rank codes (`ẑ = g ⊙ z_A + (1−g) ⊙ z_B`, then `Δ̂ = B_shared ẑ`). **Output-space** gates the finished deltas (`Δ̂ = g ⊙ Δ_A + (1−g) ⊙ Δ_B + ε`). Gates come from the layer's semantic importance vector, `g = σ(W_g · s_ℓ)`. `UNetDeltaProvider` produces real per-layer deltas from a live diffusers pipeline by differencing a LoRA-enabled pass against a LoRA-free base pass. Only gate networks and residual MLPs train; base UNet, CLIP and all LoRAs stay frozen, optimized jointly under one global objective with a three-stage curriculum.

**Tech Stack:** Python, PyTorch, diffusers (SD1.5 first, SDXL next, DiT/Flux later), ComfyUI custom node.

**Repo test reality:** On the current mainline pytest works — ComfyUI modules (`comfy`, `folder_paths`, `comfy_extras`, `nodes`) are mocked in `tests/conftest.py`; see `RUN_TESTS.md` (238 passed). The semantic branch carries its own older `pytest.ini` and `tests/conftest.py`, so confirm collection there before trusting any pass/fail count. Run from the custom-node root with `/home/lars/SD/Apps/ComfyUI/.venv/bin/python -m pytest tests/`.

**Design doc:** none — no spec was ever written for this work. The architecture summary above and the Current Status section below are the authoritative description.

**Branch:** all code lives on `feature/gradient_based_merging` (key commit `c95f89d` "semantic adapter", +7271 lines across 31 files). Nothing is on `main` or on the current `feat/gta-delta-space-merge` line. That branch sits 6 commits past the merge-base while the mainline has moved 51 commits ahead (GTA delta-space work, node renames, VRAM offload, `src/merge/` restructuring) — **Task 1 must start with a rebase.**

---

## File Structure

Existing (on `feature/gradient_based_merging`):

- `src/semantic/adapter/base.py` — `BaseAdapter` ABC.
- `src/semantic/adapter/rank_space.py` — `RankSpaceAdapter`. Built, instantiated, **never trains** (see Current Status).
- `src/semantic/adapter/output_space.py` — `OutputSpaceAdapter`. The only adapter currently training.
- `src/semantic/adapter/registry.py` — `AdapterRegistry`, per-layer adapter-type routing.
- `src/semantic/adapter/hook_manager.py` — `ForwardHookManager`, UNet injection.
- `src/semantic/training/delta_provider.py` — `UNetDeltaProvider`, real per-layer deltas + hook budget.
- `src/semantic/training/trainer.py` — `AdapterTrainer`, joint training loop.
- `src/semantic/training/losses.py` — teacher / dominance / residual / consistency.
- `src/semantic/training/curriculum.py` — three-stage schedule.
- `src/semantic/training/semantic_key_projection.py` — layer key → semantic group.
- `src/semantic/training/data_loader.py` — prompt/timestep sampling.
- `src/semantic/inference/pipeline.py` — `AdapterInferencePipeline`.
- `src/semantic/serialization/format.py` — `AdapterSerializer`.
- `test_gradient_analyzer_with_semantic_adapter_sd15_e2e.py` (repo root) — E2E driver.
- `tests/` — 21 files, incl. `test_semantic_adapter_e2e.py`, `test_semantic_adapter_validation.py`, `test_hook_integration.py`, `test_adapter_initialization.py`, `test_adapter_dtype_safety.py`, `test_semantic_key_projection.py`.

To create:

- `src/nodes_semantic_adapter.py` — NEW, `PM_SemanticAdapterTrainer` + `PM_SemanticAdapterInference` (Task 5).
- `tests/test_semantic_adapter_nodes.py` — NEW, node wiring guard (Task 5).
- `docs/SEMANTIC_ADAPTER_GUIDE.md` — NEW, user documentation (Task 6).

---

## Current Status

| Phase | Scope | Status |
|---|---|---|
| 1 | Adapter architecture — base, rank-space, output-space, registry | ✅ Implemented |
| 2 | Training infra — losses, curriculum, trainer, checkpoints, synthetic deltas | ✅ Implemented |
| 3 | UNet hook integration — real deltas from a live SD1.5 pipeline | ✅ Runs end-to-end; ⚠️ convergence never verified |
| 4 | E2E testing & validation | 🟡 21 test files exist; full-workflow validation not signed off |
| 5 | ComfyUI nodes | ❌ Not started — `nodes_semantic_adapter.py` does not exist |
| 6 | Documentation & release | ❌ Not started |

✅ below means *observed in a run log*. Anything unobserved is marked explicitly.

### Losses and curriculum as built

| Loss | Form | Purpose |
|---|---|---|
| Teacher alignment | `‖Δ̂ − Δ_sem‖²` | never do worse than the static semantic merge |
| Feature dominance | `Σ_f r_f ‖Δ̂ − Δ_src(f)‖²` | requested feature comes from the requested LoRA |
| Residual economy | `relu(‖Δ̂‖/‖Δ_sem‖ − 1)²` | don't inflate the delta |
| Output consistency | `‖ε̂ − ε_target‖²` | preserve global image quality |

`L = Σ_t w_t Σ_ℓ [ L_teacher + λ_dom·L_dom + λ_res·L_res + λ_out·L_out ]`

| Stage | Epochs | λ_teacher | λ_dom | λ_res | λ_out | LR |
|---|---|---|---|---|---|---|
| alignment | 50 | 1.0 | 0.0 | 0.1 | 0.0 | 1e-4 |
| dominance | 50 | 1.0 | 0.5 | 0.1 | 0.0 | 5e-5 |
| full | 50 | 0.5 | 1.0 | 0.1 | 0.3 | 1e-5 |

### The delta-isolation invariant — must not regress

```python
self._enable_lora(None)                 # base pass — NO LoRA
base_acts = self._run_and_capture(...)
for lora_name in self.lora_names:
    self._enable_lora(lora_name)
    lora_acts = self._run_and_capture(...)
    delta = lora_acts[key] - base_acts[key]     # true isolated delta
    assert delta.shape == base_acts[key].shape
    assert not torch.isnan(delta).any() and not torch.isinf(delta).any()
```

Reported clean across 100+ forward passes. Any refactor of `delta_provider.py` must preserve it.

### Issues already resolved in Phase 3

Phase 3 swapped synthetic deltas for real ones from a live SD1.5 pipeline. Synthetic deltas were "forgiving" — they could stand in for either representation — while real activations are physically output-space, which broke several assumptions at once.

1. **Adapter type mismatch → `Trained 0 layers`.** The registry created 160 `RankSpaceAdapter`s for attention layers while the provider emitted output-space deltas from 96 hooked Linear layers, and the 32 `OutputSpaceAdapter`s sat on unhooked Conv2d layers. Overlap: zero. Fixed by aligning `registry.py:_determine_adapter_type` with the hook predicate:

   ```python
   if layer_key.endswith('.to_q') or layer_key.endswith('.to_v') or layer_key.endswith('.to_out'):
       return "output_space"          # matches UNetDeltaProvider._should_hook()
   elif is_attention_layer(layer_key) or is_mlp_layer(layer_key):
       return "rank_space"            # not hooked in Phase 3
   else:
       return "output_space"
   ```

   Rejected alternatives: reconstructing rank-space deltas from outputs (not uniquely invertible, numerically fragile); a hybrid fallback inside `RankSpaceAdapter` (postponed — the per-layer type mechanism already supports it).

2. **Tensor-shape incompatibility.** Attention deltas are `[batch, seq_len, channels]` (channels-last, e.g. `[3, 6144, 320]`); `OutputSpaceAdapter` assumed conv-style channels-first. Fixed with ndim-based detection in `output_space.py:forward` — 3D linear `[B, L, C]`, 4D attention `[B, L, S, C]` (already channels-last, no permute), 5D conv2d `[B, L, C, H, W]` (permute to channels-last and back).

3. **Semantic vector missing its batch dimension.** Built as 1D `[num_features]`, but the gate network expects `[batch_size, num_features]`. Fixed by reading batch size off the first delta.

4. **VRAM overflow on a 7.6 GB GPU.** Three leaks: the residual MLP over 6144 sequence positions (now skipped when `total_spatial_size > 1000`); all activations resident on GPU (`_store_activation` now does `.detach().cpu().to(torch.float16)`); adapter forward on GPU for long sequences (deltas, semantic vector and module move to CPU when `ndim == 3 and shape[1] > 1000`). Plus the "Tier 1" hook budget — hook only `to_q`, `to_v`, `to_out`; skip `to_k`, MLP (`ff.net`), Conv2d, residuals. **1050 → 480 hooks (−54%)**, on the rationale that `to_q` carries query intent, `to_v` semantic content, `to_out` consolidates the LoRA effect, and MLP is expensive per unit of routing specificity.

5. **Device mismatch.** Semantic importance scores could arrive as CUDA tensors while deltas sat on CPU. Fixed by coercing importance to a Python float and aligning device before accumulation.

6. **Teacher loss stuck at zero.** With 96 layers finally training, every loss still read `0.000000` — teacher targets fell back to zero tensors because semantic-map keys did not match delta-provider keys (`.base_layer` suffixes, `to_out.0` vs `to_out`, differing normalization). `training/semantic_key_projection.py` now projects verbose diffusers keys onto stable semantic groups (`down_blocks.0…attn1.to_q` → `self_attention.query`) and `trainer.py:198` calls `get_semantic_importance` through it. **Whether losses are now non-zero has never been verified — that is Task 1.**

### Reported numbers — claims to reconfirm, not re-verified

| | |
|---|---|
| Pipeline | SD1.5, 859,520,964 params, 2 LoRAs (SarahF, TessaF), fp16, cuda |
| Hooks | 480 registered, 96 producing deltas |
| Deltas | 96 layers, `[3, 6144, 320/640/1280]` |
| Layers trained | 96 / 96 |
| VRAM | ~7.0 GB before CPU offload, ~4.5 GB after (of 7.6 GB) |
| Epoch time | ~8 s before CPU offload; ~1–2 min/epoch estimated after |

### Known deviations from the intended design

Compromises made to fit a 7.6 GB GPU. Record them so they aren't mistaken for architecture:

1. **Rank-space adapters are dead code in practice.** Instantiated for `to_k` and MLP layers, but those are never hooked, so they never train. The system runs *entirely* output-space — the "hybrid" design is aspirational.
2. **The residual MLP is skipped on every attention layer** (seq_len 6144 ≫ 1000). What trains there is a gated weighted sum — closer to a learned per-layer blend weight than to nonlinear composition.
3. **`to_k`, all MLP layers and all Conv2d layers are excluded** from training.
4. **Adapter forward for long sequences runs on CPU**, the dominant per-epoch cost.
5. **Only 2 LoRAs, only SD1.5** have ever been exercised.

Consequence: the system cannot yet demonstrate its core claim — learned nonlinear composition beating a static merge. Fixing 1–3 needs VRAM headroom or gradient checkpointing, not more architecture.

---

## Task 1: Rebase and prove the training loop learns

**Files:**
- Modify: none expected (rebase conflict resolution only)
- Test: `test_gradient_analyzer_with_semantic_adapter_sd15_e2e.py`, `tests/` (21 files)

- [ ] **Step 1: Rebase onto the current mainline**

```bash
cd /home/lars/SD/Apps/ComfyUI/custom_nodes/LoRA-Merger-ComfyUI
git checkout feature/gradient_based_merging
git rebase main    # expect conflicts in src/merge/, node widget names, pytest.ini
```

Resolve toward the mainline for anything outside `src/semantic/`. The node widget renames (`ties`→`sign_consensus`, `normalize`→`average_weights`, `lambda_`→`output_scale`) and the `src/merge/` restructuring all landed after this branch forked.

- [ ] **Step 2: Get the existing suite collecting and green**

```bash
/home/lars/SD/Apps/ComfyUI/.venv/bin/python -m pytest tests/
```

Reconcile the branch's older `conftest.py`/`pytest.ini` with the mainline mocking setup. Record the real pass count; do not proceed on a suite that does not collect.

- [ ] **Step 3: Run the E2E driver and capture logs**

```bash
/home/lars/SD/Apps/ComfyUI/.venv/bin/python \
  test_gradient_analyzer_with_semantic_adapter_sd15_e2e.py 2>&1 | tee /tmp/phase3_run.log
```

- [ ] **Step 4: Assert the four things that were never confirmed**

From `/tmp/phase3_run.log`:
- `Trained 96 layers (out of 96 delta layers)` — not 0.
- Teacher loss **> 0** on epoch 1. This is the fix from Current Status item 6, still unproven.
- Adapter parameters change between epochs — snapshot a few gate weights and diff them. A printed loss number is not evidence of gradient flow.
- No NaN/Inf in gradients.

- [ ] **Step 5: Commit the rebase + any fixes**

```bash
git add -A && git commit -m "chore(semantic): rebase adapter work onto mainline; verify training loop"
```

**Exit criterion:** a logged loss curve that decreases. Until it exists, every ✅ in Phase 3 means "runs", not "works".

---

## Task 2: Full training run

**Files:**
- Modify: `src/semantic/training/trainer.py` (logging only, if needed)
- Test: `test_gradient_analyzer_with_semantic_adapter_sd15_e2e.py`

- [ ] **Step 1: Run all 150 epochs across the three curriculum stages**

Confirm automatic stage transitions fire at epochs 50 and 100 and that the optimizer LR actually changes (1e-4 → 5e-5 → 1e-5).

- [ ] **Step 2: Checkpoint round-trip mid-run**

Save at epoch 60, reload, confirm curriculum stage index, epoch counter, optimizer state and adapter weights all restore.

- [ ] **Step 3: Record gate specialization**

Log per-layer gate statistics. Do gates diverge from 0.5, and do they differ between `attn1` (self) and `attn2` (cross)? **Flat gates mean the adapter learned nothing useful** — report that outcome rather than hiding it.

- [ ] **Step 4: Commit results**

---

## Task 3: Validate against the static merge — STOP/GO GATE

**Files:**
- Test: `tests/test_semantic_adapter_validation.py` (extend)

This is the research question the whole project exists to answer, and it was never asked.

- [ ] **Step 1: Generate matched image sets**

Static semantic merge vs trained adapter, identical prompts and seeds.

- [ ] **Step 2: Measure feature leakage via CLIP similarity**

Does the requested feature actually come from the requested LoRA more often than under the static merge?

- [ ] **Step 3: Per-layer contribution stats and adapter magnitude monitoring**

- [ ] **Step 4: Measure inference overhead**

Hooked pipeline vs a pre-merged LoRA, same sampler settings.

- [ ] **Step 5: Decide**

If the adapter does **not** beat the static merge, stop here and record the negative result in this file. Do not proceed to Tasks 4–6. A parked spike with a clear answer is worth more than shipped nodes for an unvalidated method.

---

## Task 4: Lift the VRAM compromises

Only if Task 3 is positive.

**Files:**
- Modify: `src/semantic/training/delta_provider.py`, `src/semantic/adapter/output_space.py`, `src/semantic/adapter/registry.py`

- [ ] **Step 1: Add gradient checkpointing** so the residual MLP can run on attention layers (removes deviation 2).
- [ ] **Step 2: Re-enable Conv2d hooks** (~+500 MB) to train the 32 idle `OutputSpaceAdapter`s.
- [ ] **Step 3: Optional — capture A/B intermediates** in `UNetDeltaProvider` for true rank-space training, then flip `_determine_adapter_type` back to `rank_space` for those layers. The adapter architecture needs no change; the per-layer type mechanism already supports it.
- [ ] **Step 4: Re-run Task 3 metrics** to confirm the lifted compromises actually improve results.

---

## Task 5: ComfyUI nodes

**Files:**
- Create: `src/nodes_semantic_adapter.py`
- Create: `tests/test_semantic_adapter_nodes.py`
- Modify: `__init__.py` (node registration)

Target workflow:

```
[LoRAStack] → [PM Semantic Analyzer] → [PM Semantic Adapter Trainer] → [Trained Adapter]
                     ↓
            [PM Semantic Merge Spec]

[Base Model] + [LoRAs] + [Trained Adapter] → [PM Adapter Inference] → [Generate]
```

- [ ] **Step 1: Write the failing node-wiring test** — widget names and types, following `tests/test_merge_node_names.py`.
- [ ] **Step 2: `PM_SemanticAdapterTrainer`** — inputs `lora_stack`, `semantic_maps`, `merge_spec`, `base_model`, `num_epochs`, `device`; returns `AdapterPackage`. Progress via `ThreadSafeProgressBar`.
- [ ] **Step 3: `PM_SemanticAdapterInference`** — inputs `model`, `adapter_package`, `lora_stack`; returns `MODEL` with hooks registered.
- [ ] **Step 4: Match current widget-naming conventions.** Do not reintroduce `ties`, `normalize` or `lambda_` — the mainline renamed them.
- [ ] **Step 5: Verify backward compatibility** — the heuristic semantic merge and all existing nodes must be untouched.
- [ ] **Step 6: Run the test, then commit.**

---

## Task 6: Documentation

**Files:**
- Create: `docs/SEMANTIC_ADAPTER_GUIDE.md`
- Modify: `README.md`

- [ ] **Step 1: Usage guide + quickstart** covering the adapter workflow end to end.
- [ ] **Step 2: Architecture notes** — adapter types, hook flow, curriculum rationale, extension guide for new architectures.
- [ ] **Step 3: Example ComfyUI workflow JSON.**
- [ ] **Step 4: README section** describing the adapter path as an optional alternative to static merging.
- [ ] **Step 5: Commit.**

---

## Open risks

| Risk | Mitigation |
|---|---|
| Diffusers doesn't expose LoRA down-projections cleanly | Blocked rank-space from day one; the output-space routing sidesteps it. Revisit only in Task 4. |
| Training instability from a four-term loss | Progressive curriculum, extensive logging. Untested at scale. |
| Hook overhead at inference | Measure in Task 3 before shipping nodes. |
| Branch rot against a fast-moving mainline | Task 1 rebases first, always. |
| Scope creep | Task 3 is a hard stop/go gate. Do not build nodes for an unvalidated method. |

---

## Done criteria

- Training demonstrably converges: teacher loss > 0, decreasing across a logged 150-epoch run, with adapter weights provably changing.
- Task 3 has produced a documented verdict — adapter beats static merge, or it does not and the project stops.
- If positive: `PM_SemanticAdapterTrainer` and `PM_SemanticAdapterInference` appear in ComfyUI and execute a full workflow.
- Adapters save and load correctly through `AdapterSerializer`.
- The heuristic semantic merge path still works unchanged.
- Test suite green on a branch rebased onto the current mainline.
