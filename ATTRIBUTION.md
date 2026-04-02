# Attribution and upstream relationships

This document clarifies how this repository relates to other projects under `D:\Project\CursorProj\…` **before you publish on GitHub**. It is **not** a substitute for the reference list in your manuscripts (MCA / AIS); it helps **license hygiene** and **good-faith disclosure**.

## 1. `model_based_diffusion` (LeCAR-Lab — *trajectory optimization* MBD)

- **Upstream:** [LeCAR-Lab/model-based-diffusion](https://github.com/LeCAR-Lab/model-based-diffusion) — “Model-based Diffusion for Trajectory Optimization”, **Apache License 2.0**.
- **Stack:** JAX / Brax, **RL-style trajectory** planning (MuJoCo-style environments).
- **This repository:** PyTorch implementation of **degradation time-series imputation** (`models/diffusion/mbd_degradation.py`, etc.). There is **no JAX/Brax code** and **no copied source files** from that upstream tree in this release.
- **Relationship:** **Conceptual** alignment with the *model-based diffusion* idea (e.g. using a dynamics/degradation model inside the sampling loop; hyperparameters such as `Nsample`, `temp_sample`, cosine-style noise schedules are common in diffusion literature).
- **Recommended action:**
  - In the **paper**, cite the MBD trajectory paper if you discuss generic MBD background (e.g. arXiv:2407.01573 / official publication).
  - In this **repo**, this file is sufficient for transparency; **Apache NOTICE** is only mandatory if you **distribute verbatim Apache-licensed source** from LeCAR (currently **not** the case).

## 2. `Degradation_Model_Based_Diffusion_v1.0` (internal “project 3” MBD backbone)

- **Role:** Snapshot / workspace that hosts the **PyTorch** package `Models/model_based_diffusion/` (`mbd_degradation.py`, `degradation_models.py`, `utils.py`). This is the **direct code ancestor** of `models/diffusion/mbd_degradation.py` in `Component_DT_PINN_V1.1` and of the same module in **this** public release (modulo formatting, shortened docstrings, and an explicit source note in the docstring).
- **Relationship to LeCAR `model_based_diffusion`:** **Name collision only.** The v1.0 module is **not** JAX code from LeCAR; internal docs (`MBD_TRANSPLANT_VERIFICATION.md`) state **PyTorch-only, no JAX**.
- **Relationship to `diffusion_ts`:** **Parallel / complementary.** `diffusion_ts` centers on **Diffusion-TS** (`Diffusion_TS` transformer diffusion). v1.0 centers on **degradation-model–guided MBD** sampling. They can be used in the same lab workflow but are **different model classes**.
- **License:** The v1.0 tree **had no root `LICENSE` file** at audit time. If the authors are the same research group as this release, align v1.0 with your chosen license (e.g. MIT) when you publish either tree; no **external** attribution is required beyond your own copyright.
- **Recommended action:** Treat this folder as **proprietary / first-party lineage**; optional citation in a thesis or tech report: “MBD degradation imputation module (v1.0) → Component_DT_PINN”.

## 3. `diffusion_ts` (degradation / Diffusion-TS style imputation)

- **Role in your lab notes:** Internal integration report (`report-项目集成检查-2026-02-15.md` in the private Component project) traces the **PyTorch MBD degradation backbone** to **“project 3 — Degradation_Model_Based_Diffusion”** — that maps to **`Degradation_Model_Based_Diffusion_v1.0`** above, not to the Diffusion-TS neural core inside `diffusion_ts`.
- **`diffusion_ts` itself** bundles **Y-debug-sys/Diffusion-TS** (MIT) plus other third parties; see prior audit.
- **This public repository:** Source headers that say “From project 3” refer to **`Degradation_Model_Based_Diffusion_v1.0`**, not to Diffusion-TS.
- **Recommended action:**
  - Keep **Diffusion-TS** obligations (MIT notice, paper citation) confined to the **`diffusion_ts`** distribution if you publish it.
  - Do **not** conflate Diffusion-TS licensing with the **MBD** files inherited from v1.0 (first-party unless you imported someone else’s MBD code into v1.0).

## 4. `PINN` (folder of tutorials / demos)

- Contains multiple independent repos (e.g. Poisson PINN notebook, PIML degradation diagnostics, TCN-related material). **Licenses differ** (e.g. MIT in at least one subfolder).
- **This repository:** No automated scan was run for byte-level overlap. **We did not find** obvious imports or paths pointing into `PINN` in the released tree.
- **Recommended action:**
  - If **no files** from `PINN` were pasted into this codebase: **no repository-level attribution** is strictly required; your **paper** already cites PINN / physics-informed ML literature.
  - If you **did** copy specific functions/notebook code, retain the **original copyright and license notice** at the top of those files (MIT requires preserving the notice).

## 5. Summary table

| Path | License (typical) | Code in this repo? | Typical action |
|------|-------------------|--------------------|----------------|
| `model_based_diffusion` (LeCAR) | Apache-2.0 | No verbatim JAX code | Cite paper if used conceptually; optional README pointer |
| `Degradation_Model_Based_Diffusion_v1.0` | *set by authors* | **Yes — MBD core lineage** | First-party; add `LICENSE` to v1.0 if publishing |
| `diffusion_ts` | Diffusion-TS MIT + others | Different stack (Diffusion-TS) | MIT + citations if you publish that repo |
| `PINN` | Mixed (e.g. MIT) | Unlikely; verify if unsure | Preserve notices only if you copied source |

## 6. This repository’s license

The default open release uses **MIT** (`LICENSE`). MIT is **compatible** with combining **your own** code with **properly attributed** third-party snippets; it does **not** automatically satisfy Apache-2.0 **NOTICE** requirements — those apply only if Apache **source** is redistributed here (currently not claimed).

---

*Maintainers: keep §3 aligned with the `diffusion_ts` audit; keep §2 aligned if `Degradation_Model_Based_Diffusion_v1.0` gains a public `LICENSE`.*
