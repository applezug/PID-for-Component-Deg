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

## 2. `diffusion_ts` (degradation / Diffusion-TS style imputation)

- **Role in your lab notes:** Internal integration report (`report-项目集成检查-2026-02-15.md` in the private Component project) traces the **PyTorch MBD degradation backbone** to a **“project 3 — Degradation_Model_Based_Diffusion”** lineage, consistent with the degradation-imputation track under `diffusion_ts` / `Degradation_Data_Imputation`.
- **This repository:** Source headers still note “From project 3” in a few diffusion modules — that refers to **your prior internal consolidation**, not to a third-party obligation by itself.
- **Recommended action:**
  - If `diffusion_ts` contains **third-party** code (e.g. upstream **Diffusion-TS** or similar) that was **copied verbatim**, identify that upstream’s **license** and either (i) add a **NOTICE** / **license file** per its terms, or (ii) replace with a clean-room implementation.  
  - If the degradation MBD code is **entirely authored by your team**, no extra third-party file is required beyond honest description here.

## 3. `PINN` (folder of tutorials / demos)

- Contains multiple independent repos (e.g. Poisson PINN notebook, PIML degradation diagnostics, TCN-related material). **Licenses differ** (e.g. MIT in at least one subfolder).
- **This repository:** No automated scan was run for byte-level overlap. **We did not find** obvious imports or paths pointing into `PINN` in the released tree.
- **Recommended action:**
  - If **no files** from `PINN` were pasted into this codebase: **no repository-level attribution** is strictly required; your **paper** already cites PINN / physics-informed ML literature.
  - If you **did** copy specific functions/notebook code, retain the **original copyright and license notice** at the top of those files (MIT requires preserving the notice).

## 4. Summary table

| Path | License (typical) | Code in this repo? | Typical action |
|------|-------------------|--------------------|----------------|
| `model_based_diffusion` | Apache-2.0 | No verbatim JAX code | Cite paper if used conceptually; optional README pointer |
| `diffusion_ts` | *check upstream if any* | Evolutionary lineage only | Confirm whether any external Diffusion-TS code remains |
| `PINN` | Mixed (e.g. MIT) | Unlikely; verify if unsure | Preserve notices only if you copied source |

## 5. This repository’s license

The default open release uses **MIT** (`LICENSE`). MIT is **compatible** with combining **your own** code with **properly attributed** third-party snippets; it does **not** automatically satisfy Apache-2.0 **NOTICE** requirements — those apply only if Apache **source** is redistributed here (currently not claimed).

---

*Maintainers: update §2 after auditing `diffusion_ts` against any upstream Diffusion-TS (or related) license terms.*
