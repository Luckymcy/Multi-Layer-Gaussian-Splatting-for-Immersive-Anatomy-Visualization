# Research workflow and repository layout

This fork separates CT-derived scene construction from real-image appearance learning. It preserves the upstream training baseline and keeps optional experiments in separate entry points.

```text
data/ct/                     local DICOM, NIfTI, masks and teacher datasets
data/real/                   local real images, annotations and derived patches
output/                      local Gaussian models and experiment results
archive/                     local screening, diagnostic and migration artifacts
tools/annotations/           reusable Labelme-to-mask and RGB patch tools
tools/video/                 reusable sparse video screening and extraction tools
assets/tools/labelme/        local annotation launch helper and installation notes
assets/setup/                setup instructions; downloaded binaries are ignored
Multi-Layer-Anatomy-GS-*/     original code repositories as Git submodules
```

Datasets, annotations, patient/source manifests, models, videos, screenshots and local handoff notes are excluded from Git. A clone provides code and example configurations, not the development dataset or a claim of reproducibility without separately supplied inputs.

## 1. CT-derived representation

The data-generation submodule contains CT/HU sampling and optional heart-adjacent-fat geometry and teacher-generation scripts. Inputs and coordinate conventions must be checked on the supplied CT. Candidate fat ROIs are not automatically validated epicardial labels.

The training submodule retains the original teacher-image fitting procedure and optional geometry diagnostics. Geometry modifications are separate from the fixed-structure appearance experiment. Trained Gaussian PLY files and checkpoints belong in `output/`, not in version control.

## 2. Shared RGB LUT

`Multi-Layer-Anatomy-GS-Training/appearance/lut.py` learns a multi-control-point HU-to-RGB table per tissue. Geometry and opacity stay fixed; selected tissue RGB replaces its SH appearance. Unit tests are grouped in `tests/`, and the current executable is `python run.py appearance train`. The earlier standalone synthetic comparison/validation driver has been removed; see `appearance/README.md` for current gradient checks and invocation.

Fresh checkouts must apply the source patches documented in the training repository's `patches/README.md` before building the local rasterizer interface. Compiled extensions are not included.

## 3. Real-image preparation

Use Labelme to mark the target tissue and exclusion regions in original-resolution images. The preparation tool rasterizes supplied labels, preserves original RGB pixels, and records source image/annotation hashes and crop coordinates. It generates candidate patches and review previews without modifying raw data.

```powershell
python -m unittest discover -s tools/annotations -v
python tools/annotations/prepare_fat_dataset.py --seed-dir /path/to/annotation_seed --output /path/to/new_dataset_version
```

See `tools/annotations/README.md` for input/output conventions. Geometry validity of a mask does not certify its tissue content. Candidate patches require visual acceptance; sampled frames from one operation do not form independent patients.

## 4. Unpaired LUT training

The training repository separates `geometry/`, `appearance/` and `evaluation/`, using `run.py` as a unified task dispatcher. `python run.py appearance train` alternates discriminator updates with LUT-only updates through the shared differentiable Gaussian renderer, with all-layer contribution masks for generated-patch selection, frozen-model checks and full-state resume. Engineering checks and a 5000-step single-operation experiment have completed; useful real-domain learning has not been established, and the candidate real patches remain pending visual acceptance. Real and rendered patches must match tissue, visible surface and scale; cross-patient pixelwise L1/SSIM is not appropriate.

One tissue LUT is the first experiment. A LUT does not create independent spatial texture or uniquely separate lighting and material. Existing synthetic checks and teacher-fit metrics do not establish real-domain success, anatomical accuracy or cross-patient generalization.

`tools/training/summarize_lut_run.py` creates loss/gradient plots, LUT curves and fixed-view comparisons from a run directory without modifying its inputs. Generated plots and checkpoints remain local.

## 5. Unity LUT comparison

The Unity package adds JSON import and a version selector to `GaussianSplatRenderer`, plus an optional desktop Game-view comparison panel. Exported `hu` and `final_linear_rgb` controls replace RGB only on the explicitly selected layer; selecting Original / manual restores the prior display settings. See [the Unity import guide](../Multi-Layer-Anatomy-GS-Unity-Rendering/docs/learned-lut-json.md). Parsing/interpolation checks passed with the actual exported JSON, but full Unity Editor/GPU/UI validation has not run on this machine.

## Checks and working environments

- Annotation utilities require NumPy and Pillow; tests use synthetic fixtures and need no patient data.
- LUT CPU tests require PyTorch. Full renderer checks additionally need CUDA extensions and user-supplied model/cameras.
- CT scripts require the data-generation environment (including SimpleITK/VTK/SciPy as used by each entry point).
- Use an independent environment for Labelme; the helper README describes the free Python installation route.

No command in this document starts a long training task by default. Use explicit input/output arguments and new output directories for experiments.
