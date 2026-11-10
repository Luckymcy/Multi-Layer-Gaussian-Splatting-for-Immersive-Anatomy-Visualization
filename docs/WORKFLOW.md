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

`Multi-Layer-Anatomy-GS-Training/appearance_lut.py` learns a multi-control-point HU-to-RGB table per tissue. Geometry and opacity stay fixed; selected tissue RGB replaces its SH appearance. `validate_lut_render.py` checks the differentiable CUDA RGB path using a synthetic target and a short bounded run. See that repository's `experiments/README.md`.

Fresh checkouts must apply the source patches documented in the training repository's `patches/README.md` before building the local rasterizer interface. Compiled extensions are not included.

## 3. Real-image preparation

Use Labelme to mark the target tissue and exclusion regions in original-resolution images. The preparation tool rasterizes supplied labels, preserves original RGB pixels, and records source image/annotation hashes and crop coordinates. It generates candidate patches and review previews without modifying raw data.

```powershell
python -m unittest discover -s tools/annotations -v
python tools/annotations/prepare_fat_dataset.py --seed-dir /path/to/annotation_seed --output /path/to/new_dataset_version
```

See `tools/annotations/README.md` for input/output conventions. Geometry validity of a mask does not certify its tissue content. Candidate patches require visual acceptance; sampled frames from one operation do not form independent patients.

## 4. Planned unpaired learning

An adversarial discriminator and a real-image LUT training loop are **not yet implemented in this snapshot**. The planned loop alternates discriminator updates with LUT-only updates through the shared differentiable Gaussian renderer. Real and rendered patches must match tissue, visible surface and scale; cross-patient pixelwise L1/SSIM is not appropriate.

One tissue LUT is the first experiment. A LUT does not create independent spatial texture or uniquely separate lighting and material. Existing synthetic checks and teacher-fit metrics do not establish real-domain success, anatomical accuracy or cross-patient generalization.

## Checks and working environments

- Annotation utilities require NumPy and Pillow; tests use synthetic fixtures and need no patient data.
- LUT CPU tests require PyTorch. Full renderer checks additionally need CUDA extensions and user-supplied model/cameras.
- CT scripts require the data-generation environment (including SimpleITK/VTK/SciPy as used by each entry point).
- Use an independent environment for Labelme; the helper README describes the free Python installation route.

No command in this document starts a long training task by default. Use explicit input/output arguments and new output directories for experiments.
