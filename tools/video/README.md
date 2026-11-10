# Local video screening tools

These tools prepare sparse visual review material; they do not assign tissue ground truth, infer patient identities, or establish a train/test split. Source videos are read only. All generated imagery and source manifests belong in ignored local data/archive directories.

```powershell
python tools/video/audit_real_videos.py --input /path/to/raw_videos --output /path/to/new_review
python tools/video/prepare_real_video_candidates.py --inventory /path/to/new_review/inventory.json --plan /path/to/timestamps.json --output /path/to/new_candidates
```

Requirements: OpenCV, NumPy and Pillow. `audit_real_videos.py` saves metadata, sparse original-resolution PNG samples and contact sheets. Frame timestamps are estimated from requested frame index/reported FPS; they are not a replacement for precise source PTS decoding.

The focused extraction plan has this structure:

```json
{"timestamps_seconds": {"V01": [5, 15, 30]}}
```

Video IDs come from the first inventory. Confirm patient/operation relationships in that inventory before downstream grouping; otherwise the conservative unverified group is retained. Output manifests mark candidates unassigned and not training-ready. Sharpness/brightness and whole-frame hash values are heuristic review aids, not tissue-quality labels.

The tools use the Windows Microsoft YaHei font when available and otherwise fall back to Pillow's default font. Supply a suitable system font if the fallback cannot display your source filenames.
