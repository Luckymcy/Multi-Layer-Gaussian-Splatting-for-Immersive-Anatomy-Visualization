# 真实影像标注处理工具

本目录独立于data generation，仅依赖NumPy和Pillow，不调用Labelme内部API，不启动训练。

```powershell
# 在 E:/3dguass 执行，使用已有labelme环境
& C:/ProgramData/Anaconda3/envs/labelme/python.exe -m unittest discover -s tools/annotations -v
& C:/ProgramData/Anaconda3/envs/labelme/python.exe tools/annotations/prepare_fat_dataset.py --seed-dir data/real/surgery_001/annotation_seed --output data/real/surgery_001/training_dataset/v001
```

输出目录必须不存在；重新标注后选择v002等新版本，不覆盖输入或旧结果。先检查全部输入JSON、图像尺寸、来源SHA256、坐标和标签，再生成输出。支持polygon、circle和rectangle；同类形状取并集，exclude优先，与绘制顺序无关。

## 输出

```text
training_dataset/v001/
├─ masks/
│  ├─ fat/                人工fat并集
│  ├─ exclude/            人工exclude并集
│  ├─ valid/              fat减exclude
│  ├─ quality_exclude/    自动亮像素候选及膨胀，仅用于采样
│  └─ sampling_valid/     valid内缩并去除quality_exclude
├─ patches/{64,96,128}px/ 原图像素裁剪，未放大、未遮黑、未调色
├─ review/
│  ├─ overlays/           每帧叠加预览
│  ├─ mask_overview.jpg   全体掩膜叠加联系表
│  └─ patch_examples.jpg  各帧各尺寸裁剪示例
├─ config.json            参数、工具哈希及依赖版本
├─ frame_manifest.json    每帧标注哈希、掩膜位置和像素数
├─ patch_manifest.json    每张裁剪的原图/标注/视频/时间/框坐标
├─ summary.json
└─ COMPLETED.json         完整生成结束标记
```

二值掩膜为原图同尺寸L模式PNG，0=无效，255=有效。`bbox_xyxy_exclusive`是原图坐标，右/下边界不包含，与NumPy切片一致。叠加预览绿色=人工有效区域、红色=人工排除、洋红色=自动亮部候选、青色=采样有效区域内缘。预览图不用于训练。

## 采样约束与限制

- 默认64/96/128三种原生尺寸，每帧每尺寸最多4张；同一判别器batch应明确选择相同尺寸，不能把尺寸差异当域差异。
- 每个裁剪100%落在sampling_valid；默认内缩4像素，为人工边界和排除边界留余量。
- 自动亮部候选默认 `max(RGB)>=250 且 min(RGB)>=220`，膨胀1像素后排除。这是启发式亮像素过滤，不等价于完整镜面高光检测，也不更改人工标注。可用 `--no-highlight-filter` 关闭；参数完整写入config。
- 每尺寸候选网格步长为边长一半；固定随机seed选择。单帧所有尺寸之间最大IoU默认0.15，减少反复裁同一区域。
- 精确重复按RGB像素SHA256排除；近重复按同尺寸裁剪的16×16RGB缩略图平均绝对差<=3（0–255数值）排除。这是有限的去重启发式，不证明独立组织或患者样本。
- 若后续真实与生成图需要统一颜色空间，在训练读取端显式处理；此工具保留原RGB数值，没有做gamma转换。
- 全部来自SURGERY_001，不自动拆分训练/测试。裁剪数不能解释为独立患者数。

输出是**供复核的训练候选**，`training_ready=false`、`review_status=pending_visual_acceptance`。掩膜包含用户给出的语义，代码不能辨别被圈入的血管、非脂肪组织或漏标器械。应看叠加图与裁剪，修正源JSON后生成新版本。技术有效性检查和组织语义确认是两项独立工作。

当前v001的review、verification和COMPLETED过程文件已移至 `archive/2026-11-09/fat_dataset_v001_review/`，训练数据目录只保留裁剪、掩膜和必要清单。上面的树说明工具初次生成的完整产物；新版本完成检查后也可按此方式整理。
