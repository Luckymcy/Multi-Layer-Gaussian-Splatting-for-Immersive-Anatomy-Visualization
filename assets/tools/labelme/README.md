# Labelme 本地标注工具

通过官方PyPI安装免费的开源Python版本 Labelme 7.9.1，独立Conda环境为 `C:/ProgramData/Anaconda3/envs/labelme`（Python 3.12）。本工具与现有高斯训练环境分开安装。

安装验证：版本命令与依赖检查通过；使用离屏界面成功打开精选帧、识别12张图片，测试未保存标注。

双击本目录的 `start_labelme.cmd`：

- 打开 `data/real/surgery_001/annotation_seed/frames/` 中的精选图片。
- 标注JSON保存到同级 `annotations/`，首次启动时自动创建。
- 可用标签为 `fat`（已确认的脂肪区域）和 `exclude`（要排除的区域）。
- 用多边形圈区域，保存标注；原图不改动。没有生成或伪造任何人工标注。

也可以在终端中运行：

```powershell
conda activate labelme
labelme
```

安装命令记录：

```powershell
conda create -n labelme python=3.12 pip -y
C:/ProgramData/Anaconda3/envs/labelme/python.exe -m pip install --index-url https://pypi.org/simple labelme==7.9.1
```

此处使用开源Python安装方式，项目官网另外提供付费独立安装包，未购买或安装该版本。手工多边形标注不需要下载AI分割模型。

来源：https://github.com/wkentaro/labelme 、https://pypi.org/project/labelme/ 。
