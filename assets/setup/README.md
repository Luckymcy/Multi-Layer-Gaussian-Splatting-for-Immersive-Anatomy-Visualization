# 安装资源与权重包

- `model_weights/`：根目录迁入的7个TotalSegmentator预训练权重ZIP，保留原文件名及内容，供安装或离线恢复。
- `installers/`：Visual Studio Build Tools下载引导程序，供重装参考。

本次没有解压、安装、删除或修改这些文件。现有Conda环境和已安装CUDA/编译工具路径保持原状。根目录 `build_cuda_extensions.cmd` 继续使用已安装工具，不读取此处的引导程序。
