# LUT训练结果汇总

`summarize_lut_run.py`读取运行目录下artifacts，不修改训练数据、模型或检查点。依赖Matplotlib、NumPy和Pillow，可使用独立Labelme环境运行。

```powershell
python tools/training/summarize_lut_run.py --run-dir output/appearance/l7/2026-11-10_lut16_run001
```

在运行目录的reports下生成损失/梯度曲线、初始/学习后线性RGB控制点曲线和固定训练视角对照。可在训练中读取最近的完整检查点，也可在结束后更新最终图；重复运行只更新这些报告图。

曲线与训练视角不属于独立测试。损失变化不直接等同于真实感改善，最终判断仍需检查组织匹配、尺度、曝光及可见外观。
