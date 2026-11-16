# 先例与复用

文献及官方来源见 `RESEARCH.md`。显式 pitch/energy/duration：FastSpeech 2、Ctrl-P、
CtrlSpeech。参数生成函数：HyperNetworks；特征仿射调制：FiLM。构音中间表征：SPARC；
可微声学结构：DDSP、NSF。AR 历史干预：2605.27376；语音模型因果组件定位：2610.05080。

实际复用 Kokoro、Misaki/eSpeak、Parler、PyTorch、SciPy。Kokoro 的原生对齐流程有
少量改编，保留来源说明及 Apache 许可证。未复刻 Qt、声码器、Transformer 或音素化器。

项目价值在生产工程与可验证耦合，不声称这些研究机制具有新颖性。
