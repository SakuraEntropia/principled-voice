# 文献核验 · M0

检索：2026-11-16；整理：2026-11-17。29 个一手来源的 URL、实际响应、时间与 SHA256
见 `evidence/sources.json`。HTTP 200 仅表示取回成功，不等于结论被独立复现。
原始需求见 `docs/DESIGN.md`。
2026 年三篇研究按取回的 arXiv 版本审读；未独立确认其同行评审录用状态。

| 研究 | 已核验结论与本项目取舍 |
|---|---|
| [CtrlSpeech](https://arxiv.org/abs/2608.08362) | DiTAR 上增加音素对齐的 pitch/loudness/duration 条件；[官方代码](https://github.com/zszheng147/ctrlspeech)已公开。代码 MIT、[权重 CC BY-NC 4.0](https://huggingface.co/zhisheng01/CtrlSpeech)。可作为控制接口参考，未下载权重用于本工具。不能把作者的解耦声明当作本机验证。 |
| [FastSpeech 2](https://arxiv.org/abs/2006.04558) | pitch、energy、duration 条件的直接先例；[作者页面](https://speechresearch.github.io/fastspeech2/)。[ming024 实现](https://github.com/ming024/FastSpeech2)是社区实现，不能冒称微软官方发布；MIT 代码不自动覆盖数据和 checkpoint。未运行。 |
| [Ctrl-P](https://arxiv.org/abs/2106.08352) | Interspeech 2021，显式时序韵律控制。此次未定位到可核验的作者官方实现及权重许可证；未据此假定有现成可用 backbone。 |
| [Within-Utterance Style Control](https://arxiv.org/html/2605.27376v1) | Parler-mini 上的风格表示插值、初始 self-KV 替换、风格 cross-conditioning 替换及 prefix+window mask。仅换 KV 不等于完整方法；历史音频 token 会保留源风格。未发现论文专用官方代码链接；此次是 tiny 模型工程试验，不是论文复现。 |
| [Sparse Emotion Circuit](https://arxiv.org/html/2610.05080v1) | 在 IndexTTS2 上定位 attention heads/MLP 的情感相关作用；activation patching 不等同于通用 KV 线性情感方向。主要实验单一音色、固定参考录音，主观检查仅一位听者；不能外推 Kororu 或中文。未发现作者专用代码链接。 |
| [IndexTTS2](https://arxiv.org/abs/2506.21619) | 自回归语音和演出研究候选。[官方仓库](https://github.com/index-tts/index-tts)当前许可证是自定义模型协议，含额外用途与下游义务；未安装/运行，不声称本机 MPS 可用。 |
| [SPARC](https://arxiv.org/abs/2406.12998) | 从音频估计构音表征并重合成，适合构音接口研究；不是开箱即用的文本配音管线。[仓库](https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding)的 GitHub 许可字段为空、根 LICENSE 请求 404；代码/权重授权未充分核实，未引入。 |
| [DDSP](https://arxiv.org/abs/2001.04643) | 可微 DSP 的先例，[代码](https://github.com/magenta/ddsp) Apache-2.0。可解释合成组件不等于完整 TTS，也不使声学反演自动唯一。未引入 TensorFlow 栈。 |
| [Neural Source–Filter](https://arxiv.org/abs/1904.12088) | 显式激励与神经滤波结合；[作者代码](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts) BSD-3-Clause。Kokoro 的 F0 注入发生在神经声码器前，区别于输出 pitch shift；不意味着能直接控制真实声带张力。 |
| [Activation Addition](https://arxiv.org/abs/2308.10248)；[patching 方法](https://arxiv.org/abs/2404.15255) | 隐状态干预与因果 patching 的方法依据。方向必须由匹配数据估计并在保留集验证，不能由变量名“anger”直接赋予语义。此次 activation pilot 仅是未校准的 prefix 均值差。 |
| [HyperNetworks](https://arxiv.org/abs/1609.09106)；[FiLM](https://arxiv.org/abs/1709.07871) | 参数生成与特征调制已有成熟先例。本项目当前用解析算子调制映射增益和分布参数，未训练 hypernetwork，不主张该概念原创。 |
| [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)；[Parler](https://github.com/huggingface/parler-tts) | 前者适合低内存、显式声学输入的真实基线；后者提供真实 AR cache。两个模型职责分开，cache 不互通。权重、代码和实际版本见 LICENSES / MODEL_COMPARISON。 |

## 数学与可证伪边界

状态演化：`z_i(t+1)=F_i(z_i(t),u_i(t); θ_i(t))`；参数调制：
`θ_i(t)=θ_i⁰+Σ_j H_ij(z_j(t))`。当前可执行例：
`g=1−inhibition; tension=tanh(g·arousal); pressure=0.7·tension`；
`μ_pitch=3·tension; σ_pitch=0.2+0.5·|tension|; μ_N=0.4·pressure`。
这些是明确标记的工程假设；无真实生理标定、无已学习情感语义。

数据、参数、约束、梯度、时间流分别标注；NumPy executor 不提供自动微分。
反馈只读上一 tick 的显式状态。反问题采用有界最小二乘，报告残差与不可行性；
这是代理空间求解器，不能代替 waveform 上的约束检查或真实生理反演。

目前实验只支持局部控制有效性的有限结论。情感、自然度、音色一致性需要授权角色
参考、多文本/多 seed、独立识别与盲听。M3 的情感效果及 M6 生产适用性尚未成立。
