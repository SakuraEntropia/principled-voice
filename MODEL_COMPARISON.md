# 模型选择

| 候选 | 显式控制 | AR KV | 本机验证 | 采用状态 |
|---|---|---|---|---|
| Kokoro v1.0 / 82M | 原生 F0/N；token 对齐时长 | 无 | CPU、MPS 实际输出 WAV；CPU 默认 | M1 临时英文音色 |
| Parler tiny / 实测 316,555,314 参数 | 自然语言风格；需额外实验 | 有 | CPU 11 组 cache 对照；未验证 MPS | 独立研究 backend |
| CtrlSpeech control-150m/600m | pitch/loudness/duration 条件 | 有 AR 部分 | 未安装 | 权重 NC，暂不接入生产 |
| IndexTTS2 | 情感/时长候选 | 有 | 未安装 | 自定义许可及依赖待评估 |
| FastSpeech 2 | 显式韵律 | 无 | 未安装 | 训练/适配成本暂不优先 |
| SPARC | 构音重合成 | 非本轮目标 | 未安装 | 许可、文本前端待解决 |

来源与许可证分层见 RESEARCH / LICENSES。未测试的候选不能据参数量估计为“24GB 保证可跑”。

固定权重：Kokoro `f3ff3571791e39611d31c381e3a41a3af07b4987`；
Parler `fe1bd939bb05464d39a76e20cf8a35d4e6885571`。
Parler 代码 `d108732cd57788ec86bc857d99a6cabd66663d68`。
实际文件 SHA256 存于每条合成 manifest 和 KV results。
