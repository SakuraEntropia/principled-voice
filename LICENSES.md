# 许可审计

本项目用于 in-house 研发。源码公开可查看；新代码尚未指定开源许可证。
以下是已取回源文件的工程记录，不把“开源代码”解释成全部权重/数据无条件可用。

| 组件 | 代码 / 权重 / 数据 | 本轮处置 |
|---|---|---|
| [Kokoro](https://github.com/hexgrad/kokoro) | 代码 Apache-2.0；[固定版本模型卡](https://huggingface.co/hexgrad/Kokoro-82M/blob/f3ff3571791e39611d31c381e3a41a3af07b4987/README.md)标 Apache-2.0 权重。数据来源是作者声明，未独立逐录音审计 | 临时 af_heart 公开预置；不克隆指定真人，不宣称 Kororu 音色。许可证副本 `third_party/kokoro_LICENSE` |
| [Parler-TTS](https://github.com/huggingface/parler-tts) | Apache-2.0 代码；[tiny 模型卡](https://huggingface.co/parler-tts/parler-tts-tiny-v1)标 Apache-2.0。训练数据仍有独立来源/署名信息 | 仅本地研究；副本 `third_party/parler_LICENSE` |
| [Whisper tiny.en 转换权重](https://huggingface.co/Systran/faster-whisper-tiny.en) | 固定模型卡标 MIT；仅用于独立本地内容检查 | revision 与文件 SHA256 见 `evidence/content_check.json` |
| [CtrlSpeech](https://github.com/zszheng147/ctrlspeech/blob/main/LICENSE) | MIT 代码；[公开权重](https://huggingface.co/zhisheng01/CtrlSpeech) CC BY-NC 4.0 | 未下载模型，未接入商业生产 |
| [IndexTTS2](https://github.com/index-tts/index-tts/blob/main/LICENSE) | 当前自定义 bilibili Model Use License；含下游和用途条件 | 未安装；不沿用旧版 Apache 假设 |
| [SPARC](https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding) | 根许可证未找到，GitHub license 字段为空；权重条款未充分核验 | 不引入 |
| [DDSP](https://github.com/magenta/ddsp) / [NSF 作者代码](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts) | 分别 Apache-2.0 / BSD-3-Clause 代码；checkpoint/data 另核查 | 仅研究参考，未复制这些源码 |
| Misaki / eSpeak-ng / phonemizer | Python 包与内含本地库条款分别适用；eSpeak-ng/phonemizer 有 GPL 条款 | 本地调用；未来分发应用时须完整处理相应源码与通知义务 |

Kokoro 模型卡还列明训练集中的 [Koniwa](https://github.com/koniwa/koniwa)（CC BY 3.0）与
[SIWIS](https://datashare.ed.ac.uk/handle/10283/2353)（CC BY 4.0）。这里保留来源，
不把模型许可扩张解释为全部原始数据或输出的无条件权利担保。

论文文本许可证与代码/权重许可证不同。论文可阅读并不自动授权复用作者代码或录音。
完整来源哈希见 `evidence/sources.json`；安装包元数据见 `evidence/dependency_licenses.json`。
