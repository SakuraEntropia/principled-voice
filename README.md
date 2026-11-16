# Principled Voice

《来自星星的弦乐》Kororu in-house 配音工具，当前交付 **M0 + 可运行 M1**。
角色正式音色未建立；所有示例均标为 **Temporary Voice**，英文技术测试文本不属于剧本。

## 已有功能

- 真实神经语音：冻结 Kokoro，24kHz WAV；CPU、Apple Metal 均已实际运行。
- 声码器前 F0、连续 pitch 曲线、速度、token 时长和实验性 N 条件。
- 类型化依赖图；抑制量调制映射算子，代理状态调制韵律分布；显式延迟反馈。
- 保存自动生成值和人工编辑值；有界代理反演、锁冲突/输出约束诊断。
- scene/shot/take、帧率、seed、模型文件哈希、对齐、分析、版本清单。
- 独立 Parler KV 研究：K/V/joint/swap/低秩/层和头/window/activation 对照。

**尚无图形编辑器，也未验证正式角色情感、音色保持或动画生产适用性。**
N 是学习到的声学条件，不是已标定响度；输出 gain 明确属于后期增益。

## 本机直接运行

在安装环境与下载模型后，双击 `run.command`，或在本目录执行：

```bash
.venv/bin/pv render examples/baseline.json --out renders/my_take
```

输出目录必须不存在，以免覆盖旧 take。音频在 `voice.wav`；同目录有工程、参数原值/
编辑值、对齐、独立测量及 manifest。默认 CPU；Metal 使用 `--device mps`。

## 新环境安装

需要 `uv`、Python 3.12、Apple Silicon。首次安装/模型下载需要网络；合成在本机进行。

```bash
uv sync --locked --extra dev
uv run --locked python scripts/download_models.py
uv run --locked pv doctor
uv run --locked pv validate examples/pitch_curve.json
uv run --locked pv render examples/pitch_curve.json --out renders/pitch_test
```

模型下载固定 revision，未启用远程自定义代码。Kokoro 两个主要权重约 330MB；
Parler 另需约 1.3GB。每条合成限制 30 秒；更长对白须显式分段。

## 实验与测试

```bash
PV_MODEL_DIR="$PWD/models/kokoro" uv run --locked --extra dev pytest -q
uv run --locked python scripts/run_m1.py --out renders/control_rerun
uv run --locked pv inverse --pitch-st 1 --n 0.1
uv run --locked pv inverse --pitch-st 10 --n 5
uv sync --locked --extra dev --extra kv
uv run --locked python scripts/download_models.py --with-kv
uv run --locked --extra kv pv kv-experiment --out renders/kv_rerun --max-tokens 350 --sampling
```

KV 使用单文本/单 seed pilot；零强度干预必须逐样本一致。工程检查通过不代表情感
实验成立。`EXPERIMENTS.md` 与 `FAILURES.md` 记录负结果和评估边界。

独立 ASR 检查需 `uv sync --locked --extra evaluation`，再运行
`uv run --locked python scripts/download_models.py --with-asr`，并查看
`scripts/check_content.py --help`；识别模型不是听审或音色/情感认证。
硬锁不满足时 CLI 返回退出码 2，诊断 WAV 仍保存，不能当作批准的 production take。

## 文档与目录

- `RESEARCH.md`、`PRIOR_ART.md`、`MODEL_COMPARISON.md`、`LICENSES.md`：一手研究及选择依据。
- `ARCHITECTURE.md`、`docs/DESIGN.md`：接口、数学和实际实现边界。
- `EXPERIMENTS.md`、`FAILURES.md`、`ROADMAP.md`：结果、失败与未完成项。
- `src/principled_voice/`：可安装代码；`tests/`：单元和真实模型集成测试。
- `assets/characters/kororu/`：身份空白、临时 voicebank 与待提供资料。
- `evidence/`：检索清单、环境版本、验证摘要；`renders/`：实际实验输出。

运行入口为 CLI；Qt DCC 编辑器属于 M5。未修改动画素材仓库，未包含私有资料。
新代码暂不授予公共许可证；第三方组件继续遵循各自条款。
