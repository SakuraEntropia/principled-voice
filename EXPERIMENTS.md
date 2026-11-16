# 实际实验记录

测试日期：2026-11-16 至 17；Apple Silicon，24 GiB。未训练角色模型。

## M1 原生控制

| 条件 | 估计中位 F0 / Hz | 时长 / s | RMS / dBFS |
|---|---:|---:|---:|
| baseline | 199.69 | 4.675 | -26.30 |
| pitch_up | 236.88 | 4.675 | -26.24 |
| pitch_down | 170.38 | 4.675 | -26.76 |
| faster | 198.19 | 4.300 | -26.39 |
| slower | 196.09 | 5.950 | -26.32 |
| n_up | 199.81 | 4.675 | -26.18 |
| n_down | 199.58 | 4.675 | -26.42 |
| aroused | 222.69 | 4.675 | -26.20 |
| inhibited | 199.69 | 4.675 | -26.30 |
| absolute_f0 | 199.60 | 4.675 | -26.30 |

固定 seed 重复样本完全一致：True；与原版推理最大样本差：0.0。
原生时长 token 增加验证为每帧 600 samples；无 waveform time stretch/pitch shift。
N±0.3 只带来很小响度变化，不能宣传为已校准 dynamics 控制。
高唤醒代理与抑制测试验证的是解析耦合，不是情感识别或生理标定。

严格 200 Hz / 0.01 Hz 容差：输出中位数约 199.604 Hz，正确报告 constraint_violation。
F0 跟踪采用独立自相关估计器，条件对齐为近似归一化时间；不当作真实标注。
有限差分耦合矩阵与全部输出见 `renders/m1_controls_v2/results.json`。

## KV pilot

Parler tiny，316,555,314 参数；CPU、单文本、seed=42、native sampling。
11 组对照；no-op 样本差 0.0；总时长 30.59s；进程峰值 RSS 1.95 GiB。

| 条件 | 估计中位 F0 / Hz | RMS / dBFS |
|---|---:|---:|
| native_target | 250.72 | -13.75 |
| baseline | 243.23 | -14.86 |
| noop | 243.23 | -14.86 |
| k_only | 241.85 | -14.82 |
| v_only | 226.41 | -14.55 |
| joint_kv | 253.17 | -12.92 |
| cache_swap | 238.04 | -14.45 |
| low_rank | 234.45 | -14.65 |
| layer_head | 243.23 | -14.86 |
| swap_window | 228.47 | -16.11 |
| activation | 243.23 | -14.86 |

这些数据不能证明沿 donor 方向单调控制情感。cross-attention 保持 source，故不是论文完整方法复现。
layer/head 和 activation 条件在本例无明显效果；须保留负结果。
greedy 初轮近静音；第二轮虽有信号但出现约 86 Hz 周期读数，不能据此评价情感。
结果和 WAV 分别保留于 `renders/kv_pilot`、`kv_pilot_v2`、`kv_sampled`。

## 独立内容检查

本地 faster-whisper tiny.en，CPU int8。只作机器交叉检查，不等于人类听审。
4 条 Kokoro（baseline/pitch_up/faster/aroused）机器 WER 均为 0。
采样 KV 的 baseline、K-only、V-only、cache_swap 机器 WER 为 0；native_target 为 0.6。
因此不同风格存在内容失败，M3 尚未通过内容/情感/身份联合验收。
逐条 transcript 与 reference：`evidence/content_check.json`。

## 兼容性与统计限制

CPU M1 baseline RTF 0.627；MPS 短句首次 RTF 6.627。
两次文本和预热条件不同，不作严格性能基准；MPS 仅证明可运行，默认 CPU。
RSS 为进程 lifetime high-water mark，不等于总统一内存或 GPU 分配。
暂无多文本置信区间、speaker similarity、emotion accuracy、MOS、viseme 误差或用户听审。
较长预算可能截断；没有据“出声”和 ASR 一致宣称情感成功。
