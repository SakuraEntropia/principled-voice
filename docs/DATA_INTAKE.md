# 数据接入与防泄漏计划（等待语料）

状态：仓库已在本机复验；`data/` 目前为空，未收到任何授权录音。本文只定义接入契约，
不描述、不推断、不填充 Kororu 的性别、年龄、语言、音色、性格或情感设定。
数据到达前不得开始训练；不得把语料上传、提交进 git 或用于对外发布。

## 原则

- 只在 `data/` 本地处理。`data/` 已加入 `.gitignore`；脚本默认只读元数据，不改音频。
- 听到/测到 ≠ 有权使用。**许可证**（录音/作品版权）与**录音授权**（表演者/声音人格/
  肖像同意）分别核验；代码、权重、语料的许可证不能互相外推。
- 内容、语言、说话人、情感标签由提供方声明或人工核验后写入；工具不从声音推断。
- 口头同意、聊天截图、转述都不能标记为 `cleared`。ASR 草稿不是转写真值。
- 现有 21 项测试与 Kokoro 临时英文音色只证明工程链可运行，不证明数据适配角色。

## 目录

```text
data/                    # 本地，git 忽略
  inbox/                 # 原始交付，只读，不重命名/不转码
  audio/                 # 可选的规范副本（仍只本地）
  rights/                # 已签署许可/同意书扫描件与哈希
  corpus.json            # 语料级元数据 + 许可/授权登记
  manifest.jsonl         # 每条录音一行（schema: schemas/dataset_manifest.schema.json）
  derived/               # inventory.jsonl、splits.json、检查报告（可重建）
```

## 接入流程（数据到达后按序执行）

1. 原样放入 `data/inbox/`，记录交付日期与交付人；先备份，不编辑原文件。
2. 运行只读盘点（哈希、时长、采样率、声道；不猜说话人/语言）：
   `uv run --locked python scripts/check_dataset.py inventory`
3. 人工填写 `data/corpus.json` 与 `data/manifest.jsonl`；每个 clip 必须能追溯到
   许可记录与同意记录，签字原件放 `data/rights/` 并记 SHA256。
4. 结构与权利校验：
   `uv run --locked python scripts/check_dataset.py validate --check-files`
5. 人工复核：核对签字人身份/权限、用途范围、有效期、未成年人/撤回条款、地域；
   未覆盖 `model_training` 或 `synthesis` 的一律保持 `pending`，不得进入训练集。
6. 划分（按说话人/场次/台本连通组，整组进入同一 split）：
   `uv run --locked python scripts/check_dataset.py plan --write`
7. 防泄漏检查：`... leakage`；通过后再冻结 `derived/splits.json` 与 manifest 哈希。
8. 质量核验与适配评估（见下）单独报告；先测后定阈值，不预设“合格线”。

## 字段摘要

`corpus.json`

| 记录 | 关键字段 | 规则 |
|---|---|---|
| license | `license_id`、名称、SPDX（自定义则 null）、授权方、签署日、`scope`、地域、到期、文件路径/哈希、再分发与模型发布许可 | `scope` 至少覆盖 `model_training`、`synthesis`；目标音色建议显式覆盖 `voice_cloning` |
| authorization | `consent_id`、签署人角色（本人/监护人/权利人）、签署日、`scope`、有效期、撤回条款、未成年人、文件路径/哈希 | 被许可方代签不等于说话人同意；`licensee` 签署会给警告 |

`manifest.jsonl`：`clip_id`、相对 `file`、`sha256`、`duration_s`、`sample_rate`、
`channels`、`speaker_id`（化名）、`session_id`、`script_id`、`language`（BCP-47）、
`transcript` + `transcript_status`、`speaker_role`、`license_id`、`authorization_id`、
`usage_status`（`cleared/pending/rejected/unknown`）、`permitted_uses`、`quality`、`split`。
`cleared` 必须同时解析到许可与授权，且范围覆盖训练与合成；到期自动报错。

## 划分与防泄漏

- 分组用并查集合并 `speaker_id`、`session_id`、`script_id`；同一组绝不跨 split，
  防止同一人、同一录音场次或同一台本泄漏。
- 默认 80/10/10，按总时长贪心，seed 固定；组数 <3 时无法构成三向划分，直接报错。
  比例是本项目自建语料的约定，不套用公开数据集的官方划分。
- 检查项：组跨 split = 错误；跨 split 重复 SHA256 = 错误；跨 split 完全重复转写 =
  错误；跨 split 转写 5-gram Jaccard ≥ 0.8 = 警告；空 split = 错误。
- `test` 只用于最终报告；`dev` 用于选择；不得用 `test` 调参或选 checkpoint。
- 冻结后新增数据只进 `train`；若必须重划，重跑全部分组、生成新版本并保留旧
  `splits.json` 与 manifest 哈希，不得原地覆盖后继续报告旧结果。

## 质量核验（数据到达后测，不预判）

客观：时长/采样率/位深/声道、削波比例、直流、静音比、SNR 估计、响度分布、F0 覆盖、
文本-音频一致性抽样（人工复核，不以 ASR 一致代替）。
听审：噪声/混响/串音、发音与口音、情绪与风格覆盖（以提供方标注为准）。
适配：目标说话人条数与总时长、语言与当前 `en-us` 前端的差距、录音设备/场次一致性。
所有阈值在看到真实分布后写入报告并说明依据；不把“能出声/ASR 相似”当作音色或
情感验证，也不报告未做的 MOS、说话人相似度或情感准确率。

## 当前 blocker

- 无语料、无授权参考录音、无角色语言/音色/发音资料：训练与角色验收均不能开始。
- 前端仅 `en-us`；Kokoro 为 24 kHz 临时音色，中文台词的适配未验证。
- N 是未校准声学条件，不是响度或声门压力；hard lock 只检查输出中位 F0，
  RMS hard lock 被显式拒绝。
- Parler KV 仅为工程 pilot；`native_target` 曾出现 ASR 内容错误，不能宣称情感控制成功。
