# 实验配置 / Experiment configs

每次运行绑定一份完整 YAML。`run` 与 `score` 使用同一份配置；不要改写已运行配置去续跑。
下表仅列主要入口，完整列表用 `python -m evals.docbench.reproduce_or_run_script list` 查看。

正式选题清单位于相邻的 [selections/](../selections/)；真实 QA 从 `dataset.data_root` 读取。
参考答案只供裁判使用，不挂载到 Agent 的 Project。

| 配置 | 用途 |
| --- | --- |
| `l1_bge_m3_live_1.yaml` | 单题 CPU 工程冒烟 |
| `l1_balanced_125.yaml` | 原有 125 题，语义门开启，MPS |
| `l1_balanced_125_gate_off.yaml` | 同题，仅关闭语义门，模型取安装级活跃配置 |
| `l1_balanced_125_gate_off_highland_235b.yaml` | 同题关门，显式使用 Highland 235B 视觉模型 |

## 新增视觉模型 / VLM

新配置的视觉部分如下，其余字段与 `l1_balanced_125_gate_off.yaml` 一致：

```yaml
providers:
  main:
    source: installation
  vision:
    provider: highland
    base_url: https://www.highland-api.top
    model: qwen3-vl-235b-a22b-instruct
    api_key_env: HIGHLAND_API_KEY
```

运行前在自己的 shell 或私有凭据环境设置 `HIGHLAND_API_KEY`，不能把真实密钥写入 YAML。
`api_key_env` 只是变量名，不会自动读取 GUI 的 Highland 密钥；显式端点不切换 GUI 活跃模型。
主模型仍读取安装配置，裁判仍使用主模型。旧配置、旧评分和运行快照不改写。

已归档的 [123 题实验](../previous_results/gate_off_highland235b_123/README.md) 使用了该 235B 视觉模型。
与前轮相比，该实验同时关闭语义门并更换 VLM，不能将结果变化单独归因于语义门或视觉模型。

## 可调整字段 / Editable fields

| 字段 | 含义与边界 |
| --- | --- |
| `providers.main` / `providers.vision` | 安装级来源或显式端点；显式端点只写密钥环境变量名 |
| `dataset.data_root` | QA/PDF 根；默认 `bench://source/data`，可指定符合清单结构的其他目录 |
| `dataset.selection` | 冻结选题文件；路径与内容均参与身份核验 |
| `scoring.prompt` / `prompt_sha256` | 裁判提示词文件与固定哈希 |
| `scoring.judge` | `source: main` 或显式 DeepSeek-compatible 端点 |
| `run.output_root` | 当前支持外部根的 `bench://runs`；不能只靠改成源码内路径突破状态隔离 |
| `run.runtime_features` | 运行特性文件，语义门开关在其中 |
| `retrieval.device` | 如 `cpu`、`mps`、`cuda`；换设备要记录新实验身份 |

常规相对路径以**仓库根**而非 YAML 所在目录解析；`bench://` 以外部 DocBench 根解析。
自定义 YAML 建议放在自己的仓库外目录，完整示例与操作顺序见[操作手册](../docs/formal_l1_eval_runbook.md)。
