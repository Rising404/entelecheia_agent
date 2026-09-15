# 实验配置 / Experiment configs

每次运行绑定一份完整 YAML，`run` 与 `score` 使用同一份配置。续跑时会核对配置身份，改动后的配置会被拒绝。
下表列出主要入口；设置 `PERSONAGRAPH_BENCH_EVAL_DIR` 后，从仓库根执行 `.venv/bin/python -m evals.docbench.reproduce_or_run_script list` 可查看完整列表。

正式选题清单位于相邻的 [selections/](../selections/)；真实 QA 从 `dataset.data_root` 读取。
参考答案只供裁判使用，不挂载到 Agent 的 Project。

| 配置 | 用途 |
| --- | --- |
| `l1_bge_m3_live_1.yaml` | 单题 CPU 工程冒烟 |
| `l1_balanced_125.yaml` | 原有 125 题，语义门开启，MPS |
| `l1_balanced_125_gate_off.yaml` | 同题，仅关闭语义门，模型取安装级活跃配置 |
| `l1_balanced_125_gate_off_highland_235b.yaml` | 同题关门，显式使用 Highland 235B 视觉模型 |

## 指定视觉模型 / VLM

`l1_balanced_125_gate_off_highland_235b.yaml` 的视觉部分如下，其余字段与 `l1_balanced_125_gate_off.yaml` 一致：

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

运行前在自己的 shell 或私有凭据环境设置 `HIGHLAND_API_KEY`；YAML 的 `api_key_env` 指定密钥所在的环境变量。
`api_key_env` 只是变量名，不会自动读取 GUI 的 Highland 密钥；显式端点不切换 GUI 活跃模型。
主模型读取安装配置，裁判使用主模型。

已归档的 [123 题实验](../previous_results/gate_off_highland235b_123/README.md) 使用了该 235B 视觉模型。
与前轮相比，该实验同时关闭语义门并更换 VLM，结果反映两项配置共同变化后的表现。单独比较语义门或视觉模型的影响，需要保持另一项及其他运行条件相同。

## 可调整字段 / Editable fields

| 字段 | 含义与边界 |
| --- | --- |
| `providers.main` / `providers.vision` | 安装级来源或显式端点；显式端点只写密钥环境变量名 |
| `dataset.data_root` | QA/PDF 根；默认 `bench://source/data`，可指定符合清单结构的其他目录 |
| `dataset.selection` | 冻结选题文件；路径与内容均参与身份核验 |
| `scoring.prompt` / `prompt_sha256` | 裁判提示词文件与固定哈希 |
| `scoring.judge` | `source: main` 或显式 DeepSeek-compatible 端点 |
| `run.output_root` | 使用 `bench://runs`；活跃运行目录限定为 `<外部根>/docbench/runs/<run-id>` |
| `run.runtime_features` | 运行特性文件，语义门开关在其中 |
| `retrieval.device` | 如 `cpu`、`mps`、`cuda`；设备设置参与实验身份 |

常规相对路径以**仓库根**而非 YAML 所在目录解析；`bench://` 以外部 DocBench 根解析。
自定义 YAML 建议放在自己的仓库外目录，完整示例与操作顺序见[操作手册](../docs/formal_l1_eval_runbook.md)。
