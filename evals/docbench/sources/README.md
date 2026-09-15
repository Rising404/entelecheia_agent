# 数据准备 / DocBench sources

[上游清单](../upstream.manifest.json) 记录数据来源、固定 revision、许可和哈希。下载材料默认放在仓库外的
`<PERSONAGRAPH_BENCH_EVAL_DIR>/docbench/source/`：`data/` 保存 Drive QA/PDF，映射 JSON 与固定
`evaluation_prompt.txt` 位于其同级。仓库提供下载命令和选题清单，PDF、QA 与裁判提示词单独准备。
配置通过 `bench://source/...` 解析这些路径，与产品 `STATE_DIR` 分开。

`source/` 是默认布局。已有数据时可在实验 YAML 中配置
`dataset.data_root`，裁判提示词用 `scoring.prompt` 指定；选题清单仍会核验内容和哈希。
自定义外部根通过 `PERSONAGRAPH_BENCH_EVAL_DIR` 设置，使用仓库外的绝对路径。

从仓库根执行，先将示例替换为自己的外部目录。以下下载命令需要联网：

```bash
export PERSONAGRAPH_BENCH_EVAL_DIR="/absolute/path/to/bench_eval"
.venv/bin/python -m evals.docbench.reproduce_or_run_script prepare-data --qa-catalog-only
.venv/bin/python -m evals.docbench.reproduce_or_run_script prepare-data --balanced-pdfs-only
```

下载器也支持 `--data-root` 和 `--mapping` 指定数据、映射位置；选题生成器支持 `--output`
指定新清单。只有完全显式提供所需路径时，才不依赖默认外部根的解析。
这些下载命令不准备裁判 prompt、所有回归附件或检索模型；也没有 `prepare-data --config`。

运行只把所选 PDF 挂载给 Agent，不把 QA、参考答案或整个 source 根绑定为 Project。
历史 `previous_results/` 不参与下载、选题和文件入库。详见[操作手册](../docs/formal_l1_eval_runbook.md)。
