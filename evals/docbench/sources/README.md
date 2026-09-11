# DocBench sources

记录固定 revision、许可、下载方式和 source manifest。上游材料统一放在仓库外的
`<PERSONAGRAPH_BENCH_EVAL_DIR>/docbench/source/`：`data/` 保存 Drive QA/PDF，映射 JSON 与固定
`evaluation_prompt.txt` 位于其同级。这些材料都不得提交；现行配置通过 `bench://source/...`
解析，不再借用产品 `STATE_DIR`。
