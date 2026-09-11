"""集中规定模型所写自由文本应使用何种语言。

Runtime 中的每条规划和验证提示词都使用中文，但此前都没有说明模型应使用什么语言回答。
因此，自由文本字段——节点标题、目标、验收标准以及用户门提出的问题——一直处于漂移状态，
实际也确实如此：同一提示词和同一模型某天生成中文节点，隔天却生成英文节点。

规则并非“始终使用中文”。提及 ``US-EAST`` 或 ``mounted_document_read`` 的节点目标必须
逐字保留这些词元——翻译标识符会破坏读取它的契约，翻译从来源文档引用的术语则会破坏引用。
因此，正确输出通常是混排的：句子跟随用户的语言，其中的名称保持原样。
"""

OUTPUT_LANGUAGE_CLAUSE = """
输出语言：自由文本字段（title、objective、criterion、explanation、question、
summary、reason、value 等一切给人读的句子）一律使用用户本人所用的语言，不要改用
其他语言。

但以下内容必须逐字保留原样，绝不翻译、绝不音译：
- 字段名、枚举值、schema_version、node_key、各类 alias、capability_profile_id、
  output_contract 以及任何 ID、hash、路径、工具名；
- 来源材料中出现的专有名词、标识符、指标名、字段名、代码片段与文件名
  （例如 US-EAST、throughput、README.md、q4_sales.xlsx）；
- 直接引用来源时的摘录原文。

因此正确的输出通常是混排的：句子本身是用户的语言，句中的标识符和专名保持来源
形态。不要为了语言统一去翻译标识符，也不要因为句中含有标识符就把整句改写成另一
种语言。
""".strip()


# 规划链上还有一个更精确的判据可用：Host 会把用户原话作为一张 authority card 冻结
# 进来，语言以那张卡为准，而不是以周围任何被引用的材料为准。
PLANNING_OUTPUT_LANGUAGE_CLAUSE = (
    OUTPUT_LANGUAGE_CLAUSE
    + "\n\n用户所用语言以 authority 中 source_kind=user_instruction 的原文为准；"
    "被引用的文档或工具结果是什么语言，都不改变你输出句子的语言。"
)
