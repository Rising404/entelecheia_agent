"""文本后缀的唯一声明，供上传识别、入库路由和直接读取工具共用。

这里只声明文本格式身份，不承诺 JSON 等结构化语法合法。实际可读性仍由 reader
检查编码、大小和内容；类型检测也必须先验证字节，不能仅凭后缀放行二进制内容。
"""

from types import MappingProxyType


TEXT_MEDIA_TYPES = MappingProxyType({
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    # JSONL/NDJSON 复用纯文本解析与切块，不引入逐行 JSON 校验或独立入库流程。
    ".jsonl": "application/x-ndjson",
    ".ndjson": "application/x-ndjson",
    ".log": "text/plain",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/typescript",
    ".html": "text/html",
    ".css": "text/css",
    ".sql": "application/sql",
    ".sh": "application/x-sh",
    ".toml": "application/toml",
    ".xml": "application/xml",
    ".ini": "text/plain",
    ".rs": "text/x-rust",
    ".go": "text/x-go",
    ".java": "text/x-java",
    ".c": "text/x-c",
    ".h": "text/x-c",
    ".cpp": "text/x-c++",
    ".rb": "text/x-ruby",
})

TEXT_SUFFIXES = frozenset(TEXT_MEDIA_TYPES)
