"""请求内短引用的纯映射；不访问存储、不猜测身份、不改写用户正文。"""

from copy import deepcopy
import re


class ToolReferenceAdmissionError(ValueError):
    def __init__(self, field: str, path: str, available: tuple[str, ...]):
        self.field, self.path, self.available = field, path, available
        super().__init__(f"{field} must select a reference in the current execution and scope")


class ReferenceBindings:
    """由 Host 冻结的行重建双向索引；相同原生身份在同一 scope 内只绑定一次。"""

    def __init__(self, prefix: str, rows=()):
        self.prefix = prefix
        self.rows = deepcopy(list(rows))
        self.by_ref = {}
        self.by_identity = {}
        for row in self.rows:
            ref, value, scope = row["ref"], row["value"], tuple(row["scope"])
            if (not isinstance(ref, str) or re.fullmatch(rf"{prefix}[1-9][0-9]*", ref) is None
                    or not isinstance(value, str) or not value
                    or not all(isinstance(part, str) and part for part in scope)):
                raise ValueError("invalid frozen tool reference binding")
            key = (scope, value)
            if ref in self.by_ref or key in self.by_identity:
                raise ValueError("duplicate frozen tool reference binding")
            self.by_ref[ref] = (scope, value)
            self.by_identity[key] = ref

    @property
    def schema(self):
        return {"type": "string", "pattern": rf"^{self.prefix}[1-9][0-9]*$"}

    def project(self, value, scope=()):
        return self.by_identity.get((tuple(scope), value)) if isinstance(value, str) else None

    def materialize(self, row, *, native, field, scope=(), path):
        available = tuple(ref for ref, (bound_scope, _) in self.by_ref.items() if bound_scope == tuple(scope))
        if native in row:
            raise ToolReferenceAdmissionError(field, path + "/" + native, available[:16])
        if field not in row:
            return  # 必填字段与其他参数仍由原生 ToolSpec 校验。
        ref = row.pop(field)
        binding = self.by_ref.get(ref) if isinstance(ref, str) else None
        if binding is None or binding[0] != tuple(scope):
            raise ToolReferenceAdmissionError(field, path + "/" + field, available[:16])
        row[native] = binding[1]

    def project_field(self, row, *, native, field, scope=()):
        if native in row:
            value = row.pop(native)
            # 失败的历史调用可能含不存在的原生 ID；不为它分配有效引用。
            row[field] = self.project(value, scope)


def append_binding(bindings, *, prefix, value, scope=(), ordinal=None):
    """仅在构建快照时使用；输入顺序必须来自持久顺序，不能来自当前可见窗口。"""
    key = (tuple(scope), value)
    if key in bindings:
        return
    bindings[key] = {"ref": f"{prefix}{ordinal if ordinal is not None else len(bindings) + 1}",
                     "value": value, "scope": list(scope)}


def object_items(arguments, key):
    rows = arguments.get(key)
    if isinstance(rows, list):
        yield from ((i, row) for i, row in enumerate(rows) if isinstance(row, dict))


def file_scope(row):
    return (row.get("file_id"), row.get("file_version_id"))


def rename_schema_field(schema, native, field, replacement):
    """修改已选定对象的字段和组合约束，不递归搜索任意业务内容。"""
    if native in schema.get("properties", {}):
        schema["properties"].pop(native)
        schema["properties"][field] = deepcopy(replacement)
    if "required" in schema:
        schema["required"] = [field if key == native else key for key in schema["required"]]
    for key in ("oneOf", "anyOf", "allOf"):
        for child in schema.get(key, []):
            rename_schema_field(child, native, field, replacement)
    if "not" in schema:
        rename_schema_field(schema["not"], native, field, replacement)
