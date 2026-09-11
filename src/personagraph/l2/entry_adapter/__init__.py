"""L2 与公共 Runtime Entry 之间的私有适配层。

本包负责组合 L2 executor，并把其私有结果投影为公共 Entry 可处理的封闭结果。
Turn Window、事件持久化与最终回复仍由 :mod:`personagraph.runtime.entry` 独占。
"""

__all__: tuple[str, ...] = ()
