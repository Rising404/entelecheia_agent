"""Entry supervisor 前的并发 ingress/classifier 协调。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from personagraph.model_io.gateway import ModelGatewayError
from ..context.contracts import EntryContext
from .contracts import IngressDecision, IngressDisposition, IngressHandler
from .model_contracts import EntryClassification
from ...turn_events import EntryEventEmitter
from ...turn_deadline import TurnDeadline


EntryIngressEvaluator = Callable[[EntryContext, dict[str, Any]], IngressDecision]
EntryClassifier = Callable[[EntryContext, EntryEventEmitter, TurnDeadline], EntryClassification]
_TerminalResult = TypeVar("_TerminalResult")


@dataclass(frozen=True)
class EntryClassificationStageResult(Generic[_TerminalResult]):
    """返回给 Entry controller 的 supervisor 前协调结果。"""

    ingress: IngressDecision
    classification: EntryClassification | None = None
    terminal_result: _TerminalResult | None = None

    def __post_init__(self) -> None:
        model_ingress = (
            self.ingress.disposition is IngressDisposition.ACCEPT
            and self.ingress.handler is IngressHandler.MODEL
        )
        has_classification = self.classification is not None
        has_terminal_result = self.terminal_result is not None
        if model_ingress and has_classification == has_terminal_result:
            raise ValueError(
                "model ingress requires exactly one classification result or terminal result"
            )
        if not model_ingress and (has_classification or not has_terminal_result):
            raise ValueError("non-model ingress requires a terminal result only")


def run_entry_classification_stage(
    *,
    context: EntryContext,
    features: dict[str, Any],
    emit: EntryEventEmitter,
    deadline: TurnDeadline,
    ingress_evaluator: EntryIngressEvaluator,
    classifier: EntryClassifier,
    on_nonmodel_ingress: Callable[[IngressDecision], _TerminalResult],
    on_classifier_error: Callable[[ModelGatewayError], _TerminalResult],
) -> EntryClassificationStageResult[_TerminalResult]:
    """并行评估确定性 ingress 与有界 classifier。

    Ingress 决定模型提案是否可被消费。非模型决策会有意取消且绝不观察 classifier
    Future，使无关 provider 失败无法将安全的有类型回复变成 incomplete turn。
    终态回调由 Entry 提供，并有意在 executor 等待不可取消 classifier 前运行，以
    保留既定持久化事件顺序。本模块自身不实现 Store、Window 或最终化。

    两份 copy_context 分别继承 Session/trajectory 归属，不能在两线程并发进入同一个
    Context。Future.cancel 只能取消尚未开始的任务：非模型 ingress 可能仍伴随一次
    已开始的 classifier 调用，但该模型提案不会参与路由，不能用最终 lane 推断调用数。
    """

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="runtime-entry") as executor:
        ingress_context = copy_context()
        classifier_context = copy_context()
        ingress_future = executor.submit(
            ingress_context.run,
            ingress_evaluator,
            context,
            features,
        )
        classifier_future = executor.submit(
            classifier_context.run,
            classifier,
            context,
            emit,
            deadline,
        )
        ingress = ingress_future.result()
        if (
            ingress.disposition is not IngressDisposition.ACCEPT
            or ingress.handler is not IngressHandler.MODEL
        ):
            classifier_future.cancel()
            return EntryClassificationStageResult(
                ingress=ingress,
                terminal_result=on_nonmodel_ingress(ingress),
            )
        try:
            classification = classifier_future.result()
        except ModelGatewayError as exc:
            return EntryClassificationStageResult(
                ingress=ingress,
                terminal_result=on_classifier_error(exc),
            )
        return EntryClassificationStageResult(
            ingress=ingress,
            classification=classification,
        )
