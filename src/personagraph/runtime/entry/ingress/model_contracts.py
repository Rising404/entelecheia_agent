"""Entry 分类器的结构化等级提案。"""
from pydantic import BaseModel, ConfigDict
from ...turn.contracts import ProcessingLevel

class EntryClassification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    processing_level: ProcessingLevel

__all__ = ["EntryClassification"]
