"""文档读取、覆盖分类与切块的纯准备入口。"""

from .contracts import DocumentPrepareFailure, PreparedDocumentIngest
from .service import prepare_document_path

__all__ = [
    "DocumentPrepareFailure",
    "PreparedDocumentIngest",
    "prepare_document_path",
]
