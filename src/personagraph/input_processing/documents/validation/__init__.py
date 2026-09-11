"""文档处理结果的纯机械覆盖判定。"""

from .coverage import (
    PageAuthorityCoverageGap,
    PaperPageAuthorityEligibility,
    evaluate_page_authority_eligibility,
    evaluate_paper_page_authority_eligibility,
)

__all__ = [
    "PageAuthorityCoverageGap",
    "PaperPageAuthorityEligibility",
    "evaluate_page_authority_eligibility",
    "evaluate_paper_page_authority_eligibility",
]
