"""Long mixed-format corpus, gold evidence and four independently run questions."""

from .cases import Case
from .fixture_documents import TableImage


TOPICS = (
    "Earlier cargo schedules and withdrawn change notices",
    "East depot stock accounting and replenishment policy",
    "West station cargo operations and Orion D24",
    "Helios laboratory day registration and participant counts",
    "Helios consumable distribution and event supplies",
    "Campus parking, optional excursions and unrelated fees",
    "Orion D42 east station controlled dispatch circulars",
    "Equipment inspection, warranties and unsigned proposals",
    "East depot packaging units and order rounding",
    "Historical depot snapshots and former stock policy",
)


# These clauses are the ground truth, placed among ordinary operational records.
RECORDS = {
    (1, 4): "WITHDRAWN NOTICE O42-01, issued 2026-09-10. Orion D42 EAST cargo was provisionally due on 2026-09-18 at 16:10, Bay A-03, code OLD-410. This notice is withdrawn and replaced by controlled circular O42-E9. Do not use this withdrawn routing for dispatch.",
    (2, 4): "EAST DEPOT RELEASED-STOCK POLICY, effective 2026-09-17. Target usable stock is Cedar 90 units and Birch 65 units. For each SKU, count only green RELEASED lots in the final inventory image; subtract the RESERVED units of each counted lot. Orange QUARANTINE and red BLOCKED lots contribute zero usable units. The final signed count is D15_inventory.png dated 2026-09-17. Earlier counts do not apply. Shortfall equals max(target minus usable units, zero). Order whole packages using the packaging register; round package counts upward.",
    (3, 5): "ORION D24 WEST STATION acceptance, 2026-09-18: receiving closes at 17:25, Bay B-12, code WEST-625. This record concerns D24 at the WEST station, not D42 at the EAST station. A common project-family name does not transfer authority across station identifiers.",
    (4, 5): "HELIOS LABORATORY DAY, 2026-10-03, FINAL CONFIRMED ROSTER. Room A lists 18 participants, Room B lists 27, and Room C lists 16. Exactly six people are listed in both A and B; they are counted once in each list. There is no other overlap, and nobody is in three rooms. Three instructors are not included in these participant counts. Eight standby applicants are NOT confirmed and are excluded. The three rooms belong to one event, not three separate events.",
    (5, 3): "HELIOS MATERIALS RULE, effective for 2026-10-03. Each distinct confirmed participant receives two blue cards and one red card. Reserve nine additional blue cards and five additional red cards for the event as a whole. Instructors and unconfirmed standby applicants receive no participant cards. Reserves apply once to the entire event, not separately per room. The rule states quantities only; it does not establish any registration price.",
    (6, 4): "CAMPUS ACCESS PRICE SHEET. Public parking is 6 credits per vehicle-hour. The separately operated Lumen excursion, held 2026-10-04, charges 48 credits per visitor. Its admission list and pricing authority do not cover the Helios laboratory day on 2026-10-03. Shared use of the same campus does not make these fees laboratory registration fees.",
    (7, 6): "CONTROLLED CIRCULAR O42-E9. Approved 2026-09-16; effective for Orion D42 EAST cold-chain sample dispatch on 2026-09-18. The latest receiving cutoff is 18:35 local time, delivery point Bay C-17, handover code OR42-E9. This signed circular replaces the withdrawn O42-01 schedule and overrides unsigned planning notes. Do not confuse the D42 EAST shipment with D24 WEST. No late-delivery penalty amount is specified in this circular.",
    (8, 5): "UNSIGNED WORKING ESTIMATE H-COST-7. The planning group explored a Helios participant price of 120 credits, but this number was not approved and was explicitly withdrawn before registration opened. It is an internal planning assumption, not a quoted or payable fee. No registration charge was established by this estimate. The final registration notice is in D12_notices.txt.",
    (9, 3): "EAST DEPOT PACKAGING REGISTER, approved 2026-09-17. Cedar is supplied in packages of 6 units. Birch is supplied in packages of 10 units. Orders must use whole packages, rounding upward to cover the shortfall. These package sizes apply to released-stock replenishment after the 2026-09-17 final count. Packaging rules do not change the definition of usable inventory.",
    (10, 4): "ARCHIVED DEPOT RULE, expired 2026-09-16. Earlier targets were Cedar 80 units and Birch 50 units, and old package sizes were Cedar 5 and Birch 8. The historical image D16_inventory.png records 2026-09-10 stock, including Cedar 82 and Birch 61. Neither this rule nor that image is valid for the approved 2026-09-17 replenishment order.",
}


def routine(doc: int, page: int, entry: int) -> str:
    site = ("Aster", "Maple", "River", "Hill", "Harbor")[(doc + page + entry) % 5]
    item = ("sensor cases", "empty crates", "display stands", "training binders", "cable reels")[(doc * 2 + entry) % 5]
    count = 70 + doc * 11 + page * 7 + entry * 3
    return (
        f"Routine record R-{doc:02d}-{page:02d}-{entry:02d}: the {site} support team inspected {count} {item} "
        f"during cycle {page + entry}. The clerk recorded container seals, storage labels, shelf clearance and "
        "the order of the inspection forms. Two observers signed the handling checklist after comparing the "
        "physical labels with the local rack inventory. These routine quantities describe support supplies "
        "only; they are not event participant counts, product availability or dispatch instructions. "
        "A missing seal requires a separate maintenance ticket. Completed forms remain in their original "
        "record series, and local room identifiers must not be substituted for receiving locations in a controlled circular."
    )


def pdf_document(doc: int) -> str:
    pages = []
    for page in range(1, 7):
        paragraphs = [f"Record series: {TOPICS[doc - 1]}. Section {page} of 6."]
        paragraphs.extend(routine(doc, page, entry) for entry in range(1, 4))
        paragraphs.insert(2, RECORDS.get((doc, page), routine(doc, page, 4)))
        pages.append(f"OPERATIONS RECORD {doc:02d} | SECTION {page}\n" + "\n".join(paragraphs))
    return "\f".join(pages)


def text_document(doc: int, heading: str, special: str) -> str:
    sections = [f"# {heading}"]
    for section in range(1, 19):
        sections.extend((f"## Section {section}", routine(doc, section, 1), routine(doc, section, 2)))
        if section == 14:
            sections.append(special)
    return "\n\n".join(sections)


DOCUMENTS = tuple((f"D{doc:02d}_records.pdf", pdf_document(doc)) for doc in range(1, 11)) + (
    ("D11_planning.txt", text_document(11, "Dispatch working notes", "UNSIGNED DRAFT: a planner considered an Orion D42 EAST cutoff of 17:45 at Bay E-08. This draft was not approved. Controlled circular O42-E9 is authoritative; the draft must not override it.")),
    ("D12_notices.txt", text_document(12, "Event registration notices", "HELIOS 2026-10-03 FINAL REGISTRATION NOTICE. A confirmed roster is available. The event registration fee has not been published in this document set. This absence must not be interpreted as free admission. The withdrawn internal estimate of 120 credits is not an approved fee.")),
    ("D13_facilities.txt", text_document(13, "Campus facilities archive", "The Lumen excursion uses a separate organizer and separate admissions. Its published 48-credit fee is unrelated to Helios. Parking is charged per vehicle-hour; no per-person laboratory registration amount can be derived from it.")),
    ("D14_mailroom.txt", text_document(14, "Internal mailroom and uniform orders", "Mailroom batch HEL-03 contains 91 envelopes and 14 spare labels. These are mail supplies, not Helios participant records. Uniform issue counts and courier visits must not be used to infer laboratory attendance.")),
)

IMAGES = (
    TableImage("D15_inventory.png", "EAST DEPOT - FINAL INVENTORY", "2026-09-17 | quantities are units | signed count",
               (("Cedar", "CE-A", "56", "8", "released"), ("Cedar", "CE-B", "17", "0", "quarantine"),
                ("Birch", "BI-A", "41", "5", "released"), ("Birch", "BI-B", "12", "0", "blocked"),
                ("Larch", "LA-A", "99", "2", "released"))),
    TableImage("D16_inventory.png", "EAST DEPOT - HISTORICAL INVENTORY", "2026-09-10 | ARCHIVED | not the current signed count",
               (("Cedar", "OLD-C", "82", "0", "released"), ("Birch", "OLD-B", "61", "0", "released"),
                ("Larch", "OLD-L", "75", "0", "released"))),
)

COMMON = "请只依据附件回答，并为每项关键结论注明文件名和PDF物理页码；图片注明文件名及相关行。不要把草稿、旧版或其他项目数据当作本次依据。"
COMPLEX_CASES = (
    Case("deep_lookup", "长文档深页定位与近似项目干扰",
         "Orion D42东站在2026年9月18日的冷链样本，最晚几点送到哪里，交接代码是什么？为什么不采用其他出现过的时间和地点？" + COMMON,
         DOCUMENTS, "18:35当地时间；Bay C-17；OR42-E9。D07_records.pdf物理第6页。旧版16:10/Bay A-03、未签草稿17:45/Bay E-08、D24西站17:25/Bay B-12均不适用。",
         ("18:35", "C-17", "OR42-E9", "D07_records.pdf"), suite="complex", images=IMAGES,
         evidence=(("D07_records.pdf", 6, "18:35 local time, delivery point Bay C-17, handover code OR42-E9"),),
         review_rubric=("核心时间地点代码均正确", "引用D07物理第6页", "排除撤回通知、草稿和西站数据", "没有虚构罚款或附加交接条件")),
    Case("visual_replenishment", "长PDF规则、颜色状态与图片库存联合计算",
         "根据东库2026年9月17日最终盘点和当日有效规则，Cedar与Birch各有多少可用库存、短缺多少、应订几包以及实际订购多少件？请说明图片中哪些批次被排除。" + COMMON,
         DOCUMENTS, "D15图片：绿色CE-A可用56-8=48，橙色CE-B排除；绿色BI-A可用41-5=36，红色BI-B排除。D02第4页目标90/65，短缺42/29。D09第3页每包6/10，订Cedar7包42件，Birch3包30件。D16是旧盘点，不可使用。",
         ("48", "36", "42", "29", "30", "D15_inventory.png", "D02_records.pdf", "D09_records.pdf"),
         needs_vision=True, suite="complex", images=IMAGES,
         evidence=(("D02_records.pdf", 4, "Target usable stock is Cedar 90 units and Birch 65 units"),
                   ("D09_records.pdf", 3, "Cedar is supplied in packages of 6 units"),
                   ("D15_inventory.png", None, "CE-A 56/8 released; BI-A 41/5 released; CE-B quarantine; BI-B blocked")),
         review_rubric=("读取当前图片并正确对应颜色图例", "扣除预留且排除待检冻结", "短缺42/29，整包订购7/3", "来源和物理页码正确，无虚构库存行")),
    Case("deduplicated_materials", "分散长文档去重、范围与预留口径",
         "Helios在2026年10月3日这场实验日有多少名不同的已确认参与者？蓝卡和红卡分别需要准备多少张？请逐步计算，明确重叠人员、教师、候补和预留卡片的处理方式。" + COMMON,
         DOCUMENTS, "D04第5页：18+27+16-6=55名不同参与者，教师3人和候补8人不计。D05第3页：蓝卡55*2+9=119，红卡55+5=60；预留全场一次。",
         ("55", "119", "60", "D04_records.pdf", "D05_records.pdf"), suite="complex", images=IMAGES,
         evidence=(("D04_records.pdf", 5, "Exactly six people are listed in both A and B"),
                   ("D05_records.pdf", 3, "Reserve nine additional blue cards and five additional red cards")),
         review_rubric=("参与者55，蓝卡119，红卡60", "重叠只扣一次，预留只加一次", "教师和候补不加入人数", "引用D04第5页和D05第3页")),
    Case("partial_answer_abstention", "已知事实与缺失金额混合，诱导数字干扰",
         "Helios在2026年10月3日需要准备多少张红卡？按已确认的不同参与者人数计算，主办方应该收取的报名费总额是多少？请区分附件能确定和不能确定的部分，不要猜测或把缺失价格当作免费。" + COMMON,
         DOCUMENTS, "55名不同参与者，红卡60张（D04第5页、D05第3页）。D12_notices.txt第14节明确报名费未公布，不能确定总额；D08第5页120只是撤回估算，D06第4页48是Lumen费用、6是停车费，均不能移用。",
         ("60", "D12_notices.txt"), suite="complex", images=IMAGES,
         evidence=(("D04_records.pdf", 5, "Exactly six people are listed in both A and B"),
                   ("D05_records.pdf", 3, "five additional red cards"),
                   ("D12_notices.txt", None, "event registration fee has not been published")),
         review_rubric=("正确回答红卡60", "明确报名费总额无法确定", "不把120/48/6当作报名费，也不声称免费", "来源支撑结论且未编造金额或审批信息")),
)
