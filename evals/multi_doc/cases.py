"""Original synthetic multi-document questions; answers never enter the Project."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from .fixture_documents import TableImage, write_pdf, write_table_image


@dataclass(frozen=True)
class SourcePDF:
    name: str
    path: Path
    sha256: str
    physical_pages: int


@dataclass(frozen=True)
class Case:
    id: str
    scenario: str
    question: str
    documents: tuple[tuple[str, str], ...]
    reference: str
    expected_facts: tuple[str, ...]
    needs_vision: bool = False
    suite: str = "smoke"
    images: tuple[TableImage, ...] = ()
    evidence: tuple[tuple[str, int | None, str], ...] = ()
    review_rubric: tuple[str, ...] = ()
    source_pdfs: tuple[SourcePDF, ...] = ()


CASES = (
    Case(
        "single_source", "答案只在四份文档中的一份",
        "青岚项目的冷链样本最晚何时送达、送到哪里？仅依据附件回答并标注文件名。",
        (
            ("01_meeting.md", "# 青岚项目会议纪要\n周三09:00讨论传感器校准。会议在B楼201室。会议地点不是样本交接点。\n"),
            ("02_delivery.md", "# 冷链交接通知\n青岚项目：样本最晚于周四17:40送达C楼冷藏室，交接代码QL-728。\n赤湾项目：周五15:10送达A楼。上述时间均为本地时间。\n"),
            ("03_budget.md", "# 青岚采购预算\n温度记录仪预算8200元，包装耗材预算1300元。财务发票在D楼提交。\n"),
            ("04_roster.txt", "青岚项目人员：记录员林桥，设备员叶舟。培训签到时间为周二08:30。"),
        ),
        "周四17:40，C楼冷藏室；证据来自02_delivery.md。",
        ("17:40", "C楼", "02_delivery"),
    ),
    Case(
        "cross_document", "跨文档筛选与计算",
        "本周六星港工作坊需要准备多少张材料卡？请结合附件给出人数、计算过程和文件依据。",
        (
            ("01_rules.md", "# 星港工作坊材料规则\n每名正式参与者领取3张材料卡，每场另备7张。引导员不领卡；候补人员未获确认不计入。每周六仅举办一场。\n"),
            ("02_confirmed.txt", "星港本周六名单：已确认参与者23人；引导员2人；未确认候补5人。"),
            ("03_archive.txt", "星港上周六归档：正式参与者18人，引导员3人。此档仅记录上周活动。"),
            ("04_schedule.md", "# 本周安排\n周六星港工作坊14:00开始，16:00结束。周日另有观鸟活动12人。\n"),
        ),
        "23名正式参与者，23×3+7=76张；01_rules.md和02_confirmed.txt，不计引导员、候补或上周人员。",
        ("23", "76", "01_rules", "02_confirmed"),
    ),
    Case(
        "distractor_revision", "同名项目、旧通知和相邻日期干扰",
        "依据附件中生效的安排，北站的澄星项目验收应在什么日期、哪个房间进行？请解释为什么不采用其他通知。",
        (
            ("01_initial.txt", "澄星项目北站初始通知，发布于2026-04-02：4月18日在B-201验收。此通知已被4月9日修订通知替代。"),
            ("02_south.txt", "澄星项目南站通知，发布于2026-04-10：4月21日在D-101验收。仅适用于南站。"),
            ("03_revision.txt", "澄星项目北站生效修订通知，发布于2026-04-09：替代4月2日初始通知；正式验收改为2026年4月20日，地点C-305。"),
            ("04_rehearsal.txt", "澄星项目北站彩排：2026年4月19日，地点C-302。彩排不等于正式验收。"),
        ),
        "2026年4月20日，C-305；03_revision.txt取代初始通知，南站和北站彩排均不适用。",
        ("20", "C-305", "03_revision"),
    ),
    Case(
        "missing_information", "全部文档均未提供答案",
        "参加石桥科学营每人要交多少报名费？请仅依据附件；若无法确定，明确说明，不要将未提及收费解释为免费。",
        (
            ("01_program.txt", "石桥科学营：周六09:00至17:00，课程为天文观察与传感器制作。"),
            ("02_materials.txt", "主办方统一准备护目镜和实验材料。参与者需自带午餐和水壶。"),
            ("03_registration.txt", "报名截止周三18:00，最多30人；登记姓名和紧急联系人。报名确认会以编号回复。"),
            ("04_transport.txt", "场地附近公共停车场收费每小时6元，此费用由停车场收取，与科学营报名无关。"),
        ),
        "附件未提供报名费，无法确定金额；停车费不是报名费，也不能推断免费。",
        (),
    ),
    Case(
        "visual_source", "图片独有数据与文本规则联合计算",
        "请读取附件中本周的库存图，并结合补货规则，说明 Cedar 和 Birch 各需补多少件。标明图像和规则文件依据。",
        (
            ("01_policy.txt", "补货规则：Cedar的目标库存为50件，Birch为35件。补货数量=max(目标库存-本周现存库存,0)。本周数据仅见02_inventory.png；不得使用上周数据。"),
            ("03_last_week.txt", "上周库存归档：Cedar 49件，Birch 18件。此记录不代表本周。"),
            ("04_shipping.txt", "仓库周二和周五统一发货。每箱最多装10件，允许零散装箱。"),
        ),
        "02_inventory.png本周Cedar=37、Birch=22；结合01_policy.txt，两者均需补13件。",
        ("37", "22", "13", "02_inventory", "01_policy"),
        needs_vision=True,
    ),
)


def materialize(case: Case, directory: Path) -> list[Path]:
    """Write only attachments, never question/reference metadata, into this folder."""
    directory.mkdir(parents=True, exist_ok=False)
    paths = []
    if case.source_pdfs:
        if case.documents or case.images:
            raise ValueError("source PDFs cannot be mixed with generated fixtures")
        for source in case.source_pdfs:
            if Path(source.name).name != source.name or not source.name.endswith(".pdf"):
                raise ValueError("source PDF needs a plain PDF filename")
            content = source.path.read_bytes()
            if hashlib.sha256(content).hexdigest() != source.sha256:
                raise ValueError(f"source PDF hash drift: {source.name}")
            path = directory / source.name
            with path.open("xb") as stream:
                stream.write(content)
            paths.append(path)
        return paths
    for name, content in case.documents:
        path = directory / name
        if path.suffix == ".pdf":
            write_pdf(path, content)
        else:
            path.write_text(content, encoding="utf-8")
        paths.append(path)
    for spec in case.images:
        path = directory / spec.name
        write_table_image(path, spec)
        paths.append(path)
    if case.needs_vision and not case.images:
        from PIL import Image, ImageDraw, ImageFont

        picture = Image.new("RGB", (1000, 600), "white")
        draw = ImageDraw.Draw(picture)
        font = ImageFont.load_default(size=38)
        draw.text((70, 35), "THIS WEEK - INVENTORY (units)", fill="black", font=font)
        for top, label, value, color in (
            (170, "Cedar", 37, "#276AB3"), (350, "Birch", 22, "#348665"),
        ):
            draw.text((65, top), label, fill="black", font=font)
            draw.rectangle((250, top, 250 + value * 14, top + 85), fill=color)
            draw.text((270 + value * 14, top + 20), str(value), fill="black", font=font)
        path = directory / "02_inventory.png"
        picture.save(path)
        paths.append(path)
    return sorted(paths)
