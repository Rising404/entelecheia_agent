"""Four comparable PDFs; every question has exactly one sufficient source PDF."""

from dataclasses import dataclass

from .cases import Case


@dataclass(frozen=True)
class Activity:
    key: str
    project: str
    station: str
    date: str
    deadline: str
    room: str
    receiver: str
    handoff_code: str
    confirmed: int
    guides: int
    waiting: int
    cards_each: int
    spare_cards: int
    device_prefix: str
    minimum_charge: int
    charges: tuple[int, int, int, int]

    @property
    def scope(self) -> str:
        return f"{self.project}项目{self.station}"

    @property
    def handoff(self) -> str:
        return f"设备须于{self.deadline}前送达{self.room}，由{self.receiver}接收。"

    @property
    def material_rule(self) -> str:
        return (
            f"材料卡按每名已确认正式参与者{self.cards_each}张发放；"
            f"每场另备{self.spare_cards}张，全场只加一次。"
        )

    @property
    def roster(self) -> str:
        return (
            f"{self.scope}最终名单：已确认正式参与者{self.confirmed}人；"
            f"引导员{self.guides}人；未确认候补{self.waiting}人。"
        )

    @property
    def lending_rule(self) -> str:
        return f"可借设备必须同时满足状态为可用、电量至少{self.minimum_charge}%；恰好达到阈值也合格。"

    @property
    def device_rows(self) -> tuple[tuple[str, str, int], ...]:
        return tuple(
            (f"{self.device_prefix}-{11 + index}", status, charge)
            for index, (status, charge) in enumerate(
                zip(("可用", "可用", "维修中", "可用"), self.charges, strict=True)
            )
        )


ACTIVITIES = (
    Activity("qinglan_north", "青岚", "北站", "2026年6月18日", "16:20", "B楼106室", "陆溪", "QN-728",
             23, 2, 4, 3, 7, "QN", 60, (60, 82, 91, 55)),
    Activity("qinglan_south", "青岚", "南站", "2026年6月19日", "15:35", "C楼208室", "唐越", "QS-614",
             17, 3, 6, 4, 6, "QS", 70, (70, 88, 93, 64)),
    Activity("chengchuan_north", "澄川", "北站", "2026年6月20日", "17:10", "A楼315室", "孟川", "CN-953",
             28, 2, 5, 2, 9, "CN", 65, (65, 79, 92, 59)),
    Activity("chengchuan_south", "澄川", "南站", "2026年6月21日", "14:45", "D楼107室", "叶舟", "CS-482",
             19, 4, 3, 5, 4, "CS", 80, (80, 96, 98, 72)),
)


def _document(activity: Activity) -> str:
    """Physical pages are explicit; each PDF is independently sufficient."""
    a = activity
    devices = "\n".join(f"| {code} | {status} | {charge}% |" for code, status, charge in a.device_rows)
    return "\f".join((
        f"{a.scope}｜活动执行手册\n"
        f"适用范围：本手册仅用于{a.date}的{a.scope}现场活动。登记、设备交接、物料发放和设备借用均采用本手册所列口径。其他项目或站点的材料不得替代本场记录。\n"
        "一、活动概况\n"
        "本场包括入场登记、设备观察、分组操作与结束整理。所有参与者先完成签到，再由引导员带入操作区。活动当天不安排跨站流动，也不把其他场次的报名人数合并到本场。\n"
        "二、岗位与登记\n"
        "登记台负责核对报名编号；引导员负责现场秩序；设备接收人员负责核对箱数和外观。岗位名称并不意味着同一人同时承担所有职责，交接责任以第二页的正式安排为准。\n"
        "三、文档使用范围\n"
        "手册共四个物理页面。第二页记录物流交接，第三页列出材料与设备借用规则，第四页保存最终人数与设备台账。页面下方标示的页数就是物理页码。\n"
        "四、现场记录\n"
        "发现包装破损时先留存照片并登记，不在走廊拆修。签到表记录的是到场情况，不能自行改变已确认名单的统计口径。活动结束后分别归还设备和未使用材料，保留清点记录。",

        f"{a.scope}｜物流与交接\n"
        f"本页适用于{a.date}的{a.scope}。本场设备为教学观察组件，外箱按装箱单编号，电池与传感器分别包装。运输前由发出方确认配件清单，并将纸质清单随箱放置。\n"
        "一、正式交接安排\n"
        f"{a.handoff}交接代码为{a.handoff_code}。上述时间为本地时间，指设备实际送达截止时间，不是司机出发时间。接收人员核对箱数后在交接单上签字，司机保留回执。\n"
        "二、路线与操作\n"
        "车辆从园区东门进入，按访客流程登记后前往装卸点。到达后先联系接收人员，禁止把设备留在无人值守的门口。装卸完成后及时移车，不占用消防通道或无障碍通道。\n"
        "三、另行安排\n"
        "现场工作人员于当日10:00在公共会议区举行协调会。协调会讨论分组与签到，不办理设备交接。楼宇开放时间为08:00至18:00；开放时间不改变正式交接要求。\n"
        "四、异常登记\n"
        "如运输延误，应登记实际到达时间及原因并联系接收人员。未取得新的正式确认前，不能仅因会议、门禁或车辆调度变化就认定交接地点或截止时间已经调整。",

        f"{a.scope}｜材料与设备规则\n"
        f"本页规则适用于{a.scope}本次活动，与第四页的最终人数和设备台账共同使用。本场只举办一场，不按小组、签到批次或设备桌数量重复计算场次。\n"
        "一、材料卡口径\n"
        f"{a.material_rule}引导员不领取材料卡，未确认候补不计入正式参与人数。第四页列出的正式参与者已经完成确认，不再额外扣除任何人员。\n"
        "材料卡按张计数，无整包购买要求。备用卡独立于按人发放部分；在计算本场需要准备的总数时，应先按正式参与人数计算，再加一次全场备用量。此处要求准备总量，不是从现存库存中扣减后的采购量。\n"
        "二、设备借用口径\n"
        f"{a.lending_rule}处于维修中的设备即使电量高也不可借出；可用但电量不足的设备不能按预计充电后的状态计入。只根据第四页本次盘点的状态和电量判断。\n"
        "三、记录约束\n"
        "设备编号区分大小写，同一编号对应一台设备。旧照片、外箱容量和未签署的口头说明均不改变当前台账。本页未要求按人数分配设备，也未设置额外的预留设备数量。",

        f"{a.scope}｜最终名单与设备台账\n"
        f"本页为{a.date}的{a.scope}最终确认记录，人数和设备状态均已核对。该页不代表其他日期、项目或站点的情况。\n"
        "一、人数确认\n"
        f"{a.roster}三类人员分别登记，不存在重复编号。候补尚未转为正式参与者，引导员承担现场职责而不作为领卡参与者。材料规则详见本手册第三页。\n"
        "二、本次设备盘点\n"
        "下面四行列出本场待选的全部设备。电量为当前读数，不是目标电量或上次活动结束时的读数。维修状态在本次盘点时仍然有效。\n"
        "| 设备编号 | 当前状态 | 当前电量 |\n"
        f"{devices}\n"
        "三、签收与结束清点\n"
        "借出时登记设备编号和经手人；归还时核对编号、配件与外观。不得将未列入此表的设备推定为可借，也不得用房间座位数推断参与人数。活动后的实际损耗记录另行保存，不回写本次准备口径。",
    ))


# Neutral names and identical structure prevent filenames from revealing the answer.
PDF_DOCUMENTS = tuple((f"document_{index:02d}.pdf", _document(activity))
                      for index, activity in enumerate(ACTIVITIES, 1))


def _questions() -> tuple[Case, ...]:
    cases = []
    instruction = "请仅依据附件回答，注明证据所在的PDF文件名和物理页码。"
    for activity, (filename, _) in zip(ACTIVITIES, PDF_DOCUMENTS, strict=True):
        a = activity
        common = {"documents": PDF_DOCUMENTS, "suite": "pdf"}
        cases.append(Case(
            id=f"pdf_{a.key}_handoff", scenario="唯一PDF定位：交接时间、地点与责任人",
            question=f"{a.scope}本次活动的设备最晚几点送达、送到哪里、由谁接收？{instruction}",
            reference=f"{a.deadline}前，{a.room}，{a.receiver}；唯一来源{filename}第2页。",
            expected_facts=(a.deadline, a.room, a.receiver, filename),
            evidence=((filename, 2, a.handoff),),
            review_rubric=("时间、房间和接收人均正确", "选中唯一目标PDF并引用第2页",
                           "不混入同项目其他站点或另一项目的数据", "不把会议或楼宇开放时间当作交接安排"),
            **common,
        ))
        total = a.confirmed * a.cards_each + a.spare_cards
        cases.append(Case(
            id=f"pdf_{a.key}_materials", scenario="同一PDF跨页组合：正式人数与材料卡计算",
            question=f"{a.scope}本次活动需要准备多少张材料卡？请给出计入人数、计算过程及排除人员的依据。{instruction}",
            reference=f"正式参与者{a.confirmed}人，{a.confirmed}×{a.cards_each}+{a.spare_cards}={total}张；唯一来源{filename}第3、4页。",
            expected_facts=(str(a.confirmed), str(total), filename),
            evidence=((filename, 3, a.material_rule), (filename, 4, a.roster)),
            review_rubric=(f"人数{a.confirmed}、总数{total}正确", "备用量全场只加一次，排除引导员和未确认候补",
                           "全部运算输入来自同一目标PDF", "引用规则第3页及名单第4页"),
            **common,
        ))
        eligible = tuple(code for code, status, charge in a.device_rows
                         if status == "可用" and charge >= a.minimum_charge)
        cases.append(Case(
            id=f"pdf_{a.key}_devices", scenario="同一PDF内规则与表格：设备条件筛选",
            question=f"依据{a.scope}本次活动的规则和台账，哪些设备可以借出？请列出编号，并逐台说明判断依据。{instruction}",
            reference=f"仅{'、'.join(eligible)}可以借出；维修设备和电量不足设备不合格；唯一来源{filename}第3、4页。",
            expected_facts=(*eligible, filename),
            evidence=((filename, 3, a.lending_rule), *(
                (filename, 4, f"{code} {status} {charge}%") for code, status, charge in a.device_rows
            )),
            review_rubric=(f"合格集合恰为{'、'.join(eligible)}", "恰好达到电量阈值合格，维修或不足阈值不合格",
                           "不借用其他PDF的规则或设备编号", "引用规则第3页及台账第4页"),
            **common,
        ))
    return tuple(cases)


PDF_CASES = _questions()
