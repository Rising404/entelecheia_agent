# 参考答案与证据修订记录

原始 DocBench QA 文件保持不变。本快照的 answer/evidence 已修订；原值、修订值、理由和来源保存在 dataset.sqlite 的 curation_events。问题文本未改写。

## docbench:167:0 — answer

原值：No, they can own only buildings, not the land itself, in certain investment zones.

修订：Yes, subject to geographic restrictions described in the document: foreign nationals may own land or real estate in designated locations in Dubai, and may acquire full real-estate rights in investment zones in Abu Dhabi.

理由：The original blanket No relied on the older building-only restriction and omitted the later investment-zone provisions and Dubai designated-location exception. This is a document-grounded answer, not an assertion about current law.

来源：{"batch_sha256": "5fb3be2bad0ff84db62f3238935c79607be41353268522090d991a4c50cbb7d8", "pages": [56], "source_unit_id": "docbench:167:chunk:ch_bcb1798b155910af93094015"}

## docbench:167:0 — evidence

原值：Law No. 19 of 2005, as amended, and Law No. 13 of 2019.

修订：The comparison table on PDF page 15 marks foreign ownership as permitted with investment-zone/designated-location restrictions. Page 56 contrasts the older building-only rule with the 2019 Abu Dhabi investment-zone provisions and explicitly states that foreign nationals may own land in designated Dubai locations.

理由：Restore all material exceptions described by the source rather than citing laws without their operative distinctions.

来源：{"batch_sha256": "5fb3be2bad0ff84db62f3238935c79607be41353268522090d991a4c50cbb7d8", "pages": [56], "source_unit_id": "docbench:167:chunk:ch_bcb1798b155910af93094015"}

## docbench:36:2 — answer

原值：singletask

修订：The singletask model performed best for French-to-English, with a BLEU score of 20.96.

理由：Original reference omitted the BLEU score explicitly requested by the unchanged query; Table 4 supplies 20.96.

来源：{"batch_sha256": "5fb3be2bad0ff84db62f3238935c79607be41353268522090d991a4c50cbb7d8", "pages": [8], "source_unit_id": "docbench:36:visual:ntu_45d33fa8cc459226d90b569f"}

## docbench:53:3 — answer

原值：The total finance income for the Group in 2021 was 48.

修订：The 2021 Finance income line was EUR 4 million. The broader Total income subtotal was EUR 48 million, comprising EUR 4 million of finance income, EUR 43 million of foreign exchange gains, and EUR 1 million of lease foreign exchange gains.

理由：The original query is retained, but its phrase total finance income is ambiguous: the table has distinct Finance income and Total income rows. State both labelled quantities instead of silently equating them or changing the query.

来源：{"batch_sha256": "5fb3be2bad0ff84db62f3238935c79607be41353268522090d991a4c50cbb7d8", "pages": [66], "source_unit_id": "docbench:53:visual:ntu_45726cc580c0a8d03188fd3a"}

## docbench:53:3 — evidence

原值：This information is directly provided in the 'Total income' row under the '2021' column in the table.

修订：On PDF page 66, Financial results has 2021 and 2020 columns. For 2021, Finance income is 4, Foreign exchange gains 43, Lease foreign exchange gains 1, and Total income 48. The consolidated accounts are expressed in millions of euros.

理由：Identify both differently labelled rows and the components of the subtotal; original evidence selected Total income without resolving the query wording.

来源：{"batch_sha256": "5fb3be2bad0ff84db62f3238935c79607be41353268522090d991a4c50cbb7d8", "pages": [66], "source_unit_id": "docbench:53:visual:ntu_45726cc580c0a8d03188fd3a"}

## docbench:67:4 — evidence

原值：The chart clearly shows that Wealth Management is the largest segment by net revenues in 2020, both in dollar amount and percentage (54% of total revenues).

修订：The 2020 net-revenue chart on PDF page 31 shows Institutional Securities at $25,948 million (54%), Wealth Management at $19,055 million (40%), and Investment Management at $3,734 million (8%). Institutional Securities is the largest segment. Percentages need not sum to 100% because of intersegment eliminations.

理由：Visually verified the chart, including its 2020 label and color legend. Original reference answer is correct, but original evidence incorrectly names Wealth Management as the 54% segment.

来源：{"batch_sha256": "5fb3be2bad0ff84db62f3238935c79607be41353268522090d991a4c50cbb7d8", "pages": [31], "source_unit_id": "docbench:67:visual:ntu_56582dc20883f06bfda54636"}

## docbench:80:5 — answer

原值：Heated Tobacco Units experienced the highest percentage increase in shipment volume, with an increase of 27.6%.

修订：Across all rows in the shipment table, the Heated Tobacco Units product category had the largest increase, at 27.6%. Among the individually named cigarette brands, Next had the largest increase, at 4.4%.

理由：The unchanged query says brand, but Heated Tobacco Units is a product category rather than an individual cigarette brand. Preserve the intended whole-table comparison and explicitly distinguish the narrower brand interpretation.

来源：{"batch_sha256": "5fb3be2bad0ff84db62f3238935c79607be41353268522090d991a4c50cbb7d8", "pages": [38], "source_unit_id": "docbench:80:visual:ntu_c790a5105d8ff1e69a0292bd"}

## docbench:80:5 — evidence

原值：By comparing the percentage change for all listed items, Heated Tobacco Units show the highest percentage increase which can be seen in the "Change" column.

修订：The complete PMI Shipment Volume by Brand table on PDF page 38 labels the columns Full-Year 2020, 2019, and Change, with shipment volume in million units. Heated Tobacco Units shows 76,111 versus 59,652 and 27.6%; Next shows 8,980 versus 8,602 and 4.4%. Every other individually named cigarette brand has a negative percentage change.

理由：Retain year, unit, comparison scope, and the distinction between a product-category row and named cigarette brands; the prior evidence did not resolve that distinction.

来源：{"batch_sha256": "5fb3be2bad0ff84db62f3238935c79607be41353268522090d991a4c50cbb7d8", "pages": [38], "source_unit_id": "docbench:80:visual:ntu_c790a5105d8ff1e69a0292bd"}

## docbench:85:0 — answer

原值：44.

修订：4 board meetings.

理由：Original PDF page 6 shows separate columns: 4 meetings held and 4 attended for Dominic LIM Kian Gam. The prior 44 concatenates two cells.

来源：{"batch_sha256": "5fb3be2bad0ff84db62f3238935c79607be41353268522090d991a4c50cbb7d8", "pages": [6], "source_unit_id": "docbench:85:chunk:transcription-ntu_5f6224e72b51ecae055891f4"}

## docbench:85:0 — evidence

原值：Dominic LIM Kian Gam 44

修订：On PDF page 6, the board attendance table lists Dominic LIM Kian Gam with 4 meetings held and 4 meetings attended in separate columns.

理由：Restore column semantics so the evidence no longer endorses the erroneous concatenated number.

来源：{"batch_sha256": "5fb3be2bad0ff84db62f3238935c79607be41353268522090d991a4c50cbb7d8", "pages": [6], "source_unit_id": "docbench:85:chunk:transcription-ntu_5f6224e72b51ecae055891f4"}
