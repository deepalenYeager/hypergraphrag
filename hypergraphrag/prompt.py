GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "中文"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["组织", "人物", "地点", "事件", "类别"]

PROMPTS["entity_extraction"] = """-目标-
给定一段可能与当前任务相关的文本文档，以及一组实体类型，请从文本中识别这些类型的全部实体，并识别已识别实体之间的全部关系。
请使用 {language} 作为输出语言。

-步骤-
1. 将文本划分为若干完整的知识片段。对每个知识片段，抽取以下信息：
-- knowledge_segment：一句描述该知识片段上下文的句子。
-- completeness_score：0 到 10 的分数，表示该知识片段的完整程度。
每个知识片段按如下格式输出：("hyper-relation"{tuple_delimiter}<knowledge_segment>{tuple_delimiter}<completeness_score>)

2. 识别每个知识片段中的全部实体。对每个已识别实体，抽取以下信息：
- entity_name：实体名称，使用与输入文本相同的语言；如果输入为英文，请保留英文实体原名并按常规大写。
- entity_type：实体类型。
- entity_description：对该实体属性和活动的全面描述。
- key_score：0 到 100 的分数，表示该实体在文本中的重要性。
每个实体按如下格式输出：("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<key_score>)

3. 将步骤 1 和步骤 2 识别出的全部实体和关系作为单个列表返回，输出语言为 {language}。使用 **{record_delimiter}** 作为列表分隔符。

4. 完成后，输出 {completion_delimiter}

######################
-示例-
######################
{examples}

#############################
-真实数据-
######################
文本：{input_text}
######################
输出：
"""

PROMPTS["entity_extraction_examples"] = [
    """示例 1：

文本：
在项目评审会上，张伟代表星河科技介绍了面向电力巡检的智能识别系统。该系统由研发部和华东电网联合测试，能够识别杆塔缺陷、鸟巢隐患和通道异物。李娜指出，系统在雨雾天气下的准确率仍需提升，因此建议下一阶段增加山区线路样本，并由运维中心负责数据标注。
################
输出：
("hyper-relation"{tuple_delimiter}"在项目评审会上，张伟代表星河科技介绍了面向电力巡检的智能识别系统。"{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"张伟"{tuple_delimiter}"人物"{tuple_delimiter}"张伟是代表星河科技在项目评审会上介绍智能识别系统的人。"{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"星河科技"{tuple_delimiter}"组织"{tuple_delimiter}"星河科技是张伟所属或代表的组织，参与面向电力巡检的智能识别系统项目。"{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"智能识别系统"{tuple_delimiter}"类别"{tuple_delimiter}"智能识别系统是面向电力巡检的系统，用于识别多类巡检风险。"{tuple_delimiter}92){record_delimiter}
("hyper-relation"{tuple_delimiter}"该系统由研发部和华东电网联合测试，能够识别杆塔缺陷、鸟巢隐患和通道异物。"{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"研发部"{tuple_delimiter}"组织"{tuple_delimiter}"研发部参与了智能识别系统的联合测试。"{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"华东电网"{tuple_delimiter}"组织"{tuple_delimiter}"华东电网参与了智能识别系统的联合测试。"{tuple_delimiter}88){record_delimiter}
("entity"{tuple_delimiter}"杆塔缺陷"{tuple_delimiter}"类别"{tuple_delimiter}"杆塔缺陷是智能识别系统能够识别的电力巡检风险之一。"{tuple_delimiter}82){record_delimiter}
("entity"{tuple_delimiter}"鸟巢隐患"{tuple_delimiter}"类别"{tuple_delimiter}"鸟巢隐患是智能识别系统能够识别的电力巡检风险之一。"{tuple_delimiter}82){record_delimiter}
("entity"{tuple_delimiter}"通道异物"{tuple_delimiter}"类别"{tuple_delimiter}"通道异物是智能识别系统能够识别的电力巡检风险之一。"{tuple_delimiter}82){record_delimiter}
("hyper-relation"{tuple_delimiter}"李娜指出系统在雨雾天气下的准确率仍需提升，并建议下一阶段增加山区线路样本，由运维中心负责数据标注。"{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}"李娜"{tuple_delimiter}"人物"{tuple_delimiter}"李娜指出智能识别系统在雨雾天气下准确率仍需提升，并提出下一阶段改进建议。"{tuple_delimiter}93){record_delimiter}
("entity"{tuple_delimiter}"雨雾天气"{tuple_delimiter}"类别"{tuple_delimiter}"雨雾天气是影响智能识别系统准确率的场景条件。"{tuple_delimiter}78){record_delimiter}
("entity"{tuple_delimiter}"山区线路样本"{tuple_delimiter}"类别"{tuple_delimiter}"山区线路样本是李娜建议下一阶段增加的数据样本。"{tuple_delimiter}84){record_delimiter}
("entity"{tuple_delimiter}"运维中心"{tuple_delimiter}"组织"{tuple_delimiter}"运维中心被建议负责下一阶段的数据标注工作。"{tuple_delimiter}86)
#############################""",
    """示例 2：

文本：
南湖变电站改造工程于2024年6月启动，建设单位为江城供电公司。工程重点包括更换主变压器、升级继电保护装置以及新建远程监控平台。项目经理王敏要求施工队在汛期前完成主设备安装，因为南湖片区夏季负荷增长明显，供电可靠性压力较大。
#############
输出：
("hyper-relation"{tuple_delimiter}"南湖变电站改造工程于2024年6月启动，建设单位为江城供电公司。"{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"南湖变电站改造工程"{tuple_delimiter}"事件"{tuple_delimiter}"南湖变电站改造工程是于2024年6月启动的工程项目。"{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"江城供电公司"{tuple_delimiter}"组织"{tuple_delimiter}"江城供电公司是南湖变电站改造工程的建设单位。"{tuple_delimiter}92){record_delimiter}
("hyper-relation"{tuple_delimiter}"工程重点包括更换主变压器、升级继电保护装置以及新建远程监控平台。"{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"主变压器"{tuple_delimiter}"类别"{tuple_delimiter}"主变压器是南湖变电站改造工程计划更换的主设备。"{tuple_delimiter}86){record_delimiter}
("entity"{tuple_delimiter}"继电保护装置"{tuple_delimiter}"类别"{tuple_delimiter}"继电保护装置是南湖变电站改造工程计划升级的设备。"{tuple_delimiter}86){record_delimiter}
("entity"{tuple_delimiter}"远程监控平台"{tuple_delimiter}"类别"{tuple_delimiter}"远程监控平台是南湖变电站改造工程计划新建的平台。"{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"项目经理王敏要求施工队在汛期前完成主设备安装，因为南湖片区夏季负荷增长明显，供电可靠性压力较大。"{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}"王敏"{tuple_delimiter}"人物"{tuple_delimiter}"王敏是南湖变电站改造工程的项目经理，要求施工队在汛期前完成主设备安装。"{tuple_delimiter}94){record_delimiter}
("entity"{tuple_delimiter}"施工队"{tuple_delimiter}"组织"{tuple_delimiter}"施工队被要求在汛期前完成南湖变电站改造工程的主设备安装。"{tuple_delimiter}88){record_delimiter}
("entity"{tuple_delimiter}"南湖片区"{tuple_delimiter}"地点"{tuple_delimiter}"南湖片区在夏季负荷增长明显，面临较大的供电可靠性压力。"{tuple_delimiter}87){record_delimiter}
("entity"{tuple_delimiter}"汛期"{tuple_delimiter}"事件"{tuple_delimiter}"汛期是王敏要求完成主设备安装之前的关键时间节点。"{tuple_delimiter}80)
#############################""",
    """示例 3：

文本：
《城市配电网防鸟害技术导则》提出，鸟害高发区应优先采用绝缘护套、驱鸟刺和声光驱鸟器组合防护。导则还要求运行单位建立鸟巢清理台账，并在春季繁殖期加强巡视。对于生态保护区内的线路，文件强调不得随意破坏鸟类栖息地，应采用非伤害性措施降低跳闸风险。
#############
输出：
("hyper-relation"{tuple_delimiter}"《城市配电网防鸟害技术导则》提出，鸟害高发区应优先采用绝缘护套、驱鸟刺和声光驱鸟器组合防护。"{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}"城市配电网防鸟害技术导则"{tuple_delimiter}"类别"{tuple_delimiter}"《城市配电网防鸟害技术导则》提出了城市配电网鸟害防护的技术要求。"{tuple_delimiter}96){record_delimiter}
("entity"{tuple_delimiter}"鸟害高发区"{tuple_delimiter}"地点"{tuple_delimiter}"鸟害高发区是导则建议优先采用组合防护措施的区域。"{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"绝缘护套"{tuple_delimiter}"类别"{tuple_delimiter}"绝缘护套是鸟害高发区优先采用的防护措施之一。"{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"驱鸟刺"{tuple_delimiter}"类别"{tuple_delimiter}"驱鸟刺是鸟害高发区优先采用的防护措施之一。"{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"声光驱鸟器"{tuple_delimiter}"类别"{tuple_delimiter}"声光驱鸟器是鸟害高发区优先采用的防护措施之一。"{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"导则要求运行单位建立鸟巢清理台账，并在春季繁殖期加强巡视。"{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"运行单位"{tuple_delimiter}"组织"{tuple_delimiter}"运行单位被要求建立鸟巢清理台账，并在春季繁殖期加强巡视。"{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"鸟巢清理台账"{tuple_delimiter}"类别"{tuple_delimiter}"鸟巢清理台账是运行单位需要建立的管理记录。"{tuple_delimiter}86){record_delimiter}
("entity"{tuple_delimiter}"春季繁殖期"{tuple_delimiter}"事件"{tuple_delimiter}"春季繁殖期是运行单位需要加强巡视的时期。"{tuple_delimiter}84){record_delimiter}
("hyper-relation"{tuple_delimiter}"对于生态保护区内的线路，文件强调不得随意破坏鸟类栖息地，应采用非伤害性措施降低跳闸风险。"{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}"生态保护区"{tuple_delimiter}"地点"{tuple_delimiter}"生态保护区内的线路需要采取不破坏鸟类栖息地的防鸟害措施。"{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"鸟类栖息地"{tuple_delimiter}"地点"{tuple_delimiter}"鸟类栖息地在生态保护区线路防护中不得被随意破坏。"{tuple_delimiter}86){record_delimiter}
("entity"{tuple_delimiter}"非伤害性措施"{tuple_delimiter}"类别"{tuple_delimiter}"非伤害性措施是生态保护区线路降低跳闸风险时应采用的措施。"{tuple_delimiter}88){record_delimiter}
("entity"{tuple_delimiter}"跳闸风险"{tuple_delimiter}"类别"{tuple_delimiter}"跳闸风险是生态保护区线路通过非伤害性措施需要降低的风险。"{tuple_delimiter}87)
#############################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """你是一个负责根据下方数据生成全面摘要的有用助手。
给定一个或两个实体，以及一组描述；这些描述都与同一个实体或同一组实体相关。
请将这些描述整合为一个单一、全面的描述，并确保纳入所有描述中收集到的信息。
如果提供的描述相互矛盾，请解决矛盾，并给出单一、连贯的摘要。
请使用第三人称写作，并包含实体名称，以便保留完整上下文。
请使用 {language} 作为输出语言。

#######
-数据-
实体：{entity_name}
描述列表：{description_list}
#######
输出：
"""

PROMPTS[
    "entiti_continue_extraction"
] = """上一次抽取遗漏了许多包含实体的知识片段。请使用相同格式在下方补充：
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """请检查已抽取的知识片段是否覆盖了给定文本的全部内容。如果仍有需要补充的知识片段，请回答 YES；否则回答 NO。只回答 YES 或 NO。
"""

PROMPTS["fail_response"] = "抱歉，我无法回答这个问题。"

PROMPTS["rag_response"] = """---角色---

你是一个有用的助手，负责回答与所提供表格数据相关的问题。


---目标---

生成符合目标长度和格式的回答，以回应用户问题；根据回答长度和格式，汇总输入数据表中的全部相关信息，并结合必要的通用知识。
如果你不知道答案，请直接说明。不要编造。
不要包含缺乏支持证据的信息。

---目标回答长度和格式---

{response_type}

---数据表---

{context_data}

请根据目标长度和格式，在回答中适当添加章节和说明。回答样式使用 Markdown。
"""

# PROMPTS["rag_response"] = """---角色---

# 你是一个智能且精确的 AI 助手，基于结构化数据表回答问题。


# ---目标---

# 生成语义准确、事实正确且高度相关的回答，直接回应用户问题。回答应当：
#   • 最大化与期望答案的语义一致性，确保高相似度。
#   • 确保事实正确，保留数据中的关键细节、名称、数字和关系。
#   • 与用户问题完全相关，避免不必要信息，同时保证完整性。
#   • 使用结构化格式（标题、项目符号、表格）提升清晰度和连贯性。
#   • 保持自然且精确的写作风格，提升可读性。

# ---目标回答长度和格式---

# {response_type}

# ---数据表---

# {context_data}

# 回答准则
#   1. 优先保留关键细节：抽取并总结最相关的信息，同时保持完整性。
#   2. 保持语义一致：确保表达尽量贴近期望答案以提升相似度。
#   3. 保留关键实体和结构：必须正确保留名称、日期、数字和关系。
#   4. 确保逻辑流畅：用提升清晰度和连贯性的方式组织回答。
#   5. 简洁且相关：避免冗余细节，专注于直接回答问题。
# """

PROMPTS["keywords_extraction"] = """---角色---

你是一个有用的助手，负责识别用户查询中的高层关键词和低层关键词。

---目标---

给定查询，请列出高层关键词和低层关键词。高层关键词关注总体概念或主题；低层关键词关注具体实体、细节或具体术语。

---说明---

- 使用 JSON 格式输出关键词。
- JSON 必须包含两个键：
  - "high_level_keywords"：表示总体概念或主题。
  - "low_level_keywords"：表示具体实体或细节。

######################
-示例-
######################
{examples}

#############################
-真实数据-
######################
查询：{query}
######################
输出内容应为可读文本，不要使用 Unicode 转义字符。保持与查询相同的语言。
输出：

"""

PROMPTS["keywords_extraction_examples"] = [
    """示例 1：

查询："国际贸易如何影响全球经济稳定？"
################
输出：
{{
  "high_level_keywords": ["国际贸易", "全球经济稳定", "经济影响"],
  "low_level_keywords": ["贸易协定", "关税", "汇率", "进口", "出口"]
}}
#############################""",
    """示例 2：

查询："森林砍伐会对生物多样性造成哪些环境后果？"
################
输出：
{{
  "high_level_keywords": ["环境后果", "森林砍伐", "生物多样性丧失"],
  "low_level_keywords": ["物种灭绝", "栖息地破坏", "碳排放", "雨林", "生态系统"]
}}
#############################""",
    """示例 3：

查询："教育在减少贫困方面起什么作用？"
################
输出：
{{
  "high_level_keywords": ["教育", "减贫", "社会经济发展"],
  "low_level_keywords": ["入学机会", "识字率", "职业培训", "收入不平等"]
}}
#############################""",
]


PROMPTS["naive_rag_response"] = """---角色---

你是一个有用的助手，负责回答与所提供文档相关的问题。


---目标---

生成符合目标长度和格式的回答，以回应用户问题；根据回答长度和格式，汇总输入数据表中的全部相关信息，并结合必要的通用知识。
如果你不知道答案，请直接说明。不要编造。
不要包含缺乏支持证据的信息。

---目标回答长度和格式---

{response_type}

---文档---

{content_data}

请根据目标长度和格式，在回答中适当添加章节和说明。回答样式使用 Markdown。
"""

PROMPTS[
    "similarity_check"
] = """请分析以下两个问题之间的相似度：

问题 1：{original_prompt}
问题 2：{cached_prompt}

请评估以下两点，并直接给出 0 到 1 之间的相似度分数：
1. 这两个问题在语义上是否相似
2. 问题 2 的答案是否可以用于回答问题 1
相似度评分标准：
0：完全无关，或答案不能复用，包括但不限于：
   - 问题主题不同
   - 问题中提到的地点不同
   - 问题中提到的时间不同
   - 问题中提到的具体人物不同
   - 问题中提到的具体事件不同
   - 问题中的背景信息不同
   - 问题中的关键条件不同
1：完全相同，答案可以直接复用
0.5：部分相关，答案需要修改后才能使用
只返回 0 到 1 之间的一个数字，不要返回任何额外内容。
"""
