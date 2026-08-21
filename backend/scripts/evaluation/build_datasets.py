"""Build and validate the three MedAgent evaluation datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"
KNOWLEDGE_ROOT = BACKEND_ROOT / "data" / "knowledge"
EVAL_ROOT = BACKEND_ROOT / "data" / "eval"

DEPARTMENTS = (
    "general_medical",
    "general_surgery",
    "pediatrics",
    "neurology",
    "infectious_disease",
    "ent",
    "ophthalmology",
    "dermatology",
)

# Gold pages are curated against the PDF page numbers exposed by pypdf.  Do not
# infer them from the longest page containing a broad keyword: large reports
# repeat terms in tables of contents, references, and unrelated sections.
GOLD_PAGES = (
    17,
    18,
    10,
    15,
    8,
    53,
    104,
    20,
    20,
    91,
    84,
    25,
    81,
    25,
    34,
    10,
    7,
    29,
    74,
    38,
    108,
    153,
    225,
    182,
    193,
    111,
    31,
    62,
    213,
    204,
    342,
    13,
    17,
    33,
    49,
    82,
    70,
    143,
    12,
    41,
    45,
    119,
    54,
    100,
    14,
    11,
    37,
    30,
    36,
    23,
)


@dataclass(frozen=True)
class Topic:
    department: str
    source: str
    anchor: str
    question: str
    hard_question: str
    min_page: int = 1


TOPICS = (
    Topic(
        "dermatology",
        "WHO_Scabies_Diagnostic_TPP_2022.pdf",
        "confirmatory test sensitivity",
        "疥疮确证检测的目标敏感度要求是什么？",
        "初筛后再做疥疮确认时，检测漏诊能力应达到什么目标？",
        8,
    ),
    Topic(
        "dermatology",
        "WHO_Scabies_Diagnostic_TPP_2022.pdf",
        "confirmatory test specificity",
        "疥疮确证检测的目标特异度要求是什么？",
        "初筛后用于确认疥疮的工具，排除非病例的能力应达到什么目标？",
        9,
    ),
    Topic(
        "dermatology",
        "WHO_Scabies_Diagnostic_TPP_2022.pdf",
        "Mass Drug Administration Start",
        "启动疥疮群体药物管理时，诊断产品用于什么场景？",
        "社区准备开展疥疮集体用药前，现场检测工具承担什么作用？",
        9,
    ),
    Topic(
        "dermatology",
        "WHO_Scabies_Diagnostic_TPP_2022.pdf",
        "Mass Drug Administration Stop",
        "停止疥疮群体药物管理时，诊断产品用于什么场景？",
        "社区多轮疥疮集体用药后，什么检测用途支持停止决策？",
        9,
    ),
    Topic(
        "dermatology",
        "WHO_Scabies_Diagnostic_TPP_2022.pdf",
        "target product profile",
        "WHO 疥疮诊断目标产品概况关注哪些产品要求？",
        "疥疮现场诊断工具在研发时需要按哪类目标规格进行约束？",
        8,
    ),
    Topic(
        "ent",
        "WHO_World_Report_on_Hearing.pdf",
        "safe listening",
        "安全聆听为什么能预防噪声性听力损失？",
        "耳蜗受到的声音暴露越大损伤越重，这对日常用耳有什么启示？",
        30,
    ),
    Topic(
        "ent",
        "WHO_World_Report_on_Hearing.pdf",
        "newborn hearing screening",
        "新生儿听力筛查在听力保健中有什么作用？",
        "婴儿尚不能表达听力问题时，卫生系统如何尽早识别异常？",
        30,
    ),
    Topic(
        "ent",
        "WHO_World_Report_on_Hearing.pdf",
        "hearing aids",
        "听力辅助技术包含哪些常见干预？",
        "听力下降后，除药物或手术外有哪些设备和沟通支持可选？",
        18,
    ),
    Topic(
        "ent",
        "WHO_World_Report_on_Hearing.pdf",
        "cochlear implants",
        "人工耳蜗可为听力损失患者提供什么帮助？",
        "重度听力受损者可通过哪类植入式装置改善教育和沟通机会？",
        18,
    ),
    Topic(
        "ent",
        "WHO_World_Report_on_Hearing.pdf",
        "otitis media",
        "中耳炎为什么是耳与听力保健的重要问题？",
        "儿童常见的中耳疾病如何进入综合听力照护范围？",
        30,
    ),
    Topic(
        "ophthalmology",
        "WHO_World_Report_on_Vision.pdf",
        "refractive error",
        "未矫正屈光不正如何影响视力？",
        "眼球聚焦误差没有得到眼镜等矫正时，会造成什么视觉负担？",
        20,
    ),
    Topic(
        "ophthalmology",
        "WHO_World_Report_on_Vision.pdf",
        "cataract",
        "白内障是怎样影响视力的？",
        "晶状体混浊造成视物下降时，报告如何描述这一眼病负担？",
        20,
    ),
    Topic(
        "ophthalmology",
        "WHO_World_Report_on_Vision.pdf",
        "glaucoma",
        "青光眼为什么需要早期发现和持续管理？",
        "一种可逐渐损伤视神经的眼病，为何不能只等明显视力下降后处理？",
        20,
    ),
    Topic(
        "ophthalmology",
        "WHO_World_Report_on_Vision.pdf",
        "diabetic retinopathy",
        "糖尿病视网膜病变与视力损害有什么关系？",
        "长期高血糖累及眼底微血管时，报告强调了什么视觉风险？",
        20,
    ),
    Topic(
        "ophthalmology",
        "WHO_World_Report_on_Vision.pdf",
        "myopia",
        "近视为何需要公共卫生层面的关注？",
        "看远处模糊且患病人数增长的屈光问题，为什么值得系统干预？",
        20,
    ),
    Topic(
        "neurology",
        "WHO_Intersectoral_Action_Plan_Epilepsy_Neurological_Disorders.pdf",
        "treatment gaps exceed",
        "中低收入国家的癫痫治疗缺口有多大？",
        "资源有限地区许多癫痫患者得不到治疗，行动计划给出了怎样的缺口规模？",
        7,
    ),
    Topic(
        "neurology",
        "WHO_Intersectoral_Action_Plan_Epilepsy_Neurological_Disorders.pdf",
        "stigma and discrimination",
        "污名和歧视如何影响神经系统疾病患者？",
        "神经系统疾病患者除症状外，还可能因社会偏见遭遇哪些生活和就医障碍？",
        7,
    ),
    Topic(
        "neurology",
        "WHO_Intersectoral_Action_Plan_Epilepsy_Neurological_Disorders.pdf",
        "rehabilitation",
        "神经系统疾病连续照护为什么需要康复服务？",
        "从预防到姑息的完整神经照护链中，功能恢复服务处于什么位置？",
        10,
    ),
    Topic(
        "neurology",
        "WHO_Intersectoral_Action_Plan_Epilepsy_Neurological_Disorders.pdf",
        "universal health coverage",
        "全民健康覆盖与神经系统疾病行动计划有什么关系？",
        "要让神经疾病患者获得可负担服务，行动计划如何衔接 UHC？",
        10,
    ),
    Topic(
        "neurology",
        "WHO_Intersectoral_Action_Plan_Epilepsy_Neurological_Disorders.pdf",
        "caregivers",
        "神经系统疾病照护者需要哪些支持？",
        "长期照护神经疾病患者的家属，为什么也需要技能培训和社会支持？",
        30,
    ),
    Topic(
        "pediatrics",
        "WHO_Pocket_Book_Hospital_Care_for_Children.pdf",
        "severe pneumonia",
        "儿童重症肺炎应关注哪些临床处置要点？",
        "患儿咳嗽并出现危险征象时，住院手册如何指导识别和处理严重下呼吸道感染？",
        35,
    ),
    Topic(
        "pediatrics",
        "WHO_Pocket_Book_Hospital_Care_for_Children.pdf",
        "severe dehydration",
        "儿童严重脱水如何识别和处理？",
        "腹泻患儿精神差且循环受影响时，手册如何判断液体丢失程度并补液？",
        120,
    ),
    Topic(
        "pediatrics",
        "WHO_Pocket_Book_Hospital_Care_for_Children.pdf",
        "severe acute malnutrition",
        "儿童重度急性营养不良的住院管理重点是什么？",
        "消瘦或营养性水肿患儿住院后，治疗流程需要优先覆盖哪些问题？",
        170,
    ),
    Topic(
        "pediatrics",
        "WHO_Pocket_Book_Hospital_Care_for_Children.pdf",
        "severe malaria",
        "儿童重症疟疾需要怎样的紧急处理？",
        "疟疾患儿出现危重表现时，住院手册推荐如何快速评估和治疗？",
        135,
    ),
    Topic(
        "pediatrics",
        "WHO_Pocket_Book_Hospital_Care_for_Children.pdf",
        "bacterial meningitis",
        "儿童细菌性脑膜炎如何诊疗？",
        "患儿发热、颈强直或意识异常时，手册对中枢神经感染有哪些诊疗要求？",
        140,
    ),
    Topic(
        "infectious_disease",
        "WHO_Chronic_Hepatitis_B_Guidelines_2024.pdf",
        "tenofovir disoproxil fumarate",
        "乙肝孕妇何时考虑使用 TDF 预防母婴传播？",
        "HBsAg 阳性的孕妇在什么病毒学条件下需要用替诺福韦降低垂直传播？",
        30,
    ),
    Topic(
        "infectious_disease",
        "WHO_Chronic_Hepatitis_B_Guidelines_2024.pdf",
        "entecavir",
        "慢性乙肝治疗中恩替卡韦处于什么位置？",
        "需要长期抑制 HBV 的患者，ETV 属于哪类推荐抗病毒选择？",
        30,
    ),
    Topic(
        "infectious_disease",
        "WHO_Chronic_Hepatitis_B_Guidelines_2024.pdf",
        "cirrhosis",
        "慢性乙肝合并肝硬化时为什么要重视抗病毒治疗？",
        "HBV 患者已经出现进展性肝纤维化时，治疗决策为何更积极？",
        30,
    ),
    Topic(
        "infectious_disease",
        "WHO_Chronic_Hepatitis_B_Guidelines_2024.pdf",
        "hepatocellular carcinoma",
        "慢性乙肝患者为什么需要肝细胞癌监测？",
        "长期 HBV 感染者即使接受管理，为何仍需定期排查 HCC？",
        120,
    ),
    Topic(
        "infectious_disease",
        "WHO_Managing_Epidemics_2nd_Edition.pdf",
        "cholera",
        "霍乱暴发管理需要掌握哪些核心事实？",
        "急性水样腹泻在社区快速增加时，公共卫生响应应依据哪些霍乱要点？",
        50,
    ),
    Topic(
        "infectious_disease",
        "WHO_Managing_Epidemics_2nd_Edition.pdf",
        "dengue",
        "登革热流行时应关注哪些传播和防控特点？",
        "蚊媒病毒病病例在雨季上升时，疫情管理需要抓住哪些关键点？",
        240,
    ),
    Topic(
        "general_surgery",
        "EAU_Urolithiasis_Guideline_2026.pdf",
        "non-contrast-enhanced computed tomography",
        "尿路结石诊断中无增强 CT 有什么作用？",
        "怀疑泌尿系结石时，哪种不使用造影剂的断层检查可判断密度、结构和成分？",
        10,
    ),
    Topic(
        "general_surgery",
        "EAU_Urolithiasis_Guideline_2026.pdf",
        "sepsis and/or anuria",
        "梗阻性结石合并脓毒症或无尿时如何处理？",
        "结石堵塞尿路并出现感染性休克风险或无尿时，指南强调什么紧急处置？",
        15,
    ),
    Topic(
        "general_surgery",
        "EAU_Urolithiasis_Guideline_2026.pdf",
        "medical expulsive therapy",
        "哪些尿路结石患者可考虑药物排石治疗？",
        "输尿管结石希望自行排出时，何种患者适合采用 MET 策略？",
        15,
    ),
    Topic(
        "general_surgery",
        "EAU_Urolithiasis_Guideline_2026.pdf",
        "fluid intake",
        "预防尿路结石复发时饮水有什么意义？",
        "有结石史的人进行通用复发预防时，液体摄入应处于什么核心位置？",
        45,
    ),
    Topic(
        "general_surgery",
        "WHO_Chronic_Primary_Low_Back_Pain_Guideline_2023.pdf",
        "structured exercise",
        "慢性原发性腰痛为什么推荐结构化运动？",
        "长期非特异性腰背痛患者接受有计划的锻炼项目，指南如何评价这种干预？",
        40,
    ),
    Topic(
        "general_surgery",
        "WHO_Chronic_Primary_Low_Back_Pain_Guideline_2023.pdf",
        "education and/or advice",
        "慢性原发性腰痛管理中的教育与建议有什么作用？",
        "对于长期腰痛，为什么需要向患者提供结构化知识和自我管理建议？",
        35,
    ),
    Topic(
        "general_surgery",
        "WHO_Chronic_Primary_Low_Back_Pain_Guideline_2023.pdf",
        "opioid analgesics",
        "慢性原发性腰痛使用阿片类镇痛药需要怎样权衡？",
        "长期腰痛若考虑强效中枢镇痛药，指南为何要求谨慎评估获益与伤害？",
        100,
    ),
    Topic(
        "general_surgery",
        "WHO_Surgical_Safety_Checklist_Implementation_Manual.pdf",
        "antibiotic prophylaxis",
        "手术安全核查为何确认预防性抗生素给药时间？",
        "切皮前团队为什么要核实抗菌预防药是否已在规定时间窗内使用？",
        6,
    ),
    Topic(
        "general_medical",
        "WHO_Antenatal_Care_Positive_Pregnancy_Experience.pdf",
        "iron and folic acid supplementation",
        "孕期每日铁和叶酸补充的推荐剂量与目的是什么？",
        "为减少孕妇贫血、低出生体重和早产，日常补充哪两种营养素及多少剂量？",
        12,
    ),
    Topic(
        "general_medical",
        "WHO_Antenatal_Care_Positive_Pregnancy_Experience.pdf",
        "calcium supplementation",
        "低钙摄入人群孕期补钙的推荐是什么？",
        "膳食钙不足的孕妇为降低子痫前期风险，应如何补充元素钙？",
        12,
    ),
    Topic(
        "general_medical",
        "WHO_Antenatal_Care_Positive_Pregnancy_Experience.pdf",
        "minimum of eight contacts",
        "WHO 为什么建议至少八次产前保健接触？",
        "孕期随访不只做四次时，增加到怎样的最低接触次数可改善围产结局和体验？",
        16,
    ),
    Topic(
        "general_medical",
        "KDIGO_2024_CKD_Clinical_Practice_Guideline.pdf",
        "albuminuria",
        "慢性肾病评估为什么要关注白蛋白尿？",
        "评估 CKD 风险时，除 eGFR 外为什么还要量化尿蛋白中的白蛋白？",
        20,
    ),
    Topic(
        "general_medical",
        "KDIGO_2024_CKD_Clinical_Practice_Guideline.pdf",
        "SGLT2 inhibitor",
        "SGLT2 抑制剂在慢性肾病管理中有什么作用？",
        "部分 CKD 患者即使关注点不只是降糖，为什么会考虑钠-葡萄糖协同转运蛋白 2 抑制药？",
        80,
    ),
    Topic(
        "general_medical",
        "WHO_Guide_to_Cancer_Early_Diagnosis.pdf",
        "screening that seeks",
        "癌症早期诊断与筛查有什么区别？",
        "针对已经出现症状的人尽早识别肿瘤，与面向无症状目标人群的检测有何不同？",
        7,
    ),
    Topic(
        "general_medical",
        "WHO_HEARTS_Cardiovascular_Disease_Management.pdf",
        "high blood pressure",
        "高血压为什么是心血管疾病管理的重要危险因素？",
        "持续血压升高如何与心脏、脑和肾脏并发症风险联系起来？",
        10,
    ),
    Topic(
        "general_medical",
        "WHO_PEN_Essential_NCD_Interventions.pdf",
        "cardiovascular disease risk assessment",
        "基层医疗为什么要评估总体心血管风险？",
        "管理高血压和糖尿病时，为什么不能只看一个单项指标而要估计综合 CVD 风险？",
        10,
    ),
    Topic(
        "general_medical",
        "WHO_Sickle_Cell_Disease_Pregnancy_Childbirth_2025.pdf",
        "up to 5 mg folic acid",
        "镰状细胞病孕妇的叶酸补充建议是什么？",
        "患 SCD 的孕妇在非疟疾流行地区，叶酸日剂量可提高到多少？",
        14,
    ),
    Topic(
        "general_medical",
        "WHO_Sickle_Cell_Disease_Pregnancy_Childbirth_2025.pdf",
        "hydroxycarbamide",
        "镰状细胞病孕妇如何权衡羟基脲治疗？",
        "孕前依赖 hydroxyurea 控制 SCD 的女性，孕期是否继续用药应如何共同决策？",
        14,
    ),
    Topic(
        "general_medical",
        "WHO_Basic_Emergency_Care_Participant_Workbook.pdf",
        "ABCDE approach",
        "急诊初始评估中的 ABCDE 方法包含什么思路？",
        "面对病因未明的危重患者，为什么要按气道、呼吸、循环等固定顺序先处理致命问题？",
        10,
    ),
)

# RAG evidence is extracted from English PDFs, so retrieval queries use English
# as well. Keeping these translations in source makes dataset generation fully
# reproducible and avoids measuring cross-language translation as retrieval gain.
ENGLISH_QUERIES = (
    (
        "What is the target sensitivity for confirmatory scabies testing?",
        "When confirming scabies after initial screening, what detection capability target should avoid missed cases?",
    ),
    (
        "What is the target specificity for confirmatory scabies testing?",
        "For a tool used to confirm scabies after screening, what exclusion capability target should it meet?",
    ),
    (
        "What is the diagnostic product used for when initiating mass drug administration for scabies?",
        "Before a community starts mass scabies treatment, what role does the point-of-care test play?",
    ),
    (
        "What is the diagnostic product used for when stopping mass drug administration for scabies?",
        "After multiple rounds of mass scabies treatment, what testing purpose supports the stop decision?",
    ),
    (
        "What product requirements does the WHO target product profile for scabies diagnostics focus on?",
        "What target specifications should guide development of point-of-care scabies diagnostics?",
    ),
    (
        "Why does safe listening prevent noise-induced hearing loss?",
        "The more sound exposure the cochlea receives, the greater the damage—what does this mean for daily listening habits?",
    ),
    (
        "What role does newborn hearing screening play in hearing care?",
        "When infants cannot express hearing problems, how can the health system detect issues early?",
    ),
    (
        "What common interventions do hearing assistive technologies include?",
        "After hearing loss, what devices and communication supports are available besides medication or surgery?",
    ),
    (
        "How can cochlear implants help people with hearing loss?",
        "What type of implantable device can improve education and communication opportunities for those with severe hearing loss?",
    ),
    (
        "Why is otitis media an important issue in ear and hearing care?",
        "How does this common childhood middle ear condition fit into comprehensive hearing care?",
    ),
    (
        "How does uncorrected refractive error affect vision?",
        "What visual burden does focusing error create when not corrected by glasses or similar?",
    ),
    (
        "How does cataract affect vision?",
        "When lens opacity reduces sight, how does the report describe this eye disease burden?",
    ),
    (
        "Why does glaucoma require early detection and ongoing management?",
        "An eye disease that gradually damages the optic nerve—why cannot it wait until obvious vision loss appears?",
    ),
    (
        "What is the relationship between diabetic retinopathy and vision impairment?",
        "When long-term high blood sugar affects retinal microvessels, what visual risk does the report emphasize?",
    ),
    (
        "Why does myopia require public health attention?",
        "A refractive problem causing blurred distance vision and growing prevalence—why is systematic intervention worthwhile?",
    ),
    (
        "How large is the epilepsy treatment gap in low- and middle-income countries?",
        "In resource-limited settings, many epilepsy patients receive no treatment—what scale of gap does the action plan identify?",
    ),
    (
        "How do stigma and discrimination affect people with neurological disorders?",
        "Beyond symptoms, what barriers in daily life and healthcare do patients face due to social prejudice?",
    ),
    (
        "Why is rehabilitation needed in continuous care for neurological disorders?",
        "In the full neurological care chain from prevention to palliative care, where does functional recovery fit?",
    ),
    (
        "What is the relationship between universal health coverage and the neurological disorders action plan?",
        "How does the action plan connect with UHC to make services affordable for patients?",
    ),
    (
        "What support do caregivers of people with neurological disorders need?",
        "Why do family members providing long-term care also need skills training and social support?",
    ),
    (
        "What are the key clinical management points for severe pneumonia in children?",
        "When a child has cough and danger signs, how does the inpatient handbook guide recognition and management of severe lower respiratory tract infection?",
    ),
    (
        "How is severe dehydration in children identified and treated?",
        "For a child with diarrhea, lethargy, and circulatory compromise, how does the handbook assess fluid loss and provide rehydration?",
    ),
    (
        "What are the priorities for inpatient management of severe acute malnutrition in children?",
        "After admission for wasting or nutritional edema, which issues should the treatment protocol prioritize?",
    ),
    (
        "What emergency management is required for severe malaria in children?",
        "When a child with malaria presents with critical manifestations, how does the handbook recommend rapid assessment and treatment?",
    ),
    (
        "How is pediatric bacterial meningitis diagnosed and treated?",
        "When a child has fever, neck stiffness, or altered consciousness, what diagnostic and therapeutic requirements does the handbook have for central nervous system infection?",
    ),
    (
        "When should pregnant women with hepatitis B consider TDF to prevent mother-to-child transmission?",
        "Under what virological conditions should HBsAg-positive pregnant women use tenofovir to reduce vertical transmission?",
    ),
    (
        "What is the role of entecavir in chronic hepatitis B treatment?",
        "For patients needing long-term HBV suppression, what class of recommended antiviral option is ETV?",
    ),
    (
        "Why is antiviral therapy important in chronic hepatitis B with cirrhosis?",
        "Why are treatment decisions more aggressive when HBV patients have advanced liver fibrosis?",
    ),
    (
        "Why do patients with chronic hepatitis B need hepatocellular carcinoma surveillance?",
        "Why do people with long-term HBV infection still need regular HCC surveillance even while receiving care?",
    ),
    (
        "What core facts are needed for cholera outbreak management?",
        "When acute watery diarrhea increases rapidly in a community, what key cholera facts should the public health response rely on?",
    ),
    (
        "During a dengue epidemic, what transmission and control features matter?",
        "When mosquito-borne viral cases rise in the rainy season, what key points should outbreak management focus on?",
    ),
    (
        "What is the role of unenhanced CT in diagnosing urinary stones?",
        "For suspected urinary stones, which non-contrast tomographic examination can assess density, structure, and composition?",
    ),
    (
        "How should obstructive stones with sepsis or anuria be managed?",
        "When stones obstruct the urinary tract with a risk of septic shock or anuria, what emergency action do guidelines emphasize?",
    ),
    (
        "Which urinary stone patients may consider medical expulsive therapy?",
        "For ureteral stones expected to pass spontaneously, which patients are suitable for a MET strategy?",
    ),
    (
        "What is the role of water intake in preventing urinary stone recurrence?",
        "In general recurrence prevention for people with a history of stones, what central role does fluid intake have?",
    ),
    (
        "Why is structured exercise recommended for chronic primary low back pain?",
        "For patients with long-term nonspecific low back pain receiving a planned exercise program, how do guidelines evaluate this intervention?",
    ),
    (
        "What is the role of education and advice in managing chronic primary low back pain?",
        "For long-term low back pain, why is it necessary to provide structured knowledge and self-management advice?",
    ),
    (
        "How should the use of opioid analgesics for chronic primary low back pain be weighed?",
        "If strong centrally acting analgesics are considered for long-term low back pain, why do guidelines require careful assessment of benefits and harms?",
    ),
    (
        "Why does the surgical safety checklist confirm the timing of prophylactic antibiotic administration?",
        "Why should the team verify before skin incision whether antimicrobial prophylaxis was given within the specified time window?",
    ),
    (
        "What are the recommended doses and purposes of daily iron and folic acid supplementation during pregnancy?",
        "To reduce maternal anemia, low birth weight, and preterm birth, which two nutrients should be supplemented daily and at what doses?",
    ),
    (
        "What calcium supplementation is recommended during pregnancy for women with low calcium intake?",
        "How should pregnant women with inadequate dietary calcium supplement elemental calcium to reduce the risk of pre-eclampsia?",
    ),
    (
        "Why does WHO recommend at least eight antenatal care contacts?",
        "What minimum number of antenatal contacts beyond the traditional four can improve perinatal outcomes and care experience?",
    ),
    (
        "Why is albuminuria important in the evaluation of chronic kidney disease?",
        "When assessing CKD risk, why should urinary albumin be quantified in addition to eGFR?",
    ),
    (
        "What is the role of SGLT2 inhibitors in chronic kidney disease management?",
        "Why consider sodium-glucose cotransporter 2 inhibitors in some CKD patients even when glucose lowering is not the only goal?",
    ),
    (
        "What is the difference between early diagnosis and screening for cancer?",
        "How does early identification of tumors in symptomatic people differ from testing asymptomatic target populations?",
    ),
    (
        "Why is hypertension an important risk factor in cardiovascular disease management?",
        "How does sustained high blood pressure relate to the risks of heart, brain, and kidney complications?",
    ),
    (
        "Why should primary care assess overall cardiovascular risk?",
        "When managing hypertension and diabetes, why estimate combined CVD risk instead of focusing on one indicator?",
    ),
    (
        "What folic acid supplementation is recommended for pregnant women with sickle cell disease?",
        "In non-malaria-endemic areas, to what daily folic acid dose can pregnant women with SCD increase?",
    ),
    (
        "How should pregnant women with sickle cell disease weigh hydroxyurea treatment?",
        "For women relying on hydroxyurea to control SCD before pregnancy, how should a decision about continuing it during pregnancy be shared?",
    ),
    (
        "What is the rationale behind the ABCDE approach in initial emergency assessment?",
        "For critically ill patients with an unknown cause, why address life-threatening issues in a fixed order such as airway, breathing, and circulation?",
    ),
)


ROUTING_MEDICAL_QUESTIONS = {
    "general_medical": [
        "高血压长期控制需要关注哪些指标？",
        "慢性肾病患者为什么要检查尿白蛋白？",
        "孕期铁和叶酸应该怎样补充？",
        "2026 年最新高血压治疗指南有哪些变化？",
        "请联网查一下本周 WHO 发布的慢性病新建议。",
    ],
    "general_surgery": [
        "肾结石复发应该怎样预防？",
        "外伤后伤口持续出血要如何处理？",
        "慢性腰痛适合做哪些运动？",
        "2026 年最新尿路结石指南推荐什么治疗？",
        "请查最新手术安全核查规范有没有更新。",
    ],
    "pediatrics": [
        "儿童肺炎出现哪些表现需要住院？",
        "宝宝腹泻后怎样判断是否严重脱水？",
        "儿童重度营养不良如何处理？",
        "2026 年儿童肺炎最新治疗建议是什么？",
        "儿童细菌性脑膜炎有哪些典型表现？",
    ],
    "neurology": [
        "癫痫患者的治疗缺口是什么意思？",
        "抽搐后意识不清应该看哪个科？",
        "脑卒中康复为什么要尽早开始？",
        "2026 年最新癫痫用药指南有什么变化？",
        "神经系统疾病患者的照护者需要哪些支持？",
    ],
    "infectious_disease": [
        "慢性乙肝患者为什么要监测肝癌？",
        "登革热主要通过什么途径传播？",
        "霍乱暴发时应怎样进行公共卫生处置？",
        "2026 年最新乙肝抗病毒指南是什么？",
        "乙肝孕妇如何降低母婴传播风险？",
    ],
    "ent": [
        "耳鸣伴听力下降应该看哪个科？",
        "新生儿为什么要做听力筛查？",
        "长时间戴耳机怎样保护听力？",
        "2026 年最新人工耳蜗适应证是什么？",
        "中耳炎为什么可能影响儿童听力？",
    ],
    "ophthalmology": [
        "糖尿病患者为什么要定期检查眼底？",
        "视力下降和白内障有什么关系？",
        "青光眼为什么需要长期随访？",
        "2026 年近视防控最新指南是什么？",
        "未矫正屈光不正会造成什么影响？",
    ],
    "dermatology": [
        "全身瘙痒并出现皮疹应该看哪个科？",
        "疥疮诊断通常关注哪些检查性能？",
        "湿疹反复发作应该怎样护理？",
        "2026 年最新疥疮诊疗指南有哪些变化？",
        "疥疮检测为什么要同时关注敏感度和特异度？",
    ],
}

NON_MEDICAL_QUESTIONS = (
    "请帮我写一段求职自我介绍。",
    "Python 字典怎样按值排序？",
    "北京到上海坐高铁要多久？",
    "给我推荐三部科幻电影。",
    "如何学习线性代数？",
    "帮我设计一个旅行计划。",
    "这段 JavaScript 为什么报错？",
    "解释一下什么是通货膨胀。",
    "写一首关于秋天的短诗。",
    "明天开会的议程怎么安排？",
)

REDIS_POSITIVE_PAIRS = (
    (
        "高血压患者可以喝咖啡吗？",
        "血压高的人能不能喝咖啡？",
        "多数人可少量饮用，并观察血压反应。",
    ),
    (
        "糖尿病患者如何控制饮食？",
        "血糖高的人平时吃饭要注意什么？",
        "控制总能量，优先全谷物、蔬菜和优质蛋白。",
    ),
    (
        "感冒后一直咳嗽怎么办？",
        "上呼吸道感染后咳个不停该怎么处理？",
        "可先补液和观察，持续或加重时就医评估。",
    ),
    (
        "儿童发烧什么时候需要就医？",
        "孩子发热到什么情况应去医院？",
        "精神反应差、呼吸困难或持续高热时应及时就医。",
    ),
    (
        "慢性肾病为什么要查尿蛋白？",
        "CKD 患者检测白蛋白尿有什么用？",
        "白蛋白尿有助于评估肾损伤和进展风险。",
    ),
    (
        "糖尿病患者为什么要查眼底？",
        "血糖高的人定期做视网膜检查有什么意义？",
        "眼底检查有助于早期发现糖尿病视网膜病变。",
    ),
    (
        "近视如何保护视力？",
        "眼睛近视后日常怎样避免进一步加深？",
        "增加户外活动并减少连续近距离用眼。",
    ),
    (
        "耳鸣伴听力下降怎么办？",
        "耳朵响而且听不清应该如何处理？",
        "需要耳鼻喉科评估耳道、中耳和听力。",
    ),
    (
        "肾结石患者应该多喝水吗？",
        "泌尿系结石增加饮水量有帮助吗？",
        "充足饮水有助于增加尿量并降低复发风险。",
    ),
    (
        "慢性腰痛适合运动吗？",
        "长期下腰背疼能不能进行锻炼？",
        "多数患者可在评估后进行循序渐进的结构化运动。",
    ),
    (
        "乙肝患者为什么要监测肝癌？",
        "慢性 HBV 感染为何要定期筛查 HCC？",
        "慢性乙肝会增加肝细胞癌风险，需要风险分层监测。",
    ),
    (
        "孕期为什么要补叶酸？",
        "怀孕后吃叶酸有什么作用？",
        "叶酸有助于预防胎儿神经管缺陷并支持孕期造血。",
    ),
    (
        "儿童腹泻怎样判断脱水？",
        "宝宝拉肚子后怎么知道有没有缺水？",
        "可观察精神、口渴、眼窝、皮肤弹性和尿量。",
    ),
    (
        "青光眼为什么要长期随访？",
        "得了青光眼为何需要持续复查？",
        "持续监测眼压和视神经有助于减慢不可逆视损伤。",
    ),
    (
        "白内障会导致视力下降吗？",
        "晶状体混浊是不是会看不清？",
        "白内障可使进入眼内的光线受阻并降低视力。",
    ),
    (
        "安全使用耳机要注意什么？",
        "戴耳机怎样减少噪声性听力损伤？",
        "控制音量和时长，并在嘈杂环境中避免继续加大音量。",
    ),
    (
        "疥疮是怎样传播的？",
        "疥螨感染通常通过什么方式传给别人？",
        "疥疮主要通过较长时间的密切皮肤接触传播。",
    ),
    (
        "癫痫患者为什么会受到污名影响？",
        "社会偏见会给癫痫病人带来什么问题？",
        "污名可影响教育、就业、社会关系和就医。",
    ),
    (
        "总体心血管风险评估有什么用？",
        "为什么心血管管理不能只看血压一个指标？",
        "综合评估可结合多种危险因素决定干预强度。",
    ),
    (
        "癌症早期诊断和筛查一样吗？",
        "肿瘤早诊是不是就等于给无症状人群做筛查？",
        "早期诊断针对有症状者，筛查针对无症状目标人群。",
    ),
    (
        "孕期低钙人群需要补钙吗？",
        "饮食钙不足的孕妇是否应额外补充钙？",
        "低膳食钙人群补钙可帮助降低子痫前期风险。",
    ),
    (
        "手术前为什么要核查患者身份？",
        "进手术室后为何还要再次确认姓名和术式？",
        "重复核查可降低错误患者、错误部位和错误术式风险。",
    ),
    (
        "新生儿为什么要做听力筛查？",
        "婴儿出生后检查听觉有什么必要？",
        "早期筛查有助于尽早识别听力损失并开展干预。",
    ),
    (
        "慢性呼吸病为什么要戒烟？",
        "哮喘或 COPD 患者停止吸烟有什么意义？",
        "戒烟可减少持续气道损伤和疾病恶化风险。",
    ),
    (
        "急诊为什么采用 ABCDE 顺序？",
        "危重患者为何先按气道呼吸循环依次检查？",
        "固定顺序有助于优先发现和处理立即威胁生命的问题。",
    ),
)

REDIS_NEGATIVE_PAIRS = (
    (
        "高血压患者可以喝咖啡吗？",
        "低血压患者可以喝咖啡吗？",
        "多数人可少量饮用，并观察血压反应。",
    ),
    (
        "糖尿病患者如何控制饮食？",
        "妊娠期糖尿病患者如何控制饮食？",
        "控制总能量，优先全谷物、蔬菜和优质蛋白。",
    ),
    (
        "成人感冒后一直咳嗽怎么办？",
        "三个月婴儿感冒后一直咳嗽怎么办？",
        "可先补液和观察，持续或加重时就医评估。",
    ),
    (
        "儿童发烧什么时候需要就医？",
        "成年人低热什么时候需要就医？",
        "精神反应差、呼吸困难或持续高热时应及时就医。",
    ),
    (
        "慢性肾病为什么要查尿蛋白？",
        "尿路感染为什么要查尿培养？",
        "白蛋白尿有助于评估肾损伤和进展风险。",
    ),
    (
        "糖尿病患者为什么要查眼底？",
        "高血压患者为什么要查眼底？",
        "眼底检查有助于早期发现糖尿病视网膜病变。",
    ),
    (
        "近视如何保护视力？",
        "青光眼如何保护视力？",
        "增加户外活动并减少连续近距离用眼。",
    ),
    (
        "耳鸣伴听力下降怎么办？",
        "耳鸣伴剧烈眩晕怎么办？",
        "需要耳鼻喉科评估耳道、中耳和听力。",
    ),
    (
        "肾结石患者应该多喝水吗？",
        "心力衰竭患者应该多喝水吗？",
        "充足饮水有助于增加尿量并降低复发风险。",
    ),
    (
        "慢性腰痛适合运动吗？",
        "急性腰椎骨折适合运动吗？",
        "多数患者可在评估后进行循序渐进的结构化运动。",
    ),
    (
        "乙肝患者为什么要监测肝癌？",
        "丙肝患者为什么要监测肝癌？",
        "慢性乙肝会增加肝细胞癌风险，需要风险分层监测。",
    ),
    (
        "孕期为什么要补叶酸？",
        "备孕男性为什么要补叶酸？",
        "叶酸有助于预防胎儿神经管缺陷并支持孕期造血。",
    ),
    (
        "儿童腹泻怎样判断脱水？",
        "儿童呕吐怎样判断脑膜炎？",
        "可观察精神、口渴、眼窝、皮肤弹性和尿量。",
    ),
    (
        "青光眼为什么要长期随访？",
        "白内障为什么要长期随访？",
        "持续监测眼压和视神经有助于减慢不可逆视损伤。",
    ),
    (
        "白内障会导致视力下降吗？",
        "干眼症会导致视力下降吗？",
        "白内障可使进入眼内的光线受阻并降低视力。",
    ),
    (
        "安全使用耳机要注意什么？",
        "使用助听器要注意什么？",
        "控制音量和时长，并在嘈杂环境中避免继续加大音量。",
    ),
    (
        "疥疮是怎样传播的？",
        "湿疹是怎样传播的？",
        "疥疮主要通过较长时间的密切皮肤接触传播。",
    ),
    (
        "癫痫患者为什么会受到污名影响？",
        "偏头痛患者为什么会受到污名影响？",
        "污名可影响教育、就业、社会关系和就医。",
    ),
    (
        "总体心血管风险评估有什么用？",
        "单次心电图检查有什么用？",
        "综合评估可结合多种危险因素决定干预强度。",
    ),
    (
        "癌症早期诊断和筛查一样吗？",
        "传染病早期诊断和筛查一样吗？",
        "早期诊断针对有症状者，筛查针对无症状目标人群。",
    ),
    (
        "孕期低钙人群需要补钙吗？",
        "肾结石孕妇需要补钙吗？",
        "低膳食钙人群补钙可帮助降低子痫前期风险。",
    ),
    (
        "手术前为什么要核查患者身份？",
        "门诊取药前为什么要核查患者身份？",
        "重复核查可降低错误患者、错误部位和错误术式风险。",
    ),
    (
        "新生儿为什么要做听力筛查？",
        "成年人为什么要做听力筛查？",
        "早期筛查有助于尽早识别听力损失并开展干预。",
    ),
    (
        "慢性呼吸病为什么要戒烟？",
        "慢性肾病为什么要戒烟？",
        "戒烟可减少持续气道损伤和疾病恶化风险。",
    ),
    (
        "急诊为什么采用 ABCDE 顺序？",
        "常规体检为什么采用 ABCDE 顺序？",
        "固定顺序有助于优先发现和处理立即威胁生命的问题。",
    ),
)


def _find_source(filename: str) -> Path:
    matches = list(KNOWLEDGE_ROOT.rglob(filename))
    if len(matches) != 1:
        raise ValueError(f"Expected one PDF named {filename}, found {len(matches)}")
    return matches[0]


def _page_texts(path: Path, cache: dict[Path, list[str]]) -> list[str]:
    if path not in cache:
        cache[path] = [
            " ".join((page.extract_text() or "").split())
            for page in PdfReader(path).pages
        ]
    return cache[path]


def _extract_evidence(
    topic: Topic,
    cache: dict[Path, list[str]],
    *,
    gold_page: int,
) -> dict[str, Any]:
    path = _find_source(topic.source)
    pages = _page_texts(path, cache)
    if not 1 <= gold_page <= len(pages):
        raise ValueError(f"Invalid gold page {gold_page}: {topic.source}")
    text = pages[gold_page - 1].strip()
    if not text:
        raise ValueError(f"Empty gold page {gold_page}: {topic.source}")
    return {"source": topic.source, "page": gold_page, "text": text}


def _build_rag_dataset() -> dict[str, Any]:
    cache: dict[Path, list[str]] = {}
    bases = []
    if len(GOLD_PAGES) != len(TOPICS):
        raise ValueError("GOLD_PAGES must align one-to-one with TOPICS")
    if len(ENGLISH_QUERIES) != len(TOPICS):
        raise ValueError("ENGLISH_QUERIES must align one-to-one with TOPICS")
    for index, topic in enumerate(TOPICS, start=1):
        evidence = _extract_evidence(topic, cache, gold_page=GOLD_PAGES[index - 1])
        question_en, hard_question_en = ENGLISH_QUERIES[index - 1]
        bases.append((topic, evidence, index, question_en, hard_question_en))

    samples = []
    for topic, evidence, index, question_en, _ in bases:
        samples.append(
            {
                "id": f"rag_single_{index:03d}",
                "category": "single_hop",
                "department": topic.department,
                "language": "en",
                "question": question_en,
                "gold_evidence": [evidence],
                "reference_answer": evidence["text"],
            }
        )

    by_department: dict[
        str, list[tuple[Topic, dict[str, Any], int, str, str]]
    ] = defaultdict(list)
    for base in bases:
        by_department[base[0].department].append(base)
    for index, base in enumerate(bases, start=1):
        topic, evidence, _, question_en, _ = base
        candidates = by_department[topic.department]
        position = candidates.index(base)
        partner = candidates[(position + 1) % len(candidates)]
        if (partner[1]["source"], partner[1]["page"]) == (
            evidence["source"],
            evidence["page"],
        ):
            partner = candidates[(position + 2) % len(candidates)]
        _, partner_evidence, _, partner_question_en, _ = partner
        samples.append(
            {
                "id": f"rag_multi_{index:03d}",
                "category": "multi_hop",
                "department": topic.department,
                "language": "en",
                "question": (
                    "Use both pieces of evidence to answer: "
                    f"{question_en} Also, {partner_question_en}"
                ),
                "gold_evidence": [evidence, partner_evidence],
                "reference_answer": f"{evidence['text']}\n\n{partner_evidence['text']}",
            }
        )

    for topic, evidence, index, _, hard_question_en in bases:
        samples.append(
            {
                "id": f"rag_hard_{index:03d}",
                "category": "hard_retrieval",
                "department": topic.department,
                "language": "en",
                "question": hard_question_en,
                "gold_evidence": [evidence],
                "reference_answer": evidence["text"],
            }
        )

    return {
        "metadata": {
            "name": "medagent-rag-150-v1",
            "version": "1.0",
            "total": 150,
            "split": {"single_hop": 50, "multi_hop": 50, "hard_retrieval": 50},
            "departments": list(DEPARTMENTS),
            "evidence_source": "local_pdf_page_text",
            "query_language": "en",
        },
        "samples": samples,
    }


def _build_routing_dataset() -> dict[str, Any]:
    samples = []
    index = 1
    for department, questions in ROUTING_MEDICAL_QUESTIONS.items():
        for question in questions:
            wants_web = "最新" in question or "联网" in question or "本周" in question
            samples.append(
                {
                    "id": f"route_{index:03d}",
                    "question": question,
                    "expected_route": "web_search" if wants_web else "local_rag",
                    "expected_department": department,
                }
            )
            index += 1
    for question in NON_MEDICAL_QUESTIONS:
        samples.append(
            {
                "id": f"route_{index:03d}",
                "question": question,
                "expected_route": "non_medical",
                "expected_department": None,
            }
        )
        index += 1
    return {
        "metadata": {
            "name": "medagent-routing-50-v1",
            "version": "1.0",
            "total": 50,
            "split": {"local_rag": 30, "web_search": 10, "non_medical": 10},
            "medical_per_department": 5,
            "departments": list(DEPARTMENTS),
        },
        "samples": samples,
    }


def _build_redis_dataset() -> dict[str, Any]:
    samples = []
    for index, (cached, probe, answer) in enumerate(REDIS_POSITIVE_PAIRS, start=1):
        samples.append(
            {
                "id": f"cache_{index:03d}",
                "category": "semantic_equivalent",
                "cached_question": cached,
                "cached_answer": answer,
                "probe_question": probe,
                "expected_hit": True,
            }
        )
    for index, (cached, probe, answer) in enumerate(REDIS_NEGATIVE_PAIRS, start=26):
        samples.append(
            {
                "id": f"cache_{index:03d}",
                "category": "non_reusable",
                "cached_question": cached,
                "cached_answer": answer,
                "probe_question": probe,
                "expected_hit": False,
            }
        )
    return {
        "metadata": {
            "name": "medagent-redis-cache-50-v1",
            "version": "1.0",
            "total": 50,
            "unit": "cached_question_probe_question_pair",
            "split": {"semantic_equivalent": 25, "non_reusable": 25},
        },
        "samples": samples,
    }


def _validate_rag(dataset: dict[str, Any]) -> None:
    samples = dataset["samples"]
    assert len(samples) == 150
    assert Counter(row["category"] for row in samples) == {
        "single_hop": 50,
        "multi_hop": 50,
        "hard_retrieval": 50,
    }
    assert {row["department"] for row in samples} == set(DEPARTMENTS)
    for row in samples:
        assert row["question"] and row["reference_answer"]
        assert row["language"] == "en"
        assert not any("\u4e00" <= char <= "\u9fff" for char in row["question"])
        assert len(row["gold_evidence"]) == (2 if row["category"] == "multi_hop" else 1)
        for evidence in row["gold_evidence"]:
            assert evidence["source"] and evidence["page"] > 0 and evidence["text"]
            assert _find_source(evidence["source"]).exists()


def _validate_routing(dataset: dict[str, Any]) -> None:
    samples = dataset["samples"]
    assert len(samples) == 50
    assert Counter(row["expected_route"] for row in samples) == {
        "local_rag": 30,
        "web_search": 10,
        "non_medical": 10,
    }
    medical = [row for row in samples if row["expected_route"] != "non_medical"]
    assert Counter(row["expected_department"] for row in medical) == {
        department: 5 for department in DEPARTMENTS
    }


def _validate_redis(dataset: dict[str, Any]) -> None:
    samples = dataset["samples"]
    assert len(samples) == 50
    assert Counter(row["expected_hit"] for row in samples) == {True: 25, False: 25}
    for row in samples:
        assert row["cached_question"] and row["cached_answer"] and row["probe_question"]


def _validate_unique_ids(*datasets: dict[str, Any]) -> None:
    ids = [row["id"] for dataset in datasets for row in dataset["samples"]]
    assert len(ids) == len(set(ids)) == 250


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_all(*, validate_only: bool = False) -> dict[str, dict[str, Any]]:
    rag = _build_rag_dataset()
    routing = _build_routing_dataset()
    redis_cache = _build_redis_dataset()
    _validate_rag(rag)
    _validate_routing(routing)
    _validate_redis(redis_cache)
    _validate_unique_ids(rag, routing, redis_cache)
    datasets = {"rag": rag, "routing": routing, "redis": redis_cache}
    if not validate_only:
        _write_json(EVAL_ROOT / "rag" / "dataset_150.json", rag)
        _write_json(EVAL_ROOT / "routing" / "dataset_50.json", routing)
        _write_json(EVAL_ROOT / "redis" / "dataset_50.json", redis_cache)
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    datasets = build_all(validate_only=args.validate_only)
    counts = {name: len(dataset["samples"]) for name, dataset in datasets.items()}
    print(json.dumps({"valid": True, "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
