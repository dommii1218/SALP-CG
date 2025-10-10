# run_privacy_classify_gpt5.py
# -*- coding: utf-8 -*-
import os, time, json, re
from typing import Any, Dict, List, Tuple
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

import openai
from openai import OpenAI

# ========= 载入密钥 & 初始化 =========
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "未读到 OPENAI_API_KEY，请创建 .env 并写入 OPENAI_API_KEY=sk-xxxx"

client = OpenAI(api_key=api_key, organization=os.getenv("OPENAI_ORG_ID"))

# 只用 gpt-5（如需固定版本，可改为 'gpt-5-2025-08-07'）
MODEL = "gpt-5-mini-2025-08-07"

# ========= 基本配置 =========
INPUT_FILE = "2020(101-1000).xlsx"
SHEET_NAME = 0                # 也可填具体 sheet 名
TEXT_COL = "Description"      # 你的文本列名
MAX_ITEMS_PER_ROW = 50        # 每行最多返回多少条，防止过长
CHUNK_LONG_TEXT = True        # 对超长文本分块处理并合并去重
CHARS_PER_CHUNK = 3500        # 每块最大字符数
SLEEP_SEC = 0.25              # 简单节流，避免429

# 若想只跑前几行做小样本验证，改为正整数；全量跑就设为 None
DEBUG_FIRST_N = None

# One-shot 配置：off | jsonl | file | mini
FEWSHOT_MODE = "jsonl"            # 默认用 jsonl few-shot；想关掉就设 "off"
FEWSHOT_JSONL = "fewshots_with_level_5.jsonl"
FEWSHOT_TOPK = 6
FEWSHOT_SELECT = "first"          # "first" | "random"

OUTPUT_FILE = f"{os.path.splitext(INPUT_FILE)[0]}_classified_entities({MODEL})_{FEWSHOT_MODE}.xlsx"

# 允许的字段枚举（你给的清单）

FIELD_ENUM = [
    "疾病","疾病-疑似","疾病-已排除",
    "特殊病种-性生殖疾病","特殊病种-传染病","特殊病种-心理疾病","特殊病种-恶性肿瘤","特殊病种-遗传性疾病","特殊病种-肛门疾病","特殊病种-罕见病","特殊病种-不治之症",
    "主诉","过敏史","家族史","生活方式",
    "生命体征-体温","生命体征-脉搏","生命体征-呼吸","生命体征-心率","生命体征-血压","生命体征-血氧饱和度","身高","体重",
    "日期-日","日期-月","日期-年",
    "检查检验名称","检查检验结果","敏感检查结果",
    "药物名称","药物用法","药物使用-频率","药物使用-剂量单位","药物使用-次剂量","药物使用-总剂量","药物类型",
    "手术名称","麻醉方式",
    "医生姓名","医生姓名-姓","就诊医院","科室名称","病区名称",
    "检查结果报告单号","检验结果报告单号","住院号","门（急）诊号","病床号","病房号","手术间编号",
    "患者姓名","患者姓名-姓","性别","年龄","出生日期-日","出生日期-月","国籍","民族","婚姻状态","爱好","信仰","工作单位","职业","收入","家庭成员姓名","联系人姓名",
    "地址-省","地址-市","地址-区","地址-门牌号码","地址-村","住址-乡",
    "手机号","电话","邮箱","社保账号","身份证号","驾驶证号","车牌号","个税号","IP地址","DeviceID",
    "保险账号","保险状态","保险金额",
    "挂号费用","支付信息","消费金额","交易记录",
    "医院机构类别","医院学科门类","床位数","医院地址","医院电话",
    "医院运营数据","公共卫生数据"
]

# ========= 分级规则（摘要） =========
RULES = """
你将看到一段互联网医院问诊文本，请抽取“实体、对应字段（类别）、级别（1~5）”。去重后按级别从高到低排序。
分级规则（摘取要点，严格执行）：
[人口属性数据]
- 患者姓名、地址(门牌/村/乡)、家庭成员姓名、联系人姓名、爱好、信仰 -> 4
*“家庭成员姓名/联系人姓名”必须是明确姓名，如“我妈妈张三”或“我老婆李四”，而非“我妈妈”“我老婆”。
- 患者姓名-姓、性别、出生日期-日、年龄、工作单位、职业、地址-区 -> 3
- 出生日期-月、民族、国籍、收入、婚姻状态、地址(省/市) -> 2
[个人身份/通讯/信用]
- 身份证、工作证、居住证、社保卡、健康卡号、住院号、各类检查检验相关单号 -> 4
- 手机号、电话、邮箱、银行账号、支付宝账号、微信账号、社保账号、身份证号码、医院住院卡账号、驾驶证、车牌、个税号、IP、手机Device ID -> 4
- 个人信用档案/评分/报告 -> 4
[健康状况数据]
- 特殊病种：性生殖疾病、传染病(甲/乙/丙类)、心理疾病、恶性肿瘤、遗传性疾病、肛门疾病、罕见病、其他不治之症-> 5
- 疾病、疾病-已排除、疾病-疑似-> 2
- 生命体征：体温、脉搏、呼吸、心率、血压、血氧饱和度、身高、体重 -> 3
- 主诉、过敏史、家族史、生活方式 -> 2
[医疗应用数据]
- 日期-日 -> 3；日期-月/年 -> 2
- 医疗机构内部所用号码：检验/检查结果单号、住院号、门(急)诊号 -> 4；病床号/病房号/手术间编号 -> 3
- 医疗服务信息：医生姓名 -> 4；医生姓名-姓 -> 3；就诊医院/科室/病区名称 -> 2
- 检查检验：敏感检查结果（HIV、肝炎、HPV高危等）-> 5；一般检查检验结果 -> 3；检查检验名称 -> 2
- 用药信息：药物名称、药物类型、药物用法、药物使用-频率、药物使用-剂量单位、药物使用-次剂量、药物使用-总剂量 -> 2
- 手术记录：手术名称、麻醉方式 -> 2
[医疗支付数据]
- 医疗交易：挂号费用/支付信息/消费金额/交易记录 -> 3
- 保险信息：保险账号/状态/金额 -> 4
[卫生资源/公共卫生]
- 医院基本数据（机构类别/学科门类/床位数/地址/电话等）-> 1
- 医院运营数据（人力/财务/物资/后勤/基础运行）-> 2
抽取要求：
1) 不要臆造；尽量使用原文短语作为“实体”。
2) “特殊病种-xxx”命中时 level=5；若文本明确“疑似/已排除”，则统一为 level=2。
3) 医生姓名：全名=4级；只出现姓氏（如“陈主任”视为姓）=3级。
4) “检查检验名称/结果”要区分：名称=2级；一般结果=3级；敏感结果=5级。
5) 去重（同一实体+字段+级别仅保留一次）。
6) 输出必须为 JSON，遵循下方 JSON Schema；总数不超过 MAX_ITEMS。
7) field 只能从给定枚举中选择。
"""

# -------- few-shot 工具 --------
def _load_jsonl(path):
    data = []
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "input" in obj and "output" in obj and "items" in obj["output"] and "max_risk_level" in obj["output"]:
                data.append(obj)
    return data

def _pick_examples(examples, topk=6, mode="first"):
    if not examples:
        return []
    if mode == "random":
        import random
        return random.sample(examples, k=min(topk, len(examples)))
    return examples[:topk]

# -------- Prompt & Schema 构造 --------
def build_messages_and_schema(text: str, max_items: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    instruction = (
        "TASK: 抽取文本中的所有实体并给出分级；严格执行规则。\n"
        f"MAX_ITEMS: {max_items}\n"
        + RULES
        + "\n字段 field 只能从下面枚举中选择：\n- " + "\n- ".join(FIELD_ENUM) +
        "\n请按 JSON Schema 输出，不要添加任何解释或额外字段。"
    )

    fewshot_msgs: List[Dict[str, Any]] = []
    if FEWSHOT_MODE == "jsonl":
        pairs = _pick_examples(_load_jsonl(FEWSHOT_JSONL), FEWSHOT_TOPK, FEWSHOT_SELECT)
        for p in pairs:
            ex_in  = (p.get("input") or "").strip()
            ex_out = p.get("output") or {}
            if ex_in and ex_out:
                fewshot_msgs.append({"role":"user","content": f"【示例输入】\n---\n{ex_in}\n---"})
                fewshot_msgs.append({"role":"assistant","content": json.dumps(ex_out, ensure_ascii=False, separators=(',',':'))})

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "你是一个医疗数据分级抽取器。只输出 JSON，中文作答。"},
        {"role": "user",   "content": instruction},
        *fewshot_msgs,
        {"role": "user",   "content": f"【现在请处理这段文本】\n---\n{text}\n---"}
    ]

    response_format: Dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {
            "name": "med_privacy_entities",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "maxItems": max_items,
                        "items": {
                            "type": "object",
                            "properties": {
                                "entity": {"type": "string"},
                                "field":  {"type": "string", "enum": FIELD_ENUM},
                                "level":  {"type": "integer", "minimum": 1, "maximum": 5}
                            },
                            "required": ["entity", "field", "level"],
                            "additionalProperties": False
                        }
                    },
                    "max_risk_level": {"type": "integer", "minimum": 0, "maximum": 5}
                },
                "required": ["items", "max_risk_level"],
                "additionalProperties": False
            }
        }
    }
    return messages, response_format

# -------- 解析辅助 --------
def _parse_json(raw: str) -> Dict[str, Any]:
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "items" in obj and "max_risk_level" in obj:
            return obj
    except Exception:
        pass
    # 兜底：返回空结构
    return {"items": [], "max_risk_level": 0}

# -------- 调用一次 gpt-5 抽取 --------
def call_once(text: str, max_items: int) -> Dict[str, Any]:
    # 组装 prompt 和 JSON Schema
    messages, response_format = build_messages_and_schema(text, max_items)

    # 路径 A：Chat Completions + Structured Outputs（首选；不要传 temperature/top_p）
    try:
        comp = client.chat.completions.create(
            model=MODEL,              # 例如 "gpt-5" 或 "gpt-5-2025-08-07"
            messages=messages,
            response_format=response_format,  # {"type":"json_schema","json_schema":{"strict":True,...}}
        )
        raw = comp.choices[0].message.content
        return _parse_json(raw)

    # 某些旧版 SDK 不支持在 chat.completions 里传 response_format → TypeError
    except TypeError:
        # 路径 B：无 response_format 时，用“硬提示”把 Schema 严格化
        schema_str = json.dumps(response_format["json_schema"]["schema"], ensure_ascii=False)
        msgs = list(messages) + [{
            "role": "user",
            "content": (
                "只输出一个 JSON，必须完全满足这个 JSON Schema（字段、类型、必填、不得有多余键）：\n"
                f"{schema_str}\n只输出 JSON，不要任何额外文字。"
            )
        }]
        comp2 = client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            # 仍然不要传 temperature/top_p/penalties
        )
        raw2 = comp2.choices[0].message.content
        return _parse_json(raw2)


# -------- 文本分块/合并/升级规则 --------
def chunk_text(text: str, chunk_size: int) -> List[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def merge_items(list_of_items: List[List[Dict[str, Any]]], max_items: int) -> List[Dict[str, Any]]:
    seen = set()
    merged = []
    for items in list_of_items:
        for it in items:
            key = (it.get("entity","").strip(), it.get("field","").strip(), int(it.get("level",0)))
            if not key[0] or key in seen:
                continue
            seen.add(key)
            merged.append({"entity": key[0], "field": key[1], "level": key[2]})
    merged.sort(key=lambda x: (-x["level"], x["field"], x["entity"]))
    return merged[:max_items]

SPECIAL_MAP = {
    "传染病":  [
        r"\bHIV\b|艾滋", "甲肝", r"乙肝|HBV", r"丙肝|HCV", r"非典|SARS", "肺结核|结核|TB",
        "狂犬病", "麻疹", "登革热", "炭疽", "伤寒", "霍乱", "鼠疫", "流脑", "流感", "手足口",
        "腮腺炎", r"布鲁氏菌|布氏", "钩端螺旋体", "血吸虫", "疟疾", "猴痘", "COVID|新冠",
        "疱疹", "巨细胞病毒|CMV"
    ],
    "恶性肿瘤": ["癌", "肉瘤", "黑色素瘤", "白血病", "淋巴瘤", "肝癌", "肺癌", "胃癌", "乳腺癌", "宫颈癌",
               "结直肠癌|大肠癌|结肠癌|直肠癌", "食道癌", "膀胱癌", "前列腺癌", "卵巢癌"],
    "心理疾病": ["抑郁症", "焦虑症", "强迫症", "双相", "精神分裂|精分", "失眠症", "躁狂症",
               "PTSD|创伤后应激", "进食障碍", "自闭症", "多动症|ADHD", "精神障碍"],
    "遗传性疾病": ["地中海贫血", "苯丙酮尿症", "马凡", "囊性纤维化"],
    "性生殖疾病": ["梅毒", "淋病", "尖锐湿疣", "HPV", "支原体|解脲支原体|人型支原体|生殖支原体", "衣原体"],
    "肛门疾病": ["痔疮", "肛裂", "肛瘘"],
    "罕见病":   ["亨廷顿", "戈谢病"],
    "不治之症": ["阿尔茨海默|老年痴呆", "帕金森", "运动神经元病|渐冻症|ALS"]
}

HR_TYPES = {"16","18","31","33","35","39","45","51","52","56","58","59","68","73","82"}
HR_TYPE_GROUP = r"(?:%s)(?:\s*型)?" % "|".join(sorted(HR_TYPES, key=int))
HR_HPV_POS_PATTERNS = [
    r"(?i)高危(?:型)?\s*HPV.*?(阳性|positive)",
    r"(?i)hr-?\s*hpv.*?(阳性|positive)",
    rf"(?i)\bhpv[\s:]*(?:{HR_TYPE_GROUP})(?:\s*[\/、,，]\s*{HR_TYPE_GROUP})*\s*.*?(阳性|positive)",
]

def is_high_risk_hpv_positive(text: str) -> bool:
    if not text:
        return False
    t = str(text)
    if re.search(r"(阴性|未检出|negative)", t, flags=re.I):
        return False
    return any(re.search(p, t, flags=re.I) for p in HR_HPV_POS_PATTERNS)

def upgrade_to_special(items, text):
    out = []
    for it in items:
        ent, field, level = it["entity"], it["field"], it["level"]
        new_field, new_level = field, level
        if field.startswith("疾病"):
            for cat, patterns in SPECIAL_MAP.items():
                if any(re.search(p, ent) for p in patterns):
                    new_field = f"特殊病种-{cat}"
                    new_level = 2 if ("疑似" in field or "已排除" in field) else 5
                    break
        out.append({**it, "field": new_field, "level": new_level})
    return out

def upgrade_sensitive_hpv(items):
    out = []
    for it in items:
        ent = str(it.get("entity",""))
        field = str(it.get("field",""))
        if is_high_risk_hpv_positive(ent):
            it["field"] = "敏感检查结果"
            it["level"] = 2 if re.search(r"(疑似|考虑|待排|已排除)", ent) else 5
        out.append(it)
    return out

def to_tuple_string(items: List[Dict[str, Any]]) -> str:
    return "，".join([f"（{i['entity']}，{i['field']}，{i['level']}）" for i in items])

# -------- 主循环 --------
def process_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results_rows: List[Dict[str, Any]] = []
    details_rows: List[Dict[str, Any]] = []

    total = len(df) if DEBUG_FIRST_N is None else min(DEBUG_FIRST_N, len(df))

    for idx in tqdm(range(total), desc="Processing"):
        raw = df.loc[idx, TEXT_COL] if TEXT_COL in df.columns else ""
        text = "" if pd.isna(raw) else str(raw).strip()

        if not text:
            results_rows.append({"row_id": idx, "max_risk_level": 0, "items_json": "[]", "items_tuple_str": ""})
            continue

        if CHUNK_LONG_TEXT and len(text) > CHARS_PER_CHUNK:
            chunks = chunk_text(text, CHARS_PER_CHUNK)
            chunk_items: List[List[Dict[str, Any]]] = []
            for ck in chunks:
                data = call_once(ck, MAX_ITEMS_PER_ROW)
                chunk_items.append(data.get("items", []))
                time.sleep(SLEEP_SEC)
            items = merge_items(chunk_items, MAX_ITEMS_PER_ROW)
        else:
            data = call_once(text, MAX_ITEMS_PER_ROW)
            items = merge_items([data.get("items", [])], MAX_ITEMS_PER_ROW)

        # 升级规则（词表 → 特殊病种；高危 HPV → 敏感结果）
        items = upgrade_to_special(items, text)
        items = upgrade_sensitive_hpv(items)

        max_level = max([int(it.get("level", 0) or 0) for it in items] + [0])

        results_rows.append({
            "row_id": idx,
            "max_risk_level": max_level,
            "items_json": json.dumps(items, ensure_ascii=False),
            "items_tuple_str": to_tuple_string(items)
        })

        for rank, it in enumerate(items, start=1):
            details_rows.append({
                "row_id": idx, "rank": rank,
                "entity": it.get("entity", ""),
                "field": it.get("field", ""),
                "level": int(it.get("level", 0) or 0)
            })

        time.sleep(SLEEP_SEC)

    return pd.DataFrame(results_rows), pd.DataFrame(details_rows)

def main():
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    if TEXT_COL not in df.columns:
        raise ValueError(f"找不到列 '{TEXT_COL}'，当前列：{list(df.columns)}")

    try:
        df_results, df_details = process_dataframe(df)
    except openai.APIStatusError as e:
        try:
            err = e.response.json().get("error", {}) if hasattr(e, "response") else {}
        except Exception:
            err = {}
        code = err.get("code")
        if code == "insufficient_quota":
            raise SystemExit("❌ 配额不足：请到 OpenAI 平台 Usage/Billing 检查用量与计费，或确认组织/项目是否正确。")
        elif getattr(e, "status_code", None) == 429:
            raise SystemExit("⚠️ 触发速率限制：请降低并发、增大 SLEEP_SEC 或拆分批次。")
        else:
            raise
    except (openai.RateLimitError, openai.APIConnectionError, openai.APIError) as e:
        raise

    # 导出
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        out = df.iloc[:len(df_results)].copy()
        out["max_risk_level"] = df_results["max_risk_level"]
        out["entities_json"] = df_results["items_json"]
        out["entities_tuples"] = df_results["items_tuple_str"]
        out.to_excel(writer, index=False, sheet_name="results")
        df_details.to_excel(writer, index=False, sheet_name="entities")

    print(f"✅ 已生成：{OUTPUT_FILE}\n- Sheet 'results': 行级汇总（含 tuple 串 & JSON）\n- Sheet 'entities': 明细拆分（每个实体一行）")

if __name__ == "__main__":
    main()
