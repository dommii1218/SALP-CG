# run_privacy_classify.py
# -*- coding: utf-8 -*-
import os, time, json, re
from typing import Any, Dict, List, Tuple
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from openai import APIStatusError

import requests
from requests.exceptions import RequestException

# ========= 载入密钥 & 初始化 =========
load_dotenv()

# —— 供应商切换 —— 
PROVIDER = "openai"                  # ← 关键：切换到 "ollama", "openai"
MODEL = "gpt-4o-mini"  # "qwen2.5:3b-instruct"或 "llama3.1:8b-instruct" 、"gpt-4o-mini"等
# 可选：自定义本地服务地址（默认 http://localhost:11434）
# export OLLAMA_BASE_URL=http://localhost:11434

# —— Ollama 服务配置（本机默认即可）——
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))

# —— OpenAI 仅在 PROVIDER=openai 时初始化 —— 
client = None
if PROVIDER.lower() == "openai":
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "未读到 OPENAI_API_KEY，请创建 .env 并写入 OPENAI_API_KEY=sk-xxxx"
    client = OpenAI(api_key=api_key, organization=os.getenv("OPENAI_ORG_ID"))

# ========= 基本配置 =========
INPUT_FILE = "2020(1-1000).xlsx"

SHEET_NAME = 0                # 也可填具体 sheet 名
TEXT_COL = "Description"      # 你的文本列名
MAX_ITEMS_PER_ROW = 50        # 每行最多返回多少条，防止过长
CHUNK_LONG_TEXT = True        # 对超长文本分块处理并合并去重
CHARS_PER_CHUNK = 3500        # 每块最大字符数
SLEEP_SEC = 0.25              # 简单节流，避免429
USE_SECOND_STAGE_CLASSIFIER = False  # 是否在抽取后再用二次分类器过滤（见文档）True|False
SECOND_STAGE_MAX_PER_ROW = 20  # 二次分类器每行最多处理多少条

# 若想只跑前几行做小样本验证，改为正整数；全量跑就设为 None
DEBUG_FIRST_N = 100

# One-shot 配置：off | mini | file | jsonl
FEWSHOT_MODE = "mini"            # 默认用精简版 one-shot；想关掉就设 "off","mini"
FEWSHOT_JSONL = "fewshots_with_level_5.jsonl"   # 或 "fewshots/fewshot.jsonl"
FEWSHOT_TOPK = 6                  # 每次放进上下文的示例条数
FEWSHOT_SELECT = "first"          # 可选: "first" | "random"

OUTPUT_FILE = f"{os.path.splitext(INPUT_FILE)[0]}_classified_entities({MODEL})_{FEWSHOT_MODE}.xlsx"

# —— Schema 开关（用于 ablation）——
USE_SCHEMA_HARD = True   # 是否启用 JSON-Schema 硬约束（response_format）
USE_SCHEMA_PROMPT = True  # 是否在提示里保留字段枚举/“只输出 JSON”的软提示

# 允许的字段枚举（完全按你给的写法）
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

# ========= 规则摘要（来自你给的表） =========
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
*特殊病种词表（节选）：
- 传染病：HIV, 艾滋, 乙肝, HBV, 丙肝, HCV, 梅毒, 淋病, 结核, TB, 新冠, 甲肝, 乙脑, 小三阳, 大三阳, 伤寒, 霍乱, 疟疾, 登革热, 流脑, 流感, 手足口, SARS, MERS, 禽流感, 埃博拉, 狂犬病, 布鲁氏菌病, 疱疹, 疱疹病毒, 巨细胞病毒, 乙型脑炎, 乙型脑膜炎
- 罕见病：戈谢病, 亨廷顿, 进行性肌营养不良, 进行性核上性麻痹, 视网膜色素变性, 血友病, 先天性心脏病
- 不治之症：阿尔茨海默, 老年痴呆, 帕金森, 运动神经元病, 渐冻症
- 恶性肿瘤：癌, 肉瘤, 黑色素瘤, 白血病, 淋巴瘤, 肝癌, 肺癌, 胃癌, 乳腺癌, 宫颈癌, 结直肠癌, 食道癌, 膀胱癌, 前列腺癌, 卵巢癌, 母细胞瘤
- 心理疾病：抑郁症, 焦虑症, 强迫症, 双相, 精分, 精神分裂, 失眠症, 躁狂症, 创伤后应激障碍, PTSD, 进食障碍, 自闭症, 多动症, ADHD, 精神障碍
- 遗传性：地中海贫血, 苯丙酮尿症, 马凡, 囊性纤维化
- 性生殖：梅毒, 淋病, 尖锐湿疣, HPV高危, HIV, 艾滋, 生殖器疱疹, 性病, 性传播疾病, 不孕, 不育
- 肛门疾病：痔疮, 肛裂, 肛瘘
命中以上词表时：field 优先用“特殊病种-对应类目”，level=5，哪怕是既往病史也标为特殊病种；若文本明确“疑似/已排除”，则改为 level=2。
- 疾病、疾病-已排除、疾病-疑似-> 2
- 生命体征：体温、脉搏、呼吸、心率、舒张压/收缩压、血氧饱和度、身高、体重 -> 3
- 主诉、过敏史、家族史、生活方式 -> 2
[医疗应用数据]
- 日期-日 -> 3；日期-月/年 -> 2
- 医疗机构内部所用号码：检验/检查结果单号、住院号、门(急)诊号 -> 4；病床号/病房号/手术间编号 -> 3
- 医疗服务信息：医生姓名 -> 4；医生姓名-姓 -> 3；就诊医院/科室/病区名称 -> 2
- 检查检验：敏感检查结果（HIV、肝炎、HPV高危(含 16、18、31、33、35、39、45、51、52、56、58、59、68、73、82 阳性/持续阳性/复阳”)等）-> 5；一般检查检验结果 -> 3；检查检验名称 -> 2
- 用药信息：药物名称、药物类型、药物用法、药物使用-频率、药物使用-剂量单位、药物使用-次剂量、药物使用-总剂量 -> 2
- 手术记录：手术名称、麻醉方式 -> 2
[医疗支付数据]
- 医疗交易：挂号费用/支付信息/消费金额/交易记录 -> 3
- 保险信息：保险账号/状态/金额 -> 4
[卫生资源/公共卫生]
- 医院基本数据（机构类别/学科门类/床位数/地址/电话等）-> 1
- 医院运营数据（人力/财务/物资/后勤/基础运行）-> 2
- 公共卫生数据（环境卫生、传染病暴发、疾病监测、预防数据、出生死亡率等）-> 2

抽取要求：
1) 不要臆造；尽量使用原文短语作为“实体”。
2) 对疾病：若属于“特殊病种”，请标出“特殊病种-xxx(病种类型，如恶性肿瘤)“正常情况为5级；如文本明确写“疑似/已排除”，统一降为2级。
3) 对医生姓名：全名=4级；只出现姓氏（如“陈主任”视为姓）=3级。
4) “检查检验名称/结果”要区分：名称=2级；一般结果=3级；敏感结果(HIV/肝炎等)=5级。
5) 对于地址，提取尽量完整的地址信息。
6) 去重（同一实体+字段+级别仅保留一次）。
7) 输出必须为 JSON，且总数不超过MAX_ITEMS。 
8) 仅从下述枚举中选择 field。
"""
#格式见下方 schema，

def _load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # 基础校验：输出必须含 items[] & max_risk_level
            if "input" in obj and "output" in obj and "items" in obj["output"] and "max_risk_level" in obj["output"]:
                data.append(obj)
    return data

def _pick_examples(examples, topk=6, mode="first"):
    if not examples:
        return []
    if mode == "random":
        import random
        return random.sample(examples, k=min(topk, len(examples)))
    return examples[:topk]  # first

def load_fewshot_pair():
    """
    在 FEWSHOT_MODE == 'mini' 时，从 FEWSHOT_JSONL 中只取 1 条示例
    （用 FEWSHOT_SELECT 控制 'first' / 'random'）。
    在 FEWSHOT_MODE == 'file' 时，从文件对读取。
    """
    mode = globals().get("FEWSHOT_MODE", "off")
    if mode == "mini":
        jsonl_path = globals().get("FEWSHOT_JSONL", "fewshots.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"⚠️ mini模式未找到 {jsonl_path}，降级为无 few-shot")
            return "", {}
        examples = _load_jsonl(jsonl_path)  # 你已实现
        picked = _pick_examples(examples, topk=1, mode=globals().get("FEWSHOT_SELECT","first"))
        if not picked:
            print("⚠️ JSONL 为空，mini模式降级为无 few-shot")
            return "", {}
        p = picked[0]
        return (p.get("input","").strip(), p.get("output") or {})

    elif mode == "file":
        in_path  = globals().get("FEWSHOT_FILE_INPUT", "")
        out_path = globals().get("FEWSHOT_FILE_OUTPUT", "")
        if not in_path or not os.path.exists(in_path) or not out_path or not os.path.exists(out_path):
            print("⚠️ One-shot 文件未找到，降级为无 few-shot")
            return "", {}
        with open(in_path, "r", encoding="utf-8") as f:
            ex_in = f.read().strip()
        with open(out_path, "r", encoding="utf-8") as f:
            ex_out = json.load(f)
        return ex_in, ex_out

    else:
        return "", {}

def ollama_chat(messages: List[Dict[str, str]], model: str) -> str:
    """
    用 Ollama /api/chat 发起一次对话，返回模型输出文本。
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "num_ctx": OLLAMA_NUM_CTX
        }
    }
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", "")


def build_messages_and_schema(text: str, max_items: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    构造给 Responses API / Chat Completions 使用的消息与严格 JSON Schema。
    依赖全局变量/函数：
      - FIELD_ENUM: List[str]  # 字段枚举（你提供的清单）
      - RULES: str             # 规则要点（分级说明）
      - FEWSHOT_MODE: str      # "jsonl" | "file" | "mini" | "off"
      - FEWSHOT_JSONL: str     # 当 jsonl 模式时的路径
      - FEWSHOT_TOPK: int      # 选用的示例条数
      - FEWSHOT_SELECT: str    # "first" | "random"
      - FEWSHOT_FILE_INPUT/FEWSHOT_FILE_OUTPUT: str  # 当 file 模式时的路径
      - _load_jsonl(path) -> List[dict], _pick_examples(examples, topk, mode), load_fewshot_pair() -> (ex_in, ex_out)
    """

    # ---------- 指令正文（含字段枚举 & 规则） ----------
    instruction = (
        "TASK: 抽取文本中的所有实体并给出分级；严格执行规则。\n"
        f"MAX_ITEMS: {max_items}\n"
        + RULES
    )
    if USE_SCHEMA_PROMPT:
        instruction += (
            "\n字段 field 只能从下面枚举中选择：\n- "
            + "\n- ".join(FIELD_ENUM)
            + "\n只输出 JSON，不要额外解释。"
            )
        
    if not USE_SCHEMA_HARD:
        instruction += """
        【输出格式（必须严格遵守）】
        请只输出以下 JSON 对象（不要输出额外文字）：
        {
            "items": [
            {"entity": "<原文短语>", "field": "<从枚举中选择>", "level": <1-5 的整数>}
            ],
            "max_risk_level": <0-5 的整数>
        }
        要求：
        - 若文本中出现任何可疑 PHI，请勿返回空列表；在不确定时选择最接近的类别，宁可少量过召回也不要漏掉。
        - 若确无可抽取项，才允许 items=[]
        """

    # ---------- few-shot 组装 ----------
    fewshot_msgs: List[Dict[str, Any]] = []
    if FEWSHOT_MODE == "jsonl":
        pairs = _pick_examples(_load_jsonl(FEWSHOT_JSONL), FEWSHOT_TOPK, FEWSHOT_SELECT)
        for p in pairs:
            ex_in  = (p.get("input") or "").strip()
            ex_out = p.get("output") or {}
            if ex_in and ex_out:
                fewshot_msgs.append({"role":"user","content": f"【示例输入】\n---\n{ex_in}\n---"})
                fewshot_msgs.append({"role":"assistant","content": json.dumps(ex_out, ensure_ascii=False, separators=(',',':'))})

    elif FEWSHOT_MODE in ("file", "mini"):
        # file/mini 模式统一由 load_fewshot_pair 提供 (ex_in, ex_out)
        ex_in, ex_out = load_fewshot_pair()
        if ex_in and ex_out:
            fewshot_msgs += [
                {"role":"user","content": f"【示例输入】\n---\n{ex_in}\n---"},
                {"role":"assistant","content": json.dumps(ex_out, ensure_ascii=False, separators=(',',':'))}
            ]
    else:
        # off：不加 few-shot
        pass

    # ---------- 最终消息 ----------
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "你是一个医疗数据分级抽取器。只输出 JSON，中文作答。"},
        {"role": "user",   "content": instruction},
        *fewshot_msgs,
        {"role": "user",   "content": f"【现在请处理这段文本】\n---\n{text}\n---"}
    ]

    # ---------- 严格 JSON Schema（Responses API 的 response_format） ----------
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

def call_once(text: str, max_items: int) -> Dict[str, Any]:
    """
    对单段文本调用一次 LLM：Ollama 或 OpenAI。
    - Ollama：不支持 response_format，只用提示约束 + 解析兜底
    - OpenAI：根据 USE_SCHEMA_HARD 决定是否传 response_format
    """
    messages, response_format = build_messages_and_schema(text, max_items)

    def _parse_json(raw: str) -> Dict[str, Any]:
        try:
            return json.loads(raw)
        except Exception:
            pass
        m = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.S)
        if not m:
            m = re.search(r"(\{.*\})", raw, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return {"items": [], "max_risk_level": 0, "_raw": raw, "_parse_error": True}
        return {"items": [], "max_risk_level": 0, "_raw": raw, "_parse_error": True}

    # ---- Ollama 路径 ----
    if PROVIDER.lower() == "ollama":
        # 这里不注入 schema 文本，只用你在 instruction 中保留的“软提示”
        raw = ollama_chat(messages, MODEL)
        return _parse_json(raw)

    # ---- OpenAI 路径 ----
    try:
        kwargs = {"model": MODEL, "input": messages, "temperature": 0}
        if USE_SCHEMA_HARD:
            kwargs["response_format"] = response_format
        resp = client.responses.create(**kwargs)
        raw = resp.output_text
        return _parse_json(raw)

    except TypeError:
        # 旧版 SDK 回退：不再硬塞 schema_str，保持“–Schema 硬约束”的纯粹消融
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0
        )
        raw = completion.choices[0].message.content
        return _parse_json(raw)

def chunk_text(text: str, chunk_size: int) -> List[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def merge_items(list_of_items: List[List[Dict[str, Any]]], max_items: int) -> List[Dict[str, Any]]:
    """合并多块抽取结果，去重并按 level desc 排序，限量"""
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

# ========= 二段式分类器：把“疾病/疑似/已排除” → 判定是否属于特殊病种 =========
# 目标类别（与你的字段严格一一对应）
SPECIAL_CATS = [
    "性生殖疾病","传染病","心理疾病","恶性肿瘤","遗传性疾病","肛门疾病","罕见病","不治之症","无"
]
CAT2FIELD = {
    "性生殖疾病": "特殊病种-性生殖疾病",
    "传染病":   "特殊病种-传染病",
    "心理疾病": "特殊病种-心理疾病",
    "恶性肿瘤": "特殊病种-恶性肿瘤",
    "遗传性疾病": "特殊病种-遗传性疾病",
    "肛门疾病": "特殊病种-肛门疾病",
    "罕见病":   "特殊病种-罕见病",
    "不治之症": "特殊病种-不治之症",
    "无": None
}

# 可选：与主模型不同的轻量模型（比如二段式用更便宜/更快的模型）
MODEL_SECOND_STAGE = os.getenv("MODEL_SECOND_STAGE", MODEL)

# 简单缓存，避免同一实体反复判定
_CLASSIFY_CACHE: Dict[str, Dict[str, Any]] = {}

def _json_parse_hard(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.S)
    if not m:
        m = re.search(r"(\{.*\})", raw, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return {"category":"无","is_suspected":False,"is_ruled_out":False,"reason":raw[:80],"__parse_error":True}
    return {"category":"无","is_suspected":False,"is_ruled_out":False,"reason":raw[:80],"__parse_error":True}


def _classify_with_openai(messages: List[Dict[str,str]]) -> Dict[str, Any]:
    # 目标 JSON 结构（给 Responses 用 & 给回退提示用）
    schema_core = {
        "type":"object",
        "properties":{
            "category":{"type":"string","enum": SPECIAL_CATS},
            "is_suspected":{"type":"boolean"},
            "is_ruled_out":{"type":"boolean"},
            "reason":{"type":"string"}
        },
        "required":["category","is_suspected","is_ruled_out","reason"],
        "additionalProperties": False
    }

    try:
        # ✅ 新版 SDK：Responses + JSON Schema（首选路径）
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "special_category",
                "strict": True,
                "schema": schema_core
            }
        }
        resp = client.responses.create(
            model=MODEL_SECOND_STAGE,
            input=messages,
            response_format=response_format,
            temperature=0
        )
        return _json_parse_hard(resp.output_text)

    except (TypeError, AttributeError):
        # 🔁 旧版 SDK 回退：用 Chat Completions + 硬提示
        msgs = list(messages) + [{
            "role": "user",
            "content": (
                "仅输出一个JSON\n" #，必须满足此 JSON Schema：
                + json.dumps(schema_core, ensure_ascii=False)
                + "\n只输出 JSON，不要任何解释或额外文本。"
            )
        }]
        comp = client.chat.completions.create(
            model=MODEL_SECOND_STAGE,
            messages=msgs,
            temperature=0
        )
        raw = comp.choices[0].message.content
        return _json_parse_hard(raw)


def classify_special_category(entity_text: str, context_snippet: str = "") -> Dict[str, Any]:
    """
    返回:
    {
      "category": "传染病|恶性肿瘤|心理疾病|遗传性疾病|肛门疾病|性生殖疾病|罕见病|不治之症|无",
      "is_suspected": bool,
      "is_ruled_out": bool,
      "reason": "简短中文理由"
    }
    """
    key = (entity_text + "||" + context_snippet).strip()
    if key in _CLASSIFY_CACHE:
        return _CLASSIFY_CACHE[key]

    prompt = f"""你是医疗分类器。判断下述“疾病实体”是否属于以下特殊病种：
[性生殖疾病, 传染病, 心理疾病, 恶性肿瘤, 遗传性疾病, 肛门疾病, 罕见病, 不治之症, 无]
- 只按“疾病临床类别”判断，不要输出其它类型字段。
- 若文本中出现“疑似/考虑/待排/排除”等，做相应布尔标记。
- 如无法归入上面任何类，则输出 "无"。

实体：{entity_text}
上下文（可空）：{context_snippet}"""

    messages = [
        {"role":"system","content":"你是一个严谨的医疗分类器，只输出 JSON。"},
        {"role":"user","content": prompt}
    ]
    if PROVIDER.lower() == "openai":
        out = _classify_with_openai(messages)
    else:
        out = _classify_with_ollama(messages)

    # 兜底清洗
    cat = out.get("category","无")
    if cat not in SPECIAL_CATS:
        cat = "无"
    out = {
        "category": cat,
        "is_suspected": bool(out.get("is_suspected", False)),
        "is_ruled_out": bool(out.get("is_ruled_out", False)),
        "reason": str(out.get("reason","")).strip()[:200]
    }
    _CLASSIFY_CACHE[key] = out
    return out

def _slice_context(text: str, entity: str, win: int = 80) -> str:
    """从原文中截取 entity 附近的上下文，帮助判别“疑似/排除”等语气。"""
    try:
        i = text.find(entity)
        if i == -1:  # 找不到就截前后 200 字
            return text[:200]
        start = max(0, i - win)
        end = min(len(text), i + len(entity) + win)
        return text[start:end]
    except Exception:
        return text[:200]

def classify_and_upgrade_special(items: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    """
    对 field 以“疾病”开头的条目逐一做“特殊病种”判定；命中则将
    field → 特殊病种-XXX，level → 5（若疑似/已排除/原字段含“疑似/已排除”则降为 2）。
    """
    if not items:
        return items

    # 仅挑未被规则升级命中的“疾病*”项（避免重复工作）
    candidates = []
    for i, it in enumerate(items):
        f = str(it.get("field",""))
        if f.startswith("疾病"):  # 疾病 / 疾病-疑似 / 疾病-已排除
            candidates.append((i, it))

    # 控制单行调用量
    max_n = int(globals().get("SECOND_STAGE_MAX_PER_ROW", 20) or 20)
    candidates = candidates[:max_n]

    new_items = list(items)  # 拷贝
    for idx, it in candidates:
        ent = str(it.get("entity","")).strip()
        if not ent:
            continue
        ctx = _slice_context(text, ent, 100)
        res = classify_special_category(ent, ctx)
        cat = res["category"]
        if cat == "无":
            continue
        # 命中：替换为特殊病种字段，并设置级别
        suspected = res["is_suspected"]
        ruled_out = res["is_ruled_out"]
        orig_field = str(it.get("field",""))
        is_uncertain = ("疑似" in orig_field) or ("已排除" in orig_field) or suspected or ruled_out

        new_field = CAT2FIELD[cat]
        new_level = 2 if is_uncertain else 5
        new_items[idx] = {**it, "field": new_field, "level": new_level}

    # 再做一次去重（避免与先前规则升级产生重复）
    seen = set()
    deduped = []
    for it in new_items:
        key = (it.get("entity","").strip(), it.get("field","").strip(), int(it.get("level",0)))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        deduped.append({"entity": key[0], "field": key[1], "level": key[2]})
    # 维持你原来的排序逻辑
    deduped.sort(key=lambda x: (-x["level"], x["field"], x["entity"]))
    return deduped


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


def upgrade_to_special(items, text):
    """把 '疾病/疑似/已排除' 命中的实体升级为 '特殊病种-类目'，并调整 level。"""
    import re
    upgraded = []
    for it in items:
        ent, field, level = it["entity"], it["field"], it["level"]
        new_field, new_level = field, level
        if field.startswith("疾病"):
            for cat, patterns in SPECIAL_MAP.items():
                if any(re.search(p, ent) for p in patterns):
                    # 是否疑似/已排除 → 2；否则 5
                    if "疑似" in field or "已排除" in field:
                        new_level = 2
                    else:
                        new_level = 5
                    new_field = f"特殊病种-{cat}"
                    break
        upgraded.append({**it, "field": new_field, "level": new_level})
    return upgraded


HR_TYPES = {"16","18","31","33","35","39","45","51","52","56","58","59","68","73","82"}

# 构造一个用于正则的类型列表（匹配 16、16型、16/18、16、18、52 等组合）
HR_TYPE_GROUP = r"(?:%s)(?:\s*型)?" % "|".join(sorted(HR_TYPES, key=int))

# 常见高危表述：hrHPV、高危(型)HPV、HPV16/18/52阳性、HPV58 持续阳性……
HR_HPV_POS_PATTERNS = [
    # 高危/HR 直接表述
    r"(?i)高危(?:型)?\s*HPV.*?(阳性|positive)",
    r"(?i)hr-?\s*hpv.*?(阳性|positive)",
    # 指定分型 + 阳性（支持 16/18/52 这种用 / 、、 ， 分隔）
    rf"(?i)\bhpv[\s:]*(?:{HR_TYPE_GROUP})(?:\s*[\/、,，]\s*{HR_TYPE_GROUP})*\s*.*?(阳性|positive)",
]

def is_high_risk_hpv_positive(text: str) -> bool:
    if not text:
        return False
    t = str(text)
    # 排除明显“阴性/未检出”
    if re.search(r"(阴性|未检出|negative)", t, flags=re.I):
        return False
    return any(re.search(p, t, flags=re.I) for p in HR_HPV_POS_PATTERNS)

def upgrade_sensitive_hpv(items):
    """把命中的高危型 HPV 阳性结果升级为 敏感检查结果(5)；疑似/已排除则降为 2。"""
    out = []
    for it in items:
        ent = str(it.get("entity",""))
        field = str(it.get("field",""))
        level = int(it.get("level", 0))

        if is_high_risk_hpv_positive(ent):
            # 若文本里有“疑似/已排除”痕迹，按你的口径记为 2，否则 5
            if re.search(r"(疑似|考虑|待排|已排除)", ent):
                it["field"] = "敏感检查结果"
                it["level"] = 2
            else:
                it["field"] = "敏感检查结果"
                it["level"] = 5

        out.append(it)
    return out

def to_tuple_string(items: List[Dict[str, Any]]) -> str:
    # 生成你示例那种格式： （实体，字段，级别），（...）
    return "，".join([f"（{i['entity']}，{i['field']}，{i['level']}）" for i in items])

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def process_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    逐行处理：
      - 调用 LLM 抽取 -> merge 去重限量
      - 升级为特殊病种（词表） -> 升级敏感结果（高危 HPV）
      - （可选）二段式分类器再升级一次
      - 写入行级汇总与实体明细
    """
    results_rows: List[Dict[str, Any]] = []
    details_rows: List[Dict[str, Any]] = []

    # 是否启用二段式分类器（如果你已经实现了 classify_and_upgrade_special）
    USE_SECOND_STAGE = bool(globals().get("USE_SECOND_STAGE_CLASSIFIER", False))
    second_stage_fn = globals().get("classify_and_upgrade_special", None)

    total = len(df) if DEBUG_FIRST_N is None else min(DEBUG_FIRST_N, len(df))

    for idx in tqdm(range(total), desc="Processing"):
        # 取文本
        raw = df.loc[idx, TEXT_COL] if TEXT_COL in df.columns else ""
        text = "" if pd.isna(raw) else str(raw).strip()

        # 空行兜底
        if not text:
            results_rows.append({
                "row_id": idx,
                "max_risk_level": 0,
                "items_json": "[]",
                "items_tuple_str": ""
            })
            continue

        # ========== 调用模型（分块 or 直接） ==========
        if CHUNK_LONG_TEXT and len(text) > CHARS_PER_CHUNK:
            chunks = chunk_text(text, CHARS_PER_CHUNK)
            chunk_items: List[List[Dict[str, Any]]] = []
            for ck in chunks:
                data = call_once(ck, MAX_ITEMS_PER_ROW)
                chunk_items.append(data.get("items", []))
                time.sleep(SLEEP_SEC)
            # 合并 & 去重 & 限量
            items = merge_items(chunk_items, MAX_ITEMS_PER_ROW)
        else:
            data = call_once(text, MAX_ITEMS_PER_ROW)
            items = merge_items([data.get("items", [])], MAX_ITEMS_PER_ROW)

        # ========== 升级规则（先特殊病种，再敏感结果） ==========
        #items = upgrade_to_special(items, text)   # “疾病/疑似/已排除” 命中词表 → 特殊病种-X（5 or 2）
        #items = upgrade_sensitive_hpv(items)      # 高危型 HPV 阳性 → 敏感检查结果(5；疑似/已排除=2)

        # ========== （可选）二段式：对“疾病类”逐条再判一次 ==========
        if USE_SECOND_STAGE and callable(second_stage_fn):
            items = second_stage_fn(items, text)

        # 重新计算当行最高级别（考虑升级后的 5 级等）
        max_level = max([int(it.get("level", 0) or 0) for it in items] + [0])

        # 行级结果
        results_rows.append({
            "row_id": idx,
            "max_risk_level": max_level,
            "items_json": json.dumps(items, ensure_ascii=False),
            "items_tuple_str": to_tuple_string(items)
        })

        # 明细行
        for rank, it in enumerate(items, start=1):
            details_rows.append({
                "row_id": idx,
                "rank": rank,
                "entity": it.get("entity", ""),
                "field": it.get("field", ""),
                "level": int(it.get("level", 0) or 0)
            })

        time.sleep(SLEEP_SEC)

    df_results = pd.DataFrame(results_rows)
    df_details = pd.DataFrame(details_rows)
    return df_results, df_details


def main():
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    if TEXT_COL not in df.columns:
        raise ValueError(f"找不到列 '{TEXT_COL}'，当前列：{list(df.columns)}")

    try:
        df_results, df_details = process_dataframe(df)
    except APIStatusError as e:
        # 更友好的报错提示
        try:
            err = e.response.json().get("error", {})
        except Exception:
            err = {}
        code = err.get("code")
        msg = err.get("message", str(e))
        if code == "insufficient_quota":
            raise SystemExit("❌ 配额不足：请到 OpenAI 平台 Usage/Billing 检查用量与计费，或确认组织/项目是否正确。")
        elif e.status_code == 429:
            raise SystemExit("⚠️ 触发速率限制：请降低并发、增大 SLEEP_SEC 或拆分批次。")
        else:
            raise

    # 合并到一个工作簿（两张表）
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        # 原始数据 + 行级结果
        out = df.copy()
        out = out.iloc[:len(df_results)].copy()
        out["max_risk_level"] = df_results["max_risk_level"]
        out["entities_json"] = df_results["items_json"]
        out["entities_tuples"] = df_results["items_tuple_str"]
        out.to_excel(writer, index=False, sheet_name="results")

        # 明细拆分
        df_details.to_excel(writer, index=False, sheet_name="entities")

    print(f"✅ 已生成：{OUTPUT_FILE}\n- Sheet 'results': 行级汇总（含 tuple 串 & JSON）\n- Sheet 'entities': 明细拆分（每个实体一行）")

if __name__ == "__main__":
    main()
