# compute_all_metrics.py
# -*- coding: utf-8 -*-
import argparse, json, re, unicodedata
from typing import List, Dict, Iterable, Tuple, Set
from collections import Counter
import pandas as pd
import numpy as np

# =========================
# 解析与归一化
# =========================
def normalize_text(s: str) -> str:
    """实体归一化：NFKC + 去首尾空白 + 连续空白折叠"""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = " ".join(s.split())
    return s

def _parse_tuple_string(s: str, allowed_categories=None):
    """
    解析形如： （实体，...，实体片段，类别，级别） 的串
    - 级别：最后一个数字
    - 类别：从右往左第一个命中 allowed_categories 的片段
    - 实体：其余全部拼回
    """
    import re, unicodedata

    if allowed_categories is None:
        # 用你代码里的 FIELD_ENUM / DEFAULT_SCHEMA 之一
        allowed_categories = set(DEFAULT_SCHEMA)  # 或 set(FIELD_ENUM)

    def norm(x):
        x = unicodedata.normalize("NFKC", str(x)).strip()
        x = " ".join(x.split())
        return x

    out = []
    for grp in re.findall(r"（([^（）]+)）", s):
        parts = [norm(p) for p in re.split(r"[，,]\s*", grp) if norm(p)]
        if not parts:
            continue

        # 1) 级别：最后一个数字
        L = 0
        if re.fullmatch(r"\d+", parts[-1]):
            L = int(parts.pop())  # 去掉最后一个，作为 level

        # 2) 类别：从右往左找第一个在枚举里的片段
        C_idx = None
        for i in range(len(parts)-1, -1, -1):
            if parts[i] in allowed_categories:
                C_idx = i
                break

        if C_idx is None:
            # 兜底：如果找不到，就按老规则取倒数第一个作为类别
            if len(parts) >= 2:
                C = parts[-1]
                P = "，".join(parts[:-1])
            else:
                # 只有一个片段：当作纯实体
                C = ""
                P = parts[0]
        else:
            C = parts[C_idx]
            P = "，".join(parts[:C_idx] + parts[C_idx+1:])  # 去掉类别，其余为实体

        out.append({"P": P, "C": C, "L": L})
    return out


def _coerce_item_dict(d):
    """
    统一成 {'P','C','L'}。兼容：
      - {'entity','field','level'}
      - {'P','C','L'} 或大小写变体
    """
    if not isinstance(d, dict):
        return None
    if "entity" in d or "field" in d or "level" in d:
        P = normalize_text(d.get("entity", ""))
        C = normalize_text(d.get("field", ""))
        try:
            L = int(d.get("level", 0))
        except Exception:
            L = 0
        return {"P": P, "C": C, "L": L}
    P = normalize_text(d.get("P", d.get("p", "")))
    C = normalize_text(d.get("C", d.get("c", "")))
    try:
        L = int(d.get("L", d.get("l", 0)))
    except Exception:
        L = 0
    if P or C or L:
        return {"P": P, "C": C, "L": L}
    return None

def parse_cell(cell):
    """
    把单元格解析为 [{'P','C','L'}, ...]
    兼容：
      - JSON 列表，或 {'items':[...]} 结构
      - 你的中文“（实体，字段，级别）”串
      - 其他文本则兜底为一个实体（P）
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()

    # 1) 尝试 JSON
    if s.startswith("[") or s.startswith("{"):
        try:
            obj = json.loads(s)
            arr = obj.get("items") if isinstance(obj, dict) and "items" in obj else obj
            out = []
            if isinstance(arr, list):
                for it in arr:
                    if isinstance(it, dict):
                        d = _coerce_item_dict(it)
                        if d: out.append(d)
                    elif isinstance(it, str):
                        out.extend(_parse_tuple_string(it))
            elif isinstance(arr, dict):
                d = _coerce_item_dict(arr)
                if d: out.append(d)
            if out:
                return out
        except Exception:
            pass

    # 2) 尝试“（实体，字段，级别）”串
    t_out = _parse_tuple_string(s)
    if t_out:
        return t_out

    # 3) 兜底：把整格文本当作一个实体
    return [{"P": normalize_text(s), "C": "", "L": 0}]

# =========================
# 工具：集合与匹配
# =========================
def dedup_entities(items: Iterable[Dict]) -> Set[str]:
    return {normalize_text(d.get("P","")) for d in items if normalize_text(d.get("P",""))}

def dedup_tuples(items: Iterable[Dict]) -> Set[Tuple[str, str, int]]:
    out = set()
    for d in items:
        P = normalize_text(d.get("P",""))
        C = normalize_text(d.get("C",""))
        try:
            L = int(d.get("L", 0))
        except Exception:
            L = 0
        out.add((P, C, L))
    return out

def build_pc_match(gold_items: Iterable[Dict], pred_items: Iterable[Dict]):
    """按 (P,C) 严格匹配，返回匹配集合与对应等级映射"""
    gold_pc, pred_pc = {}, {}
    for P, C, L in dedup_tuples(gold_items):
        gold_pc[(P, C)] = L
    for P, C, L in dedup_tuples(pred_items):
        pred_pc[(P, C)] = L
    M = set(gold_pc.keys()) & set(pred_pc.keys())
    return M, gold_pc, pred_pc

# =========================
# 指标 1：MCIF（实体体量）
# =========================
def mcif_entity(samples: List[Dict], eps: float = 1e-8) -> float:
    vals = []
    for s in samples:
        gE = len(dedup_entities(s.get("gold", [])))
        pE = len(dedup_entities(s.get("pred", [])))
        vals.append((pE + eps) / (gE + eps))
    return sum(vals) / max(len(vals), 1)

# =========================
# 指标 2：MCCR（类别合规率）
# 只看预测条目类别是否在 schema 内
# =========================
DEFAULT_SCHEMA = {
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
}

def mccr(samples: List[Dict], allowed_categories: Set[str], eps: float = 1e-8) -> float:
    vals = []
    for s in samples:
        pred = dedup_tuples(s.get("pred", []))
        if not pred:
            vals.append(1.0)  # 无预测→视为全合规（也可改 0.0）
            continue
        compliant = sum(1 for (_P, C, _L) in pred if C in allowed_categories)
        vals.append((compliant + eps) / (len(pred) + eps))
    return sum(vals) / max(len(vals), 1)

# =========================
# 指标 3：MSGQ（3–5 级分级一致性）
# micro：跨样本合并；macro：先样本内计算再取均值
# mode='sym'：金标或预测任一侧为 3–5（与文稿一致）
# =========================
def msgq(samples: List[Dict], mode: str = "sym", average: str = "macro", eps: float = 0.0) -> float:
    assert average in {"macro","micro"}
    per_sample = []

    micro_hit, micro_den = 0, 0

    for s in samples:
        M, gL, pL = build_pc_match(s.get("gold", []), s.get("pred", []))
        if not M:
            if average == "macro":
                # 无匹配：按约定跳过该样本
                pass
            continue

        # 选 S_i
        S = set()
        for pc in M:
            Lg = gL.get(pc, 0)
            Lp = pL.get(pc, 0)
            if mode == "gold":
                if Lg in {3,4,5}:
                    S.add(pc)
            else:  # "sym"
                if (Lg in {3,4,5}) or (Lp in {3,4,5}):
                    S.add(pc)

        if not S:
            if average == "macro":
                # 无 3-5 样本：跳过
                pass
            continue

        hit = sum(1 for pc in S if gL.get(pc, -999) == pL.get(pc, -1000))

        if average == "micro":
            micro_hit += hit
            micro_den += len(S)
        else:
            per_sample.append((hit + eps) / (len(S) + eps))

    if average == "micro":
        return (micro_hit + eps) / (micro_den + eps) if micro_den > 0 else 1.0
    else:
        return sum(per_sample) / max(len(per_sample), 1)

# =========================
# 指标 4：Macro-F1（max level 1–5）
# =========================
def _max_level(items: Iterable[Dict]) -> int:
    levels = []
    for d in items:
        try:
            v = int(d.get("L", 0))
        except Exception:
            continue
        if 1 <= v <= 5:
            levels.append(v)
    return max(levels) if levels else 0

def micro_f1_maxlevel(samples, missing_policy: str = "floor1") -> float:
    """
    Micro-F1 for the per-record max-level (1..5) task.
    单标签多分类下，micro-F1 等价于整体准确率（Acc），这里按通用公式实现：
      P_micro = sum_c TP_c / sum_c (TP_c + FP_c)
      R_micro = sum_c TP_c / sum_c (TP_c + FN_c)
      F1_micro = 2 P_micro R_micro / (P_micro + R_micro)
    """
    from collections import Counter

    def _max_level(items):
        levels = []
        for d in items:
            try:
                v = int(d.get("L", 0))
            except Exception:
                continue
            if 1 <= v <= 5:
                levels.append(v)
        return max(levels) if levels else 0

    y_true, y_pred = [], []
    for s in samples:
        g = _max_level(s.get("gold", []))
        p = _max_level(s.get("pred", []))
        if g == 0:
            continue  # 无金标等级，跳过
        if p == 0:
            if missing_policy == "skip":
                continue
            p = 1  # floor 到 1
        y_true.append(g); y_pred.append(p)

    if not y_true:
        return 0.0

    labels = [1,2,3,4,5]
    cm = Counter()           # (pred, true)
    true_count = Counter()   # true per class
    pred_count = Counter()   # pred per class

    for yt, yp in zip(y_true, y_pred):
        cm[(yp, yt)] += 1
        true_count[yt] += 1
        pred_count[yp] += 1

    TP_all = sum(cm[(c, c)] for c in labels)
    FP_all = sum(pred_count[c] - cm[(c, c)] for c in labels)
    FN_all = sum(true_count[c] - cm[(c, c)] for c in labels)

    P_micro = TP_all / (TP_all + FP_all) if (TP_all + FP_all) > 0 else 0.0
    R_micro = TP_all / (TP_all + FN_all) if (TP_all + FN_all) > 0 else 0.0
    return (2 * P_micro * R_micro / (P_micro + R_micro)) if (P_micro + R_micro) > 0 else 0.0


# =========================
# 主流程
# =========================
def main():
    ap = argparse.ArgumentParser(description="Compute four metrics from Excel: MCIF_entity, MCCR, MSGQ, Macro-F1(max level).")
    ap.add_argument("--file", required=True, help="Path to Excel file")
    ap.add_argument("--sheet", default="0", help="Sheet name or index (default 0)")
    ap.add_argument("--gold-col", default="Benchmark_output", help="Gold column name (default: Benchmark_output)")
    ap.add_argument("--pred-col", default="Output", help="Prediction column name (default: Output)")
    ap.add_argument("--schema-file", default="", help="Optional schema file (txt with one category per line, or JSON array)")
    ap.add_argument("--msgq-mode", default="sym", choices=["sym","gold"], help="MSGQ set selection (default: sym)")
    ap.add_argument("--msgq-avg", default="macro", choices=["macro","micro"], help="MSGQ averaging (default: macro)")
    ap.add_argument("--eps", type=float, default=1e-8, help="Smoothing epsilon (default 1e-8)")
    ap.add_argument("--missing-policy", default="floor1", choices=["floor1","skip"], help="Macro-F1 missing pred policy (default floor1)")
    args = ap.parse_args()

    sheet_arg = int(args.sheet) if args.sheet.isdigit() else args.sheet
    df = pd.read_excel(args.file, sheet_name=sheet_arg)

    if args.gold_col not in df.columns or args.pred_col not in df.columns:
        raise ValueError(f"缺少列：{args.gold_col} / {args.pred_col}；现有列：{list(df.columns)}")

    # schema
    schema = set(DEFAULT_SCHEMA)
    if args.schema_file:
        try:
            with open(args.schema_file, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            if txt.startswith("["):
                schema = set(json.loads(txt))
            else:
                schema = {normalize_text(x) for x in txt.splitlines() if normalize_text(x)}
        except Exception as e:
            print(f"⚠️ 读取 schema 文件失败（将使用内置 schema）：{e}")

    # 组装 samples
    samples = []
    for _, row in df.iterrows():
        gold_items = parse_cell(row[args.gold_col])
        pred_items = parse_cell(row[args.pred_col])
        samples.append({"gold": gold_items, "pred": pred_items})

    # 计算指标
    mcif_e = mcif_entity(samples, eps=args.eps)
    mccr_val = mccr(samples, allowed_categories=schema, eps=args.eps)
    msgq_val = msgq(samples, mode=args.msgq_mode, average=args.msgq_avg, eps=args.eps)
    micro_f1 = micro_f1_maxlevel(samples, missing_policy=args.missing_policy)

    # 打印
    print("======== Metrics ========")
    print(f"File              : {args.file}")
    print(f"Sheet             : {args.sheet}")
    print(f"Gold / Pred Cols  : {args.gold_col} / {args.pred_col}")
    print(f"Samples           : {len(samples)}")
    print(f"MCIF (entity)     : {mcif_e:.6f}")
    print(f"MCCR (schema rate): {mccr_val:.6f}")
    print(f"MSGQ ({args.msgq_avg},{args.msgq_mode}): {msgq_val:.6f}")
    print(f"Micro-F1 (max L)  : {micro_f1:.6f}")
    print("=========================")

if __name__ == "__main__":
    main()
