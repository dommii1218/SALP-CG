#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse HAOdf-like TXT blocks into a flat table and export CSV/XLSX.

Input format example:
id=1
https://www.haodf.com/doctorteam/flow_team_6477251151.htm

Doctor faculty
广东省妇幼保健院  放射介入科  

Description
... (free text until next "id=" or EOF)

Usage
-----
python parse_haodf_txt.py input.txt
python parse_haodf_txt.py /path/to/folder_with_txts  # will parse all *.txt in the folder
python parse_haodf_txt.py input.txt --csv out.csv --xlsx out.xlsx

Notes
-----
- Hospital/Faculty are split by TWO OR MORE SPACES if present; otherwise we fall back to "last token = Faculty, the rest = Hospital".
- Everything from the first "Description" until the next "id=" (or file end) is kept verbatim in the Description column,
  including Dialogue/Diagnosis blocks if they are inside that span.
"""
import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd

REC_PATTERN = re.compile(
    r'id\s*=\s*(\d+)\s*'                            # 1) ID
    r'(https?://[^\s]+)\s*'                         # 2) URL
    r'(?:Doctor\s+faculty)\s*'                      # literal header (case insensitive handled below)
    r'(.*?)\s*'                                     # 3) hospital + faculty block (one line typically)
    r'(?:Description)\s*'                           # "Description" header
    r'(.*?)(?=\n(?:id\s*=)|\Z)',                    # 4) description until next id= or EOF
    re.S | re.I
)

def split_hospital_faculty(hosp_fac_raw: str):
    """Split 'Hospital  Faculty' (two or more spaces) safely; fallback to last token heuristic."""
    # Normalize newlines and spaces
    line = hosp_fac_raw.replace('\r', ' ').strip()
    # Prefer to split on 2+ spaces or tab
    parts = re.split(r'[ \t]{2,}', line)
    if len(parts) >= 2:
        hospital = parts[0].strip()
        faculty = parts[1].strip()
        return hospital, faculty
    # Fallback: last whitespace-separated token is Faculty
    tokens = line.split()
    if len(tokens) >= 2:
        hospital = ''.join(tokens[:-1])
        faculty = tokens[-1]
    else:
        hospital = line
        faculty = ''
    return hospital, faculty

def parse_text(text: str) -> List[Dict]:
    rows: List[Dict] = []
    for m in REC_PATTERN.finditer(text):
        rec_id = int(m.group(1).strip())
        url = m.group(2).strip()
        hosp_fac = m.group(3).strip()
        desc = m.group(4).strip()
        hospital, faculty = split_hospital_faculty(hosp_fac)
        rows.append({
            "ID": rec_id,
            "url": url,
            "Hospital": hospital,
            "Faculty": faculty,
            "Description": desc
        })
    return rows

def read_file_text(path: Path, encoding: str = "utf-8") -> str:
    return path.read_text(encoding=encoding, errors='ignore')

def main():
    ap = argparse.ArgumentParser(description="Parse HAOdf-like TXT into CSV/XLSX.")
    ap.add_argument("input_path", type=str, help="TXT file path or a folder containing *.txt files")
    ap.add_argument("--csv", type=str, default=None, help="Output CSV path (default: alongside input)")
    ap.add_argument("--xlsx", type=str, default=None, help="Output XLSX path (default: alongside input)")
    ap.add_argument("--sheet", type=str, default="data", help="Sheet name for XLSX (default: data)")
    ap.add_argument("--encoding", type=str, default="utf-8", help="File encoding (default: utf-8)")
    args = ap.parse_args()

    p = Path(args.input_path)
    all_rows: List[Dict] = []

    if p.is_dir():
        txt_files = sorted(list(p.glob("*.txt")))
        if not txt_files:
            print(f"No *.txt files found in folder: {p}")
            sys.exit(1)
        for f in txt_files:
            text = read_file_text(f, args.encoding)
            recs = parse_text(text)
            if not recs:
                print(f"[WARN] No records parsed from {f}")
            all_rows.extend(recs)
    else:
        if not p.exists():
            print(f"Input path not found: {p}")
            sys.exit(1)
        text = read_file_text(p, args.encoding)
        all_rows = parse_text(text)

    if not all_rows:
        print("No records parsed. Please check the input format (id=, Doctor faculty, Description).")
        sys.exit(2)

    df = pd.DataFrame(all_rows, columns=["ID","url","Hospital","Faculty","Description"]).sort_values("ID").reset_index(drop=True)

    # Default outputs
    if args.csv is None:
        out_csv = (p if p.is_file() else p / "parsed").with_suffix(".csv")
    else:
        out_csv = Path(args.csv)

    if args.xlsx is None:
        out_xlsx = (p if p.is_file() else p / "parsed").with_suffix(".xlsx")
    else:
        out_xlsx = Path(args.xlsx)

    # Ensure parent directories exist
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=args.sheet)

    print(f"Saved CSV : {out_csv}")
    print(f"Saved XLSX: {out_xlsx}")

if __name__ == "__main__":
    main()
