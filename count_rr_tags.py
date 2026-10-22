#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm


RE_SOURCE = re.compile(
    r"<source\s*:\s*(?P<header>[^>]*)>\s*(?P<span>.*?)\s*</source>",
    flags=re.IGNORECASE | re.DOTALL,
)
RE_LEAF = re.compile(
    r"<leaf\s*:\s*(?P<header>[^>]*)>\s*(?P<span>.*?)\s*</leaf>",
    flags=re.IGNORECASE | re.DOTALL,
)

# 헤더 안에 태그가 섞인(중첩) 경우 탐지
RE_TAG_IN_HEADER = re.compile(r"<\s*(source|leaf)\b", flags=re.IGNORECASE)

# 헤더에서 'of' 이후를 대상 리스트로 간주
RE_AFTER_OF = re.compile(r"\bof\b\s*(.+)$", flags=re.IGNORECASE)

# "A and B", "A, B and C" split
RE_SPLIT = re.compile(r"\s*(?:,|and)\s*", flags=re.IGNORECASE)


def _clean_target_piece(x: str) -> str:
    x = (x or "").strip()
    x = re.sub(r'^[\s"\(\)\[\]\{\}]+', "", x)
    x = re.sub(r'[\s"\(\)\[\]\{\}\.]+$', "", x)
    return x.strip()


def count_targets_in_header(header: str) -> int:
    """
    헤더 문자열에서 대상(target) 개수를 계산.
    우선: "... of <targets>" 사용
    폴백: 전체 header를 대상으로 split
    """
    if not isinstance(header, str):
        return 0
    header = header.strip()
    if not header:
        return 0

    m = RE_AFTER_OF.search(header)
    tail = m.group(1).strip() if m else header

    if not tail:
        return 0

    parts = [p for p in RE_SPLIT.split(tail) if p and p.strip()]
    parts = [_clean_target_piece(p) for p in parts]
    parts = [p for p in parts if p]

    # 중복 제거 (case-insensitive)
    uniq = []
    seen = set()
    for p in parts:
        k = p.casefold()
        if k not in seen:
            uniq.append(p)
            seen.add(k)

    return len(uniq)


def extract_counts(rr_text: str) -> Tuple[int, int, int, int, int, int]:
    """
    Returns:
      n_source_tags, n_leaf_tags,
      n_source_header_targets, n_leaf_header_targets,
      n_bad_source_header, n_bad_leaf_header
    """
    if not isinstance(rr_text, str) or not rr_text.strip():
        return 0, 0, 0, 0, 0, 0

    source_ms = list(RE_SOURCE.finditer(rr_text))
    leaf_ms = list(RE_LEAF.finditer(rr_text))

    n_bad_source = 0
    n_source_targets = 0
    for m in source_ms:
        header = (m.group("header") or "").strip()
        if RE_TAG_IN_HEADER.search(header):
            n_bad_source += 1
        n_source_targets += count_targets_in_header(header)

    n_bad_leaf = 0
    n_leaf_targets = 0
    for m in leaf_ms:
        header = (m.group("header") or "").strip()
        if RE_TAG_IN_HEADER.search(header):
            n_bad_leaf += 1
        n_leaf_targets += count_targets_in_header(header)

    return (
        len(source_ms),
        len(leaf_ms),
        n_source_targets,
        n_leaf_targets,
        n_bad_source,
        n_bad_leaf,
    )


def parse_rr_cols(rr_cols_args: List[str]) -> List[str]:
    """
    Allow:
      --rr_col colA --rr_col colB
    and also:
      --rr_col colA,colB,colC
    """
    cols: List[str] = []
    for item in rr_cols_args:
        if not item:
            continue
        for c in item.split(","):
            c = c.strip()
            if c:
                cols.append(c)
    # unique, keep order
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Count RR tag counts + header target counts for multiple RR columns."
    )
    ap.add_argument("--input_csv", default='path_paragraphs_final3_gpt.csv')
    ap.add_argument("--output_csv", default='path_paragraphs_final3_gpt_rrcount.csv')
    ap.add_argument("--encoding", default="utf-8")

    ap.add_argument(
        "--rr_col",
        action="append",
        default=None,
        help="RR column to analyze. Repeatable, and can be comma-separated. "
             "e.g., --rr_col paragraph_rr_raw --rr_col paragraph_rr_smoothed_smooth "
             "or --rr_col paragraph_rr_raw,paragraph_rr_smoothed_smooth",
    )
    ap.add_argument(
        "--prefix",
        default="rr",
        help="Prefix for output columns (default: rr). Output names include the rr_col name too.",
    )

    args = ap.parse_args()

    rr_cols = parse_rr_cols(args.rr_col or ["paragraph_rr_raw", "paragraph_rr_smoothed_smooth"])

    df = pd.read_csv(args.input_csv, encoding=args.encoding)

    missing = [c for c in rr_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing rr_col(s): {missing}")

    # For each rr_col, compute per-row stats and add columns.
    for col in rr_cols:
        s_tags_list = []
        l_tags_list = []
        s_tgts_list = []
        l_tgts_list = []
        bad_s_list = []
        bad_l_list = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Counting: {col}", unit="row"):
            rr_text = row.get(col, "")
            s_tags, l_tags, s_tgts, l_tgts, bad_s, bad_l = extract_counts(rr_text)
            s_tags_list.append(s_tags)
            l_tags_list.append(l_tags)
            s_tgts_list.append(s_tgts)
            l_tgts_list.append(l_tgts)
            bad_s_list.append(bad_s)
            bad_l_list.append(bad_l)

        safe_col = re.sub(r"\W+", "_", col).strip("_")  # column name safe suffix

        df[f"{args.prefix}_{safe_col}_n_source_tags"] = s_tags_list
        df[f"{args.prefix}_{safe_col}_n_leaf_tags"] = l_tags_list
        df[f"{args.prefix}_{safe_col}_n_source_header_targets"] = s_tgts_list
        df[f"{args.prefix}_{safe_col}_n_leaf_header_targets"] = l_tgts_list
        df[f"{args.prefix}_{safe_col}_n_bad_source_header"] = bad_s_list
        df[f"{args.prefix}_{safe_col}_n_bad_leaf_header"] = bad_l_list

        # averages (avoid divide-by-zero)
        s_tags_series = df[f"{args.prefix}_{safe_col}_n_source_tags"].replace(0, pd.NA)
        l_tags_series = df[f"{args.prefix}_{safe_col}_n_leaf_tags"].replace(0, pd.NA)

        df[f"{args.prefix}_{safe_col}_avg_targets_per_source_tag"] = (
            df[f"{args.prefix}_{safe_col}_n_source_header_targets"] / s_tags_series
        )
        df[f"{args.prefix}_{safe_col}_avg_targets_per_leaf_tag"] = (
            df[f"{args.prefix}_{safe_col}_n_leaf_header_targets"] / l_tags_series
        )

    df.to_csv(args.output_csv, index=False, encoding=args.encoding)


if __name__ == "__main__":
    main()
