import argparse
import csv
import sys
import tempfile
import os
from collections import OrderedDict
from typing import Iterable

#!/usr/bin/env python3
# /home/nova/classifier/dedup.py
"""
CSV 去重工具
用法示例:
    python dedup.py -i input.csv -o dedup.csv
    python dedup.py -i input.csv --cols id,name --ignore-case --strip
    python dedup.py -i input.csv --keep last
"""


def normalize_value(v: str, ignore_case: bool, strip: bool) -> str:
        if v is None:
                return ""
        if strip:
                v = v.strip()
        if ignore_case:
                v = v.lower()
        return v

def row_key_from_dict(row: dict, cols: Iterable[str], ignore_case: bool, strip: bool) -> tuple:
        if cols:
                return tuple(normalize_value(row.get(c, ""), ignore_case, strip) for c in cols)
        # use full row values in header order
        return tuple(normalize_value(v, ignore_case, strip) for v in row.values())

def dedup_keep_first(in_path: str, out_path: str, cols, delimiter: str, ignore_case: bool, strip: bool):
        with open(in_path, newline='', encoding='utf-8-sig') as fin, \
                 open(out_path, 'w', newline='', encoding='utf-8') as fout:
                reader = csv.DictReader(fin, delimiter=delimiter)
                if reader.fieldnames is None:
                        # empty file
                        return
                writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter=delimiter)
                writer.writeheader()
                seen = set()
                for row in reader:
                        key = row_key_from_dict(row, cols, ignore_case, strip)
                        if key in seen:
                                continue
                        seen.add(key)
                        writer.writerow(row)

def dedup_keep_last(in_path: str, out_path: str, cols, delimiter: str, ignore_case: bool, strip: bool):
        # keep the last occurrence: read whole file into OrderedDict mapping key -> row (last wins)
        with open(in_path, newline='', encoding='utf-8-sig') as fin:
                reader = csv.DictReader(fin, delimiter=delimiter)
                if reader.fieldnames is None:
                        return
                ordered = OrderedDict()
                for row in reader:
                        key = row_key_from_dict(row, cols, ignore_case, strip)
                        ordered[key] = row  # override previous, so last remains
                # write out
                with open(out_path, 'w', newline='', encoding='utf-8') as fout:
                        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter=delimiter)
                        writer.writeheader()
                        for row in ordered.values():
                                writer.writerow(row)

def parse_cols_arg(cols_arg: str):
        if not cols_arg:
                return None
        # split by comma and strip whitespace
        return [c.strip() for c in cols_arg.split(',') if c.strip()]

def main():
        parser = argparse.ArgumentParser(description="对 CSV 中数据去重 (按列或整行去重)")
        parser.add_argument('-i', '--input', required=True, help='输入 CSV 文件路径')
        parser.add_argument('-o', '--output', help='输出路径，若不指定则覆盖输入文件')
        parser.add_argument('--cols', help='按这些列去重，逗号分隔（列名），默认按整行去重')
        parser.add_argument('--delimiter', default=',', help='CSV 分隔符，默认为 ,')
        parser.add_argument('--ignore-case', action='store_true', help='忽略大小写比较')
        parser.add_argument('--strip', action='store_true', help='去掉字段前后空白再比较')
        parser.add_argument('--keep', choices=['first', 'last'], default='first', help='保留重复项中的 first 或 last，默认 first')
        args = parser.parse_args()

        in_path = args.input
        out_path = args.output
        cols = parse_cols_arg(args.cols)
        delimiter = args.delimiter
        ignore_case = args.ignore_case
        strip = args.strip
        keep = args.keep

        # If no output specified, write to temp file then replace input
        temp_out = None
        try:
                if out_path:
                        target = out_path
                else:
                        fd, temp_out = tempfile.mkstemp(prefix='dedup_', suffix='.csv', dir=os.path.dirname(in_path) or '.')
                        os.close(fd)
                        target = temp_out

                if keep == 'first':
                        dedup_keep_first(in_path, target, cols, delimiter, ignore_case, strip)
                else:
                        dedup_keep_last(in_path, target, cols, delimiter, ignore_case, strip)

                if not out_path:
                        # replace original file
                        os.replace(target, in_path)
                        target = in_path
        finally:
                if temp_out and os.path.exists(temp_out):
                        try:
                                os.remove(temp_out)
                        except Exception:
                                pass

if __name__ == '__main__':
        main()