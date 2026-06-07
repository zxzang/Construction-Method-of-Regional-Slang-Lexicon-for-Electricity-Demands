# -*- coding: utf-8 -*-
"""Inspect the WeChat CSV data files to understand structure and content."""
import pandas as pd
import re
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for fname in ['微信群聊天数据1_脱敏.csv', '微信群聊天数据2_脱敏.csv']:
    fpath = os.path.join(DATA_DIR, fname)
    print(f"\n{'='*60}")
    print(f"File: {fname}")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(fpath, encoding='utf-8-sig', on_bad_lines='skip')
        # Clean column names
        df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]
        print(f"Columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
        
        # Find the message column
        msg_col = None
        for c in df.columns:
            if '消息' in c or '内容' in c or 'msg' in c.lower():
                msg_col = c
                break
        if msg_col is None:
            msg_col = df.columns[3] if len(df.columns) > 3 else df.columns[-1]
        
        print(f"Message column: '{msg_col}'")
        print(f"Non-null messages: {df[msg_col].notna().sum()}")
        
        # Print sample messages
        print("\nSample messages:")
        for i, val in enumerate(df[msg_col].dropna().head(15)):
            val_str = str(val)[:120]
            has_chinese = bool(re.search(r'[\u4e00-\u9fa5]', val_str))
            print(f"  [{i}] (cn={has_chinese}) {val_str}")
        
        # Count messages with Chinese characters
        cn_count = df[msg_col].astype(str).apply(lambda x: bool(re.search(r'[\u4e00-\u9fa5]', x))).sum()
        print(f"\nMessages with Chinese chars: {cn_count}/{len(df)} ({cn_count/len(df)*100:.1f}%)")
        
    except Exception as e:
        print(f"Error: {e}")
