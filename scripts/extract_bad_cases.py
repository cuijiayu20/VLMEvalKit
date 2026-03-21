import os
import os.path as osp
import argparse
import pandas as pd
from tqdm import tqdm
import sys

# Add vlmeval to path so we can import smp
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
from vlmeval.smp import load, decode_base64_to_image_file, ls

def parse_args():
    parser = argparse.ArgumentParser(description='Extract bad cases (incorrect predictions) from VLMEvalKit results.')
    parser.add_argument('result_file', type=str, help='Path to the evaluation result file (e.g. *result.pkl or .xlsx)')
    parser.add_argument('--out-dir', type=str, default='bad_cases', help='Output directory for bad cases')
    parser.add_argument('--category-col', type=str, default='category', help='Column name to group by (e.g. category, l2-category)')
    return parser.parse_args()

def main():
    args = parse_args()
    if not osp.exists(args.result_file):
        print(f"Result file {args.result_file} does not exist.")
        return

    print(f"Loading {args.result_file} ...")
    data = load(args.result_file)
    
    # Identify incorrect predictions
    if 'hit' in data.columns:
        bad_cases = data[data['hit'] == 0].copy()
    else:
        # Fallback if hit is not present
        if 'prediction' in data.columns and 'answer' in data.columns:
            bad_cases = data[data['prediction'].astype(str).str.strip().str.upper() != data['answer'].astype(str).str.strip().str.upper()].copy()
        else:
            print("Cannot find 'hit' or 'prediction'/'answer' columns to determine correctness.")
            return
            
    print(f"Total bad cases found: {len(bad_cases)} out of {len(data)}")
    if len(bad_cases) == 0:
        print("No bad cases found! The model might be 100% accurate on this subset or the criteria failed.")
        return
    
    cat_col = args.category_col
    if cat_col not in bad_cases.columns:
        print(f"Warning: column '{cat_col}' not found. Grouping all into 'All_Categories'.")
        bad_cases['category_group'] = 'All_Categories'
        cat_col = 'category_group'
        
    os.makedirs(args.out_dir, exist_ok=True)
    report_md = f"# Bad Case Analysis: {osp.basename(args.result_file)}\n\n"
    
    groups = bad_cases.groupby(cat_col)
    for cat, group in groups:
        cat_name = str(cat).replace('/', '_').replace(' ', '_')
        cat_dir = osp.join(args.out_dir, cat_name)
        os.makedirs(cat_dir, exist_ok=True)
        
        report_md += f"## Category: {cat_name} ({len(group)} errors)\n\n"
        
        for i, row in tqdm(group.iterrows(), total=len(group), desc=f"Processing {cat_name}"):
            idx = row.get('index', i)
            
            # Dump image
            img_rel_paths = []
            if 'image' in row and pd.notna(row['image']):
                img_b64 = row['image']
                if isinstance(img_b64, str) and len(img_b64) > 64:
                    img_path = osp.join(cat_dir, f"{idx}.jpg")
                    decode_base64_to_image_file(img_b64, img_path)
                    img_rel_paths.append(osp.relpath(img_path, args.out_dir))
            elif 'image_path' in row and pd.notna(row['image_path']):
                # If images are stored as paths instead of base64
                img_path = str(row['image_path'])
                if osp.exists(img_path):
                    import shutil
                    dest_path = osp.join(cat_dir, osp.basename(img_path))
                    shutil.copy(img_path, dest_path)
                    img_rel_paths.append(osp.relpath(dest_path, args.out_dir))
            
            # Options
            options = []
            for cand in ['A', 'B', 'C', 'D', 'E', 'F']:
                if cand in row and pd.notna(row[cand]):
                    options.append(f"{cand}. {row[cand]}")
            options_text = "<br>".join(options)
            
            # Format report entry
            report_md += f"### Index: {idx}\n"
            for p in img_rel_paths:
                report_md += f"<img src='{p}' width='400'/>\n"
            report_md += f"\n**Question:** {row.get('question', '')}\n\n"
            if options_text:
                report_md += f"**Options:**<br>\n{options_text}\n\n"
            report_md += f"**Ground Truth:** {row.get('answer', '')}\n\n"
            report_md += f"**Prediction:** {row.get('prediction', '')}\n\n"
            if 'log' in row and pd.notna(row['log']):
                report_md += f"<details><summary><b>Model Output Log</b></summary>\n\n```\n{row['log']}\n```\n</details>\n\n"
            report_md += "---\n\n"
            
    report_path = osp.join(args.out_dir, 'bad_case_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
        
    print(f"Bad cases extracted and report generated at {report_path}")

if __name__ == '__main__':
    main()
