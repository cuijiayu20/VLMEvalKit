import os
import os.path as osp
import argparse
import pandas as pd
from tqdm import tqdm
import sys
import glob
import shutil

# Add vlmeval to path so we can import smp
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
try:
    from vlmeval.smp import load, decode_base64_to_image_file, ls
except ImportError:
    print("Could not import vlmeval. Please make sure you run this from the project root or vlmeval is installed.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Extract bad and good cases from VLMEvalKit results.')
    parser.add_argument('result_file', type=str, nargs='?', default=None, 
                        help='Path to the evaluation result file (e.g. *result.pkl or .xlsx). If not provided, will auto-detect.')
    parser.add_argument('--model', type=str, default='LLaVA-OneVision-7B', help='Model name for auto-detecting result file')
    parser.add_argument('--dataset', type=str, default='MMBench', help='Dataset name for auto-detecting result file')
    parser.add_argument('--out-dir', type=str, default=None, help='Output base directory (defaults to extracted_cases in the result file directory)')
    parser.add_argument('--category-col', type=str, default='category', help='Column name to group by (e.g. category, l2-category)')
    return parser.parse_args()

def find_result_file(model, dataset, base_dir='.'):
    # Try common VLMEvalKit output paths
    search_patterns = [
        osp.join(base_dir, f"outputs/{model}/{model}_{dataset}*.xlsx"),
        osp.join(base_dir, f"outputs/{model}/{model}_{dataset}*.csv"),
        osp.join(base_dir, f"outputs/{model}/{model}*{dataset}*.xlsx")
    ]
    for pattern in search_patterns:
        files = glob.glob(pattern)
        if files:
            # Prefer the one with 'result' if multiple
            res_files = [f for f in files if 'result' in f]
            return res_files[0] if res_files else files[0]
            
    # Try searching anywhere in outputs
    all_files = glob.glob(osp.join(base_dir, f"outputs/**/*{model}*{dataset}*.*"), recursive=True)
    valid_files = [f for f in all_files if f.endswith('.xlsx') or f.endswith('.pkl')]
    if valid_files:
        return valid_files[0]
        
    return None

def process_cases(cases_df, out_base_dir, case_type, args):
    if len(cases_df) == 0:
        print(f"No {case_type} found!")
        return

    cat_col = args.category_col
    if cat_col not in cases_df.columns:
        print(f"Warning: column '{cat_col}' not found. Grouping {case_type} into 'All_Categories'.")
        cases_df['category_group'] = 'All_Categories'
        cat_col = 'category_group'

    case_out_dir = osp.join(out_base_dir, case_type)
    os.makedirs(case_out_dir, exist_ok=True)
    report_md = f"# {case_type.replace('_', ' ').title()} Analysis\n\n"
    
    groups = cases_df.groupby(cat_col)
    for cat, group in groups:
        cat_name = str(cat).replace('/', '_').replace(' ', '_')
        cat_dir = osp.join(case_out_dir, cat_name)
        os.makedirs(cat_dir, exist_ok=True)
        
        report_md += f"## Category: {cat_name} ({len(group)} items)\n\n"
        
        for i, row in tqdm(group.iterrows(), total=len(group), desc=f"Processing {case_type} -> {cat_name}"):
            idx = row.get('index', i)
            
            # Dump image
            img_rel_paths = []
            if 'image' in row and pd.notna(row['image']):
                img_b64 = row['image']
                if isinstance(img_b64, str) and len(img_b64) > 64:
                    img_path = osp.join(cat_dir, f"{idx}.jpg")
                    try:
                        decode_base64_to_image_file(img_b64, img_path)
                        img_rel_paths.append(osp.relpath(img_path, out_base_dir))
                    except Exception as e:
                        pass
            elif 'image_path' in row and pd.notna(row['image_path']):
                # If images are stored as paths instead of base64
                img_path = str(row['image_path'])
                if osp.exists(img_path):
                    dest_path = osp.join(cat_dir, f"{idx}_{osp.basename(img_path)}")
                    try:
                        shutil.copy(img_path, dest_path)
                        img_rel_paths.append(osp.relpath(dest_path, out_base_dir))
                    except Exception as e:
                        pass
            
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
            
    report_path = osp.join(out_base_dir, f'{case_type}_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
        
    print(f"[{case_type}] extracted and categorized. Report generated at {report_path}")

def main():
    args = parse_args()
    
    # Auto-detect result file if not provided
    result_file = args.result_file
    if not result_file:
        print(f"Result file not provided via arguments. Auto-detecting for Model: {args.model}, Dataset: {args.dataset}...")
        project_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
        result_file = find_result_file(args.model, args.dataset, base_dir=project_root)
        
    if not result_file or not osp.exists(result_file):
        print(f"Error: Could not find a valid result file for model {args.model} and dataset {args.dataset}.")
        print("Please provide the file path manually: python extract_cases.py /path/to/result.xlsx")
        return

    print(f"Loading {result_file} ...")
    try:
        data = load(result_file)
    except Exception as e:
        print(f"Error loading {result_file}: {e}")
        return
        
    if not args.out_dir:
        args.out_dir = osp.join(osp.dirname(result_file), 'extracted_cases')
        print(f"Output directory automatically set to: {args.out_dir}")
        
    print(f"Total entries loaded: {len(data)}")
    
    # Try to load the original dataset to fetch the base64 images (since they 
    # are usually omitted in the result xlsx files to save space)
    try:
        from vlmeval.dataset import build_dataset
        dataset_obj = build_dataset(args.dataset)
        original_data = dataset_obj.data
        if not original_data.empty and 'image' in original_data.columns and 'index' in original_data.columns:
            if 'image' in data.columns:
                data = data.drop(columns=['image'])
            data = data.merge(original_data[['index', 'image']], on='index', how='left')
            print(f"Successfully fetched original images from dataset '{args.dataset}'.")
    except Exception as e:
        print(f"Warning: Could not fetch original images from dataset {args.dataset}. ({e})")
    
    # Identify correctness
    if 'hit' in data.columns:
        bad_cases = data[data['hit'] == 0].copy()
        good_cases = data[data['hit'] == 1].copy()
    else:
        # Fallback if hit is not present
        if 'prediction' in data.columns and 'answer' in data.columns:
            is_correct = data['prediction'].astype(str).str.strip().str.upper() == data['answer'].astype(str).str.strip().str.upper()
            bad_cases = data[~is_correct].copy()
            good_cases = data[is_correct].copy()
        else:
            print("Warning: Cannot find 'hit' or 'prediction'/'answer' columns to determine correctness.")
            bad_cases = pd.DataFrame()
            good_cases = data.copy()
            
    print(f"Categorizing {len(bad_cases)} bad cases and {len(good_cases)} good cases based on column '{args.category_col}'.")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    if len(bad_cases) > 0:
        process_cases(bad_cases, args.out_dir, "bad_cases", args)
    if len(good_cases) > 0:
        process_cases(good_cases, args.out_dir, "good_cases", args)

if __name__ == '__main__':
    main()
