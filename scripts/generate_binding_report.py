"""
整理空间定位和跨物体对比两类错误题目的详细中文文档。
提取每道题的完整信息：序号、问题、选项、正确答案、模型预测。
"""
import os
import sys
import re
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

RESULT_FILE = os.path.join(
    os.path.dirname(__file__), '..', 'outputs', 'llava-onevision-qwen2-7b-ov-hf',
    'T20260322_G26fd4a17',
    'llava-onevision-qwen2-7b-ov-hf_MMBench_DEV_EN_openai_result.xlsx'
)
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'binding_analysis')

# 跨物体对比关键词
COMPARISON_PATTERNS = [
    r'are the two .+ the same',
    r'same color', r'same shape', r'same size',
    r'comparison picture',
    r'left and right .+ same',
    r'upper and lower .+ same',
    r'which .+ is (?:bigger|smaller|larger|taller|shorter)',
]

# 空间定位关键词
SPATIAL_PATTERNS = [
    r'which (?:direction|corner)',
    r'how many .+ are',
    r'where is the',
    r'which direction is .+ facing',
    r'which direction is .+ looking',
    r'what direction is',
    r'roughly how much .+ occupied',
    r'approximately what proportion',
    r'what is the relation between',
]


def classify(row):
    q = str(row.get('question', '')).lower()
    cat = str(row.get('category', '')).lower()
    options = ' '.join(str(row.get(c, '')) for c in ['A', 'B', 'C', 'D'] if pd.notna(row.get(c))).lower()

    labels = set()
    if cat == 'attribute_comparison':
        labels.add('cross_object_comparison')
    for p in COMPARISON_PATTERNS:
        if re.search(p, q) or re.search(p, options):
            labels.add('cross_object_comparison')

    if cat in ('object_localization', 'spatial_relationship'):
        labels.add('spatial_binding')
    if cat == 'physical_relation':
        labels.add('spatial_binding')
    for p in SPATIAL_PATTERNS:
        if re.search(p, q):
            labels.add('spatial_binding')

    return labels


def format_options(row):
    parts = []
    for c in ['A', 'B', 'C', 'D']:
        if c in row and pd.notna(row[c]):
            val = str(row[c]).strip()
            gt_mark = ' ← 正确答案' if str(row.get('answer', '')).strip().upper() == c else ''
            pred_mark = ' ← 模型预测' if str(row.get('prediction', '')).strip().upper() == c else ''
            parts.append(f'  {c}. {val}{gt_mark}{pred_mark}')
    return '\n'.join(parts)


def main():
    df = pd.read_excel(RESULT_FILE)

    df['labels'] = df.apply(classify, axis=1)

    # 分别筛选
    spatial = df[df['labels'].apply(lambda s: 'spatial_binding' in s)].copy()
    comparison = df[df['labels'].apply(lambda s: 'cross_object_comparison' in s)].copy()

    os.makedirs(OUT_DIR, exist_ok=True)
    md_path = os.path.join(OUT_DIR, '绑定错误详细分析.md')

    with open(md_path, 'w', encoding='utf-8') as f:
        # ===== 标题 =====
        f.write('# LLaVA-OneVision-7B 语言-物体绑定错误详细分析\n\n')
        f.write('> 数据来源：MMBench DEV EN 评测结果\n\n')

        # =================== 跨物体属性对比 ===================
        comp_correct = comparison[comparison['hit'] == 1]
        comp_wrong = comparison[comparison['hit'] == 0]
        f.write('---\n\n')
        f.write('## 一、跨物体属性对比\n\n')
        f.write(f'- 总题数：**{len(comparison)}**\n')
        f.write(f'- 正确：**{len(comp_correct)}** ({len(comp_correct)/len(comparison):.1%})\n')
        f.write(f'- 错误：**{len(comp_wrong)}** ({len(comp_wrong)/len(comparison):.1%})\n')
        f.write(f'- 正确题目序号：{", ".join(str(x) for x in sorted(comp_correct["index"].tolist()))}\n')
        f.write(f'- 错误题目序号：{", ".join(str(x) for x in sorted(comp_wrong["index"].tolist()))}\n\n')

        # 按子类型分组
        f.write('### 1.1 错误题目详情\n\n')
        for i, row in comp_wrong.sort_values('index').iterrows():
            idx = row['index']
            hit_label = '❌ 错误'
            f.write(f'#### 题目 {idx}（{row["category"]}）{hit_label}\n\n')
            hint = row.get('hint', '')
            if pd.notna(hint) and str(hint).strip():
                f.write(f'**提示信息：** {hint}\n\n')
            f.write(f'**问题：** {row["question"]}\n\n')
            f.write(f'**选项：**\n\n{format_options(row)}\n\n')
            f.write(f'**正确答案：{row["answer"]}　　模型预测：{row["prediction"]}**\n\n')
            f.write('---\n\n')

        # 正确题目简表
        f.write('### 1.2 正确题目列表\n\n')
        f.write('| 序号 | 类别 | 问题（截取） | 正确答案 | 模型预测 |\n')
        f.write('|------|------|-------------|---------|--------|\n')
        for i, row in comp_correct.sort_values('index').iterrows():
            q = str(row['question'])[:50].replace('|', '\\|').replace('\n', ' ')
            f.write(f'| {row["index"]} | {row["category"]} | {q} | {row["answer"]} | {row["prediction"]} |\n')
        f.write('\n')

        # =================== 空间定位+物体识别 ===================
        sp_correct = spatial[spatial['hit'] == 1]
        sp_wrong = spatial[spatial['hit'] == 0]
        f.write('---\n\n')
        f.write('## 二、空间定位 + 物体识别\n\n')
        f.write(f'- 总题数：**{len(spatial)}**\n')
        f.write(f'- 正确：**{len(sp_correct)}** ({len(sp_correct)/len(spatial):.1%})\n')
        f.write(f'- 错误：**{len(sp_wrong)}** ({len(sp_wrong)/len(spatial):.1%})\n')
        f.write(f'- 正确题目序号：{", ".join(str(x) for x in sorted(sp_correct["index"].tolist()))}\n')
        f.write(f'- 错误题目序号：{", ".join(str(x) for x in sorted(sp_wrong["index"].tolist()))}\n\n')

        # 按 category 子分类
        sp_cat_stats = spatial.groupby('category')['hit'].agg(['count', 'sum', 'mean'])
        sp_cat_stats.columns = ['总数', '正确', '正确率']
        sp_cat_stats['错误'] = sp_cat_stats['总数'] - sp_cat_stats['正确']
        sp_cat_stats = sp_cat_stats.sort_values('正确率')

        f.write('### 2.0 按子类别统计\n\n')
        f.write('| 子类别 | 总数 | 正确 | 错误 | 正确率 |\n')
        f.write('|--------|------|------|------|--------|\n')
        for cat, row in sp_cat_stats.iterrows():
            f.write(f'| {cat} | {int(row["总数"])} | {int(row["正确"])} | {int(row["错误"])} | {row["正确率"]:.1%} |\n')
        f.write('\n')

        f.write('### 2.1 错误题目详情\n\n')
        for i, row in sp_wrong.sort_values('index').iterrows():
            idx = row['index']
            f.write(f'#### 题目 {idx}（{row["category"]}）❌ 错误\n\n')
            hint = row.get('hint', '')
            if pd.notna(hint) and str(hint).strip():
                f.write(f'**提示信息：** {hint}\n\n')
            f.write(f'**问题：** {row["question"]}\n\n')
            f.write(f'**选项：**\n\n{format_options(row)}\n\n')
            f.write(f'**正确答案：{row["answer"]}　　模型预测：{row["prediction"]}**\n\n')
            f.write('---\n\n')

        # 正确题目简表
        f.write('### 2.2 正确题目列表\n\n')
        f.write('| 序号 | 类别 | 问题（截取） | 正确答案 | 模型预测 |\n')
        f.write('|------|------|-------------|---------|--------|\n')
        for i, row in sp_correct.sort_values('index').iterrows():
            q = str(row['question'])[:50].replace('|', '\\|').replace('\n', ' ')
            f.write(f'| {row["index"]} | {row["category"]} | {q} | {row["answer"]} | {row["prediction"]} |\n')
        f.write('\n')

        # ===== 总结 =====
        f.write('---\n\n')
        f.write('## 三、序号汇总\n\n')
        f.write('### 跨物体属性对比\n\n')
        f.write(f'- **错误**（{len(comp_wrong)} 题）：{", ".join(str(x) for x in sorted(comp_wrong["index"].tolist()))}\n')
        f.write(f'- **正确**（{len(comp_correct)} 题）：{", ".join(str(x) for x in sorted(comp_correct["index"].tolist()))}\n\n')
        f.write('### 空间定位 + 物体识别\n\n')
        f.write(f'- **错误**（{len(sp_wrong)} 题）：{", ".join(str(x) for x in sorted(sp_wrong["index"].tolist()))}\n')
        f.write(f'- **正确**（{len(sp_correct)} 题）：{", ".join(str(x) for x in sorted(sp_correct["index"].tolist()))}\n\n')

    print(f'文档已保存: {md_path}')


if __name__ == '__main__':
    main()
