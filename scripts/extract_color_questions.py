"""
从 MMBench 评测结果中提取所有颜色相关的问题，并统计正确率。
输出到 outputs/color_analysis/ 目录。
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

RESULT_FILE = os.path.join(
    os.path.dirname(__file__), '..', 'outputs', 'llava-onevision-qwen2-7b-ov-hf',
    'T20260322_G26fd4a17',
    'llava-onevision-qwen2-7b-ov-hf_MMBench_DEV_EN_openai_result.xlsx'
)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'color_analysis')

# 颜色关键词（问题或选项中包含这些词即视为"颜色相关"）
COLOR_KEYWORDS = [
    'color', 'colour', 'colored', 'coloured',
    'red', 'blue', 'green', 'yellow', 'black', 'white',
    'orange', 'purple', 'pink', 'gray', 'grey', 'brown',
    'golden', 'silver', 'cyan', 'magenta', 'beige', 'maroon',
    'navy', 'teal', 'violet', 'tan', 'khaki', 'crimson',
]


def is_color_related(row):
    """判断一行是否是颜色相关的题目"""
    # 检查问题
    q = str(row.get('question', '')).lower()
    if any(kw in q for kw in ['color', 'colour']):
        return True

    # 检查选项是否都是颜色词
    options = []
    for col in ['A', 'B', 'C', 'D']:
        if col in row and pd.notna(row[col]):
            options.append(str(row[col]).lower().strip())

    if options:
        color_option_count = sum(
            1 for opt in options
            if any(kw in opt for kw in COLOR_KEYWORDS)
        )
        # 如果超过一半的选项包含颜色词
        if color_option_count >= len(options) * 0.5:
            return True

    return False


def main():
    print(f'加载数据: {RESULT_FILE}')
    df = pd.read_excel(RESULT_FILE)
    print(f'总题数: {len(df)}')

    # 筛选颜色相关题目
    color_mask = df.apply(is_color_related, axis=1)
    color_df = df[color_mask].copy()
    print(f'颜色相关题目: {len(color_df)}')
    print(f'正确: {int(color_df["hit"].sum())}, 错误: {int((color_df["hit"] == 0).sum())}')
    print(f'正确率: {color_df["hit"].mean():.2%}')

    os.makedirs(OUT_DIR, exist_ok=True)

    # ===== 1. 导出详细数据到 CSV =====
    export_cols = ['index', 'question', 'hint', 'A', 'B', 'C', 'D',
                   'answer', 'category', 'l2-category', 'prediction', 'hit']
    available_cols = [c for c in export_cols if c in color_df.columns]
    csv_path = os.path.join(OUT_DIR, 'color_questions_7b.csv')
    color_df[available_cols].to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'\n详细数据已导出: {csv_path}')

    # ===== 2. 生成 Markdown 分析报告 =====
    md_path = os.path.join(OUT_DIR, 'color_analysis_7b.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# MMBench 颜色识别题目分析 (LLaVA-OneVision-7B)\n\n')

        # 总览
        total = len(color_df)
        correct = int(color_df['hit'].sum())
        wrong = total - correct
        f.write(f'## 总览\n\n')
        f.write(f'- 颜色相关题目总数: **{total}**\n')
        f.write(f'- 正确: **{correct}** ({correct/total:.1%})\n')
        f.write(f'- 错误: **{wrong}** ({wrong/total:.1%})\n\n')

        # 按 category 分组
        f.write('## 按 Category 分布\n\n')
        f.write('| Category | 总数 | 正确 | 错误 | 正确率 |\n')
        f.write('|----------|------|------|------|--------|\n')
        cat_stats = color_df.groupby('category')['hit'].agg(['count', 'sum'])
        cat_stats.columns = ['total', 'correct']
        cat_stats['wrong'] = cat_stats['total'] - cat_stats['correct']
        cat_stats['acc'] = cat_stats['correct'] / cat_stats['total']
        cat_stats = cat_stats.sort_values('total', ascending=False)
        for cat, row in cat_stats.iterrows():
            f.write(f'| {cat} | {int(row["total"])} | {int(row["correct"])} | {int(row["wrong"])} | {row["acc"]:.1%} |\n')
        f.write('\n')

        # ===== 3. 错误题目详情 =====
        wrong_df = color_df[color_df['hit'] == 0].copy()
        f.write(f'## 错误题目详情 ({len(wrong_df)} 题)\n\n')

        for i, row in wrong_df.iterrows():
            idx = row.get('index', i)
            f.write(f'### [{idx}] {row.get("category", "")}\n\n')

            hint = row.get('hint', '')
            if pd.notna(hint) and str(hint).strip():
                f.write(f'**Hint:** {hint}\n\n')

            f.write(f'**Question:** {row.get("question", "")}\n\n')

            # 选项
            options_text = ''
            for col in ['A', 'B', 'C', 'D']:
                if col in row and pd.notna(row[col]):
                    marker = '✅' if str(row.get('answer', '')).strip().upper() == col else ''
                    marker_wrong = '❌' if str(row.get('prediction', '')).strip().upper() == col else ''
                    options_text += f'- **{col}.** {row[col]} {marker} {marker_wrong}\n'

            f.write(f'**选项:**\n{options_text}\n')
            f.write(f'**正确答案:** {row.get("answer", "")}\n\n')
            f.write(f'**模型预测:** {row.get("prediction", "")}\n\n')
            f.write('---\n\n')

        # ===== 4. 正确题目列表（简要） =====
        correct_df = color_df[color_df['hit'] == 1].copy()
        f.write(f'## 正确题目列表 ({len(correct_df)} 题)\n\n')
        f.write('| Index | Category | Question (截取) | GT | Pred |\n')
        f.write('|-------|----------|----------------|-----|------|\n')
        for i, row in correct_df.iterrows():
            q = str(row.get('question', ''))[:60].replace('|', '\\|')
            f.write(f'| {row.get("index", i)} | {row.get("category", "")} | {q} | {row.get("answer", "")} | {row.get("prediction", "")} |\n')
        f.write('\n')

    print(f'分析报告已导出: {md_path}')

    # ===== 额外：打印错误题目摘要 =====
    print(f'\n===== 错误题目摘要 ({len(wrong_df)} 题) =====')
    for i, row in wrong_df.iterrows():
        idx = row.get('index', i)
        q = str(row.get('question', ''))[:60]
        gt = row.get('answer', '?')
        pred = row.get('prediction', '?')
        cat = row.get('category', '')
        print(f'  [{idx}] {cat} | Q: {q} | GT={gt} Pred={pred}')


if __name__ == '__main__':
    main()
