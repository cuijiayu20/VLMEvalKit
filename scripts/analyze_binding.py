"""
分析 MMBench 评测结果中涉及"语言-物体绑定"的题目。
判断 LLaVA-OneVision-7B 是否存在无法将语言描述和精确物体对象绑定的问题。

分析维度：
  - 单物体属性识别 vs 多物体属性绑定 vs 跨物体对比
  - 需要空间定位的题目 vs 不需要的
  - 需要关系推理的题目
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


# ===== 绑定级别分类器 =====

# 关键词模式：需要将属性绑定到特定物体
BINDING_PATTERNS = {
    # 指代特定物体并问属性（需要定位+绑定）
    'specific_object_attribute': [
        r'what (?:is|are) the (?:color|shape|size|material) of the',
        r'what color is the',
        r'the .+ (?:thing|object|sphere|cube|cylinder|block|ball) .+ what (?:color|shape)',
        r'what (?:color|shape) (?:is|does) the .+ (?:have|has)',
        r'the .+ has what color',
        r'what is the color of the .+ (?:that|which)',
    ],
    # 同类物体对比（需要分别绑定两个物体的属性）
    'cross_object_comparison': [
        r'are the two .+ the same',
        r'same color',
        r'same shape',
        r'same size',
        r'are .+ the same .+ in the',
        r'which .+ is (?:bigger|smaller|larger|taller|shorter)',
        r'comparison picture',
        r'left and right .+ same',
        r'upper and lower .+ same',
    ],
    # 空间-物体绑定（通过位置指代物体）
    'spatial_binding': [
        r'(?:left|right|top|bottom|above|below|behind|front|near|next to) .+ (?:color|shape|what)',
        r'what .+ (?:left|right|top|bottom|above|below) .+ (?:of|side)',
        r'the .+ on the (?:left|right)',
        r'the .+ (?:in front of|behind|above|below|next to)',
    ],
    # 关系推理中涉及物体属性
    'relation_with_attribute': [
        r'(?:red|blue|green|yellow|black|white|purple|pink|orange|gray|cyan|brown) .+ '
        r'(?:above|below|left|right|behind|front|near)',
        r'(?:above|below|left|right|behind|front|near) .+ '
        r'(?:red|blue|green|yellow|black|white|purple|pink|orange|gray|cyan|brown)',
        r'a (?:red|blue|green|yellow|black|white) .+ is (?:above|below|left|right)',
    ],
}


def classify_binding_level(row):
    """给每道题分类绑定需求级别"""
    q = str(row.get('question', '')).lower()
    options_str = ' '.join(str(row.get(c, '')) for c in ['A', 'B', 'C', 'D'] if pd.notna(row.get(c)))
    options_lower = options_str.lower()
    cat = str(row.get('category', '')).lower()

    levels = []

    for level, patterns in BINDING_PATTERNS.items():
        for p in patterns:
            if re.search(p, q) or re.search(p, options_lower):
                levels.append(level)
                break

    # 基于 MMBench category 的补充分类
    if cat == 'attribute_comparison':
        if 'cross_object_comparison' not in levels:
            levels.append('cross_object_comparison')
    if cat == 'attribute_recognition':
        if not levels:
            levels.append('specific_object_attribute')
    if cat == 'spatial_relationship':
        if 'spatial_binding' not in levels:
            levels.append('spatial_binding')
    if cat == 'object_localization':
        if 'spatial_binding' not in levels:
            levels.append('spatial_binding')

    if not levels:
        levels.append('no_binding_required')

    return levels


def main():
    print(f'加载数据: {RESULT_FILE}')
    df = pd.read_excel(RESULT_FILE)
    print(f'总题数: {len(df)}')

    # 分类每道题的绑定需求
    df['binding_levels'] = df.apply(classify_binding_level, axis=1)

    # 展开（一道题可能属于多个级别）
    rows = []
    for _, row in df.iterrows():
        for level in row['binding_levels']:
            rows.append({
                'index': row['index'],
                'question': row['question'],
                'category': row['category'],
                'l2_category': row.get('l2-category', ''),
                'answer': row.get('answer', ''),
                'prediction': row.get('prediction', ''),
                'hit': row['hit'],
                'binding_level': level,
            })
    expanded = pd.DataFrame(rows)

    os.makedirs(OUT_DIR, exist_ok=True)

    # ===== 统计 =====
    print('\n===== 绑定级别统计 =====')
    stats = expanded.groupby('binding_level')['hit'].agg(['count', 'sum', 'mean'])
    stats.columns = ['total', 'correct', 'accuracy']
    stats['wrong'] = stats['total'] - stats['correct']
    stats = stats.sort_values('accuracy')
    print(stats.to_string())

    # ===== 生成报告 =====
    md_path = os.path.join(OUT_DIR, 'binding_analysis_7b.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# 语言-物体绑定能力分析 (LLaVA-OneVision-7B)\n\n')
        f.write('> 假设：LLaVA 不能将语言和精确的物体对象绑定\n\n')

        # 总体统计
        f.write('## 一、按绑定需求级别统计\n\n')
        f.write('| 绑定级别 | 说明 | 题数 | 正确 | 错误 | 正确率 |\n')
        f.write('|---------|------|------|------|------|--------|\n')

        level_desc = {
            'no_binding_required': '无需物体绑定（场景/主题/常识）',
            'specific_object_attribute': '指定物体问属性（单物体绑定）',
            'spatial_binding': '空间定位+物体识别',
            'cross_object_comparison': '跨物体属性对比（需同时绑定两个）',
            'relation_with_attribute': '属性+空间关系组合推理',
        }

        for level, row in stats.iterrows():
            desc = level_desc.get(level, level)
            f.write(f'| {level} | {desc} | {int(row["total"])} | '
                    f'{int(row["correct"])} | {int(row["wrong"])} | '
                    f'{row["accuracy"]:.1%} |\n')

        f.write('\n')

        # 关键对比
        binding_levels = ['specific_object_attribute', 'cross_object_comparison',
                         'spatial_binding', 'relation_with_attribute']
        binding_data = expanded[expanded['binding_level'].isin(binding_levels)]
        no_binding_data = expanded[expanded['binding_level'] == 'no_binding_required']

        if len(binding_data) > 0 and len(no_binding_data) > 0:
            binding_acc = binding_data['hit'].mean()
            no_binding_acc = no_binding_data['hit'].mean()
            gap = no_binding_acc - binding_acc

            f.write('## 二、关键对比\n\n')
            f.write(f'| 对比项 | 题数 | 正确率 |\n')
            f.write(f'|--------|------|--------|\n')
            f.write(f'| 不需要物体绑定的题目 | {len(no_binding_data)} | {no_binding_acc:.1%} |\n')
            f.write(f'| **需要物体绑定的题目** | **{len(binding_data)}** | **{binding_acc:.1%}** |\n')
            f.write(f'| 正确率差距 | — | **{gap:.1%}** |\n\n')

        # 各绑定级别的错误题目
        f.write('## 三、需要物体绑定但回答错误的题目\n\n')

        for level in binding_levels:
            level_data = expanded[(expanded['binding_level'] == level) & (expanded['hit'] == 0)]
            if len(level_data) == 0:
                continue

            desc = level_desc.get(level, level)
            f.write(f'### {desc} (错误 {len(level_data)} 题)\n\n')
            f.write('| Index | Category | Question (截取) | GT | Pred |\n')
            f.write('|-------|----------|----------------|-----|------|\n')

            for _, row in level_data.iterrows():
                q = str(row['question'])[:70].replace('|', '\\|').replace('\n', ' ')
                f.write(f'| {row["index"]} | {row["category"]} | {q} | {row["answer"]} | {row["prediction"]} |\n')

            f.write('\n')

        # 结论
        f.write('## 四、结论\n\n')
        if len(binding_data) > 0 and len(no_binding_data) > 0:
            if gap > 0.1:
                f.write(f'**数据支持假设。** 需要物体绑定的题目正确率（{binding_acc:.1%}）'
                        f'显著低于不需要绑定的题目（{no_binding_acc:.1%}），差距 {gap:.1%}。\n\n')
                f.write('具体表现：\n\n')
            elif gap > 0.05:
                f.write(f'**数据部分支持假设。** 物体绑定题目正确率（{binding_acc:.1%}）'
                        f'略低于非绑定题目（{no_binding_acc:.1%}），差距 {gap:.1%}。\n\n')
            else:
                f.write(f'**数据不明显支持假设。** 物体绑定题目正确率（{binding_acc:.1%}）'
                        f'与非绑定题目（{no_binding_acc:.1%}）接近，差距仅 {gap:.1%}。\n\n')

            # 按子级别给出具体分析
            for level in binding_levels:
                level_stats = stats.loc[level] if level in stats.index else None
                if level_stats is not None:
                    f.write(f'- **{level_desc[level]}**: {level_stats["accuracy"]:.1%} '
                            f'({int(level_stats["correct"])}/{int(level_stats["total"])})\n')
        f.write('\n')

    print(f'\n报告已保存: {md_path}')

    # 也导出 CSV
    csv_path = os.path.join(OUT_DIR, 'binding_questions_7b.csv')
    binding_only = expanded[expanded['binding_level'] != 'no_binding_required']
    binding_only.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'绑定题目 CSV 已保存: {csv_path}')


if __name__ == '__main__':
    main()
