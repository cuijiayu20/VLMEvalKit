"""
纯色图测试脚本：对比 LLaVA-OneVision 0.5B vs 7B 模型在纯色图输入下的表现。

用法:
    # 测试两个模型（默认）
    python scripts/test_solid_color.py

    # 只测试 0.5B
    python scripts/test_solid_color.py --model 0.5b

    # 只测试 7B
    python scripts/test_solid_color.py --model 7b

    # 自定义图像尺寸和输出目录
    python scripts/test_solid_color.py --size 512 --out-dir results/solid_color_test
"""

import os
import sys
import argparse
import json
from datetime import datetime
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ===== 纯色图配置 =====
SOLID_COLORS = {
    'red':    (255, 0, 0),
    'green':  (0, 255, 0),
    'blue':   (0, 0, 255),
    'white':  (255, 255, 255),
    'black':  (0, 0, 0),
    'yellow': (255, 255, 0),
    'gray':   (128, 128, 128),
}

# ===== 测试问题 =====
QUESTIONS = [
    "What is shown in this image?",
    "What color is this image?",
    "Describe this image in detail.",
    "How many objects are in this image?",
    "Is there any text in this image?",
]

# ===== 模型名称映射 =====
MODEL_KEYS = {
    '0.5b': 'llava-onevision-qwen2-0.5b-ov-hf',
    '7b':   'llava-onevision-qwen2-7b-ov-hf',
}


def generate_solid_images(out_dir, size=384):
    """生成纯色图并保存到指定目录，返回 {颜色名: 图片路径} 字典"""
    img_dir = os.path.join(out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    image_paths = {}
    for name, rgb in SOLID_COLORS.items():
        path = os.path.join(img_dir, f'solid_{name}.png')
        img = Image.new('RGB', (size, size), color=rgb)
        img.save(path)
        image_paths[name] = path
        print(f'  生成纯色图: {name} {rgb} -> {path}')

    return image_paths


def load_model(model_key):
    """加载模型，返回模型实例"""
    from vlmeval.config import supported_VLM

    model_name = MODEL_KEYS[model_key]
    print(f'\n正在加载模型: {model_name} ...')
    model = supported_VLM[model_name]()
    print(f'模型 {model_name} 加载完成。')
    return model


def run_test(model, image_paths, model_key):
    """对所有纯色图 x 所有问题运行推理，返回结果列表"""
    results = []
    total = len(image_paths) * len(QUESTIONS)
    count = 0

    for color_name, img_path in image_paths.items():
        for question in QUESTIONS:
            count += 1
            print(f'  [{count}/{total}] {model_key} | {color_name} | {question[:40]}...')

            # 构造 message（VLMEvalKit 标准格式）
            message = [
                dict(type='image', value=img_path),
                dict(type='text', value=question),
            ]

            try:
                response = model.generate(message)
            except Exception as e:
                response = f'[ERROR] {e}'

            results.append({
                'model': MODEL_KEYS[model_key],
                'model_key': model_key,
                'color': color_name,
                'rgb': list(SOLID_COLORS[color_name]),
                'question': question,
                'response': response,
            })
            print(f'    -> {response[:100]}')

    return results


def save_results(results, out_dir):
    """保存结果为 JSON 和 Markdown 报告"""
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存 JSON
    json_path = os.path.join(out_dir, f'results_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\nJSON 结果已保存: {json_path}')

    # 生成 Markdown 报告
    md_path = os.path.join(out_dir, f'report_{timestamp}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# 纯色图测试报告\n\n')
        f.write(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')

        # 按问题分组对比
        for question in QUESTIONS:
            f.write(f'## 问题: {question}\n\n')
            f.write('| 颜色 | RGB | 0.5B 回答 | 7B 回答 |\n')
            f.write('|------|-----|-----------|--------|\n')

            for color_name in SOLID_COLORS:
                row_05b = ''
                row_7b = ''
                for r in results:
                    if r['color'] == color_name and r['question'] == question:
                        resp = r['response'].replace('\n', ' ').replace('|', '\\|')
                        if r['model_key'] == '0.5b':
                            row_05b = resp
                        elif r['model_key'] == '7b':
                            row_7b = resp

                rgb_str = str(tuple(SOLID_COLORS[color_name]))
                f.write(f'| {color_name} | {rgb_str} | {row_05b} | {row_7b} |\n')

            f.write('\n')

    print(f'Markdown 报告已保存: {md_path}')
    return json_path, md_path


def parse_args():
    parser = argparse.ArgumentParser(description='纯色图测试: LLaVA-OneVision 0.5B vs 7B')
    parser.add_argument('--model', type=str, default='both', choices=['0.5b', '7b', 'both'], help='要测试的模型 (默认: both)')
    parser.add_argument('--size', type=int, default=384, help='纯色图尺寸 (默认: 384)')
    parser.add_argument('--out-dir', type=str, default=None, help='输出目录 (默认: outputs/solid_color_test)')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.out_dir is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        args.out_dir = os.path.join(project_root, 'outputs', 'solid_color_test')

    print('=' * 60)
    print('纯色图测试: LLaVA-OneVision 0.5B vs 7B')
    print('=' * 60)

    # 1. 生成纯色图
    print('\n[1/3] 生成纯色图...')
    image_paths = generate_solid_images(args.out_dir, size=args.size)

    # 2. 确定要测试的模型
    model_keys = ['0.5b', '7b'] if args.model == 'both' else [args.model]

    # 3. 逐模型测试
    all_results = []
    for i, model_key in enumerate(model_keys):
        print(f'\n[2/3] 测试模型 ({i + 1}/{len(model_keys)}): {MODEL_KEYS[model_key]}')
        model = load_model(model_key)
        results = run_test(model, image_paths, model_key)
        all_results.extend(results)

        # 释放显存
        import torch
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f'模型 {model_key} 测试完成，已释放显存。')

    # 4. 保存结果
    print('\n[3/3] 保存结果...')
    json_path, md_path = save_results(all_results, args.out_dir)

    print('\n' + '=' * 60)
    print('测试完成！')
    print(f'  JSON: {json_path}')
    print(f'  报告: {md_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
