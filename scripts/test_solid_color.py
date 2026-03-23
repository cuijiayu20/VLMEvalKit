"""
纯色图测试脚本：对比 LLaVA-OneVision 0.5B vs 7B 模型在纯色图/色块图输入下的表现。
测试模型对颜色的识别能力，要求输出色名与 HEX/RGB 值。

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
import itertools
import random
from datetime import datetime
from PIL import Image, ImageDraw

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ===== 纯色图配置 =====
SOLID_COLORS = {
    'red':     (255, 0, 0),
    'green':   (0, 255, 0),
    'blue':    (0, 0, 255),
    'white':   (255, 255, 255),
    'black':   (0, 0, 0),
    'yellow':  (255, 255, 0),
    'cyan':    (0, 255, 255),
    'magenta': (255, 0, 255),
    'gray':    (128, 128, 128),
    'orange':  (255, 165, 0),
    'purple':  (128, 0, 128),
    'pink':    (255, 192, 203),
}

def rgb_to_hex(rgb):
    """RGB 转 HEX"""
    return '#{:02X}{:02X}{:02X}'.format(*rgb)


# ===== 测试问题 =====
# 针对纯色图的问题
QUESTIONS_SOLID = [
    "What is the exact color of this image? Please provide the color name, HEX code, and RGB values.",
    "This image is a solid color. Identify the color and output its name, HEX code (e.g., #FF0000), and RGB values (e.g., RGB(255,0,0)).",
    "Please tell me the color shown in this image. Answer in the format: Color name: xxx, HEX: #xxxxxx, RGB: (r, g, b).",
]

# 针对 2x2 色块图的问题
QUESTIONS_GRID = [
    "This image contains a 2x2 grid of colored blocks. Identify each block's color name, HEX code, and RGB values. Describe them as top-left, top-right, bottom-left, bottom-right.",
    "There are 4 color blocks arranged in a 2x2 grid. For each block (top-left, top-right, bottom-left, bottom-right), provide its color name and HEX code.",
]


# ===== 模型名称映射 =====
MODEL_KEYS = {
    '0.5b': 'llava-onevision-qwen2-0.5b-ov-hf',
    '7b':   'llava-onevision-qwen2-7b-ov-hf',
}


def generate_solid_images(out_dir, size=384):
    """生成纯色图，返回 [(图片名, 图片路径, ground_truth_dict)] 列表"""
    img_dir = os.path.join(out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    images = []
    for name, rgb in SOLID_COLORS.items():
        path = os.path.join(img_dir, f'solid_{name}.png')
        img = Image.new('RGB', (size, size), color=rgb)
        img.save(path)

        gt = {
            'type': 'solid',
            'color_name': name,
            'hex': rgb_to_hex(rgb),
            'rgb': rgb,
        }
        images.append((f'solid_{name}', path, gt))
        print(f'  生成纯色图: {name} {rgb} {rgb_to_hex(rgb)} -> {path}')

    return images


def generate_grid_images(out_dir, size=384, num_grids=5):
    """生成 2x2 色块图，随机从颜色集中选4种颜色，返回列表"""
    img_dir = os.path.join(out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    color_names = list(SOLID_COLORS.keys())
    images = []

    # 生成若干组2x2色块图
    random.seed(42)
    for i in range(num_grids):
        chosen = random.sample(color_names, 4)
        half = size // 2

        img = Image.new('RGB', (size, size))
        draw = ImageDraw.Draw(img)

        positions = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
        coords = [(0, 0, half, half), (half, 0, size, half),
                  (0, half, half, size), (half, half, size, size)]

        block_info = {}
        for pos, coord, cname in zip(positions, coords, chosen):
            rgb = SOLID_COLORS[cname]
            draw.rectangle(coord, fill=rgb)
            block_info[pos] = {
                'color_name': cname,
                'hex': rgb_to_hex(rgb),
                'rgb': rgb,
            }

        fname = f'grid_{i}_{"_".join(chosen)}'
        path = os.path.join(img_dir, f'{fname}.png')
        img.save(path)

        gt = {
            'type': 'grid_2x2',
            'blocks': block_info,
        }
        images.append((fname, path, gt))
        block_desc = ', '.join([f'{p}: {block_info[p]["color_name"]}' for p in positions])
        print(f'  生成色块图: {fname} [{block_desc}]')

    return images


def load_model(model_key):
    """加载模型，返回模型实例"""
    from vlmeval.config import supported_VLM

    model_name = MODEL_KEYS[model_key]
    print(f'\n正在加载模型: {model_name} ...')
    model = supported_VLM[model_name]()
    print(f'模型 {model_name} 加载完成。')
    return model


def run_test(model, all_images, model_key):
    """对所有图片 x 对应问题运行推理，返回结果列表"""
    results = []

    # 按类型分配问题
    solid_imgs = [(n, p, gt) for n, p, gt in all_images if gt['type'] == 'solid']
    grid_imgs = [(n, p, gt) for n, p, gt in all_images if gt['type'] == 'grid_2x2']

    tasks = []
    for name, path, gt in solid_imgs:
        for q in QUESTIONS_SOLID:
            tasks.append((name, path, gt, q))
    for name, path, gt in grid_imgs:
        for q in QUESTIONS_GRID:
            tasks.append((name, path, gt, q))

    total = len(tasks)
    for count, (name, img_path, gt, question) in enumerate(tasks, 1):
        print(f'  [{count}/{total}] {model_key} | {name} | {question[:50]}...')

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
            'image_name': name,
            'image_type': gt['type'],
            'ground_truth': gt,
            'question': question,
            'response': response,
        })
        print(f'    -> {response[:120]}')

    return results


def save_results(results, out_dir):
    """保存结果为 JSON 和 Markdown 报告"""
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存 JSON
    json_path = os.path.join(out_dir, f'results_{timestamp}.json')

    # ground_truth 中的 rgb 是 tuple，需要转为 list 方便序列化
    def convert_gt(obj):
        if isinstance(obj, tuple):
            return list(obj)
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=convert_gt)
    print(f'\nJSON 结果已保存: {json_path}')

    # 生成 Markdown 报告
    md_path = os.path.join(out_dir, f'report_{timestamp}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# 纯色图 & 色块图 颜色识别测试报告\n\n')
        f.write(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')

        # === 纯色图结果 ===
        f.write('## 一、纯色图测试\n\n')
        solid_results = [r for r in results if r['image_type'] == 'solid']

        for question in QUESTIONS_SOLID:
            f.write(f'### 问题: {question}\n\n')
            f.write('| 颜色 | GT HEX | GT RGB | 0.5B 回答 | 7B 回答 |\n')
            f.write('|------|--------|--------|-----------|--------|\n')

            for color_name, rgb in SOLID_COLORS.items():
                row_05b = ''
                row_7b = ''
                for r in solid_results:
                    gt = r['ground_truth']
                    if gt.get('color_name') == color_name and r['question'] == question:
                        resp = r['response'].replace('\n', ' ').replace('|', '\\|')[:150]
                        if r['model_key'] == '0.5b':
                            row_05b = resp
                        elif r['model_key'] == '7b':
                            row_7b = resp

                hex_val = rgb_to_hex(rgb)
                f.write(f'| {color_name} | {hex_val} | {rgb} | {row_05b} | {row_7b} |\n')

            f.write('\n')

        # === 色块图结果 ===
        f.write('## 二、2×2 色块图测试\n\n')
        grid_results = [r for r in results if r['image_type'] == 'grid_2x2']

        for r in grid_results:
            gt = r['ground_truth']
            blocks = gt['blocks']
            f.write(f'### 图片: {r["image_name"]} | 模型: {r["model_key"]}\n\n')
            f.write(f'**问题:** {r["question"]}\n\n')
            f.write('**Ground Truth:**\n\n')
            f.write('| 位置 | 颜色 | HEX | RGB |\n')
            f.write('|------|------|-----|-----|\n')
            for pos in ['top-left', 'top-right', 'bottom-left', 'bottom-right']:
                b = blocks[pos]
                f.write(f'| {pos} | {b["color_name"]} | {b["hex"]} | {b["rgb"]} |\n')
            f.write(f'\n**模型回答:**\n\n{r["response"]}\n\n---\n\n')

    print(f'Markdown 报告已保存: {md_path}')
    return json_path, md_path


def parse_args():
    parser = argparse.ArgumentParser(description='纯色图/色块图颜色识别测试: LLaVA-OneVision 0.5B vs 7B')
    parser.add_argument('--model', type=str, default='both', choices=['0.5b', '7b', 'both'], help='要测试的模型 (默认: both)')
    parser.add_argument('--size', type=int, default=384, help='图像尺寸 (默认: 384)')
    parser.add_argument('--num-grids', type=int, default=5, help='生成 2x2 色块图的数量 (默认: 5)')
    parser.add_argument('--out-dir', type=str, default=None, help='输出目录 (默认: outputs/solid_color_test)')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.out_dir is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        args.out_dir = os.path.join(project_root, 'outputs', 'solid_color_test')

    print('=' * 60)
    print('纯色图 & 色块图 颜色识别测试: LLaVA-OneVision 0.5B vs 7B')
    print('=' * 60)

    # 1. 生成测试图
    print('\n[1/3] 生成测试图...')
    solid_images = generate_solid_images(args.out_dir, size=args.size)
    grid_images = generate_grid_images(args.out_dir, size=args.size, num_grids=args.num_grids)
    all_images = solid_images + grid_images
    print(f'  共生成 {len(solid_images)} 张纯色图 + {len(grid_images)} 张色块图')

    # 2. 确定要测试的模型
    model_keys = ['0.5b', '7b'] if args.model == 'both' else [args.model]

    # 3. 逐模型测试
    all_results = []
    for i, model_key in enumerate(model_keys):
        print(f'\n[2/3] 测试模型 ({i + 1}/{len(model_keys)}): {MODEL_KEYS[model_key]}')
        model = load_model(model_key)
        results = run_test(model, all_images, model_key)
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
