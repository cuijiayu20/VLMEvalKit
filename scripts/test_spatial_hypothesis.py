"""
验证多模态模型空间理解瓶颈的实验脚本。
假设：多模态模型的空间理解瓶颈不在于缺少深度信息，而在于视觉编码器未能提供足够的触发信号来激活 LLM 中已有的空间推理能力。

实验通过 5 个对照条件来解耦视觉编码器和 LLM 的能力。

用法：
    python scripts/test_spatial_hypothesis.py --model 7b
"""

import os
import sys
import argparse
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vlmeval.config import supported_VLM


# MMBench 1004 题原图路径 (在服务器上的位置)
IMAGE_PATH = '/root/LMUData/images/MMBench/3001004.png'

# 基础问题格式 (严格对齐 MMBench Prompt 风格)
BASE_PROMPT = """Question: Where is the sheep?
Options:
A. The sheep is behind the car
B. The sheep is in the front of the car
C. The sheep is on the right of the car
D. The sheep is on the left of the car
Please select the correct answer from the options above."""

# 实验条件定义
CONDITIONS = [
    {
        'id': 'C1',
        'name': '纯图像 (Baseline)',
        'type': 'image_text',
        'use_image': True,
        'image_modifier': None,
        'prompt': BASE_PROMPT,
    },
    
    # --- 条件2：纯文本（测试 LLM 空间推理）---
    {
        'id': 'C2_L1',
        'name': '无图纯文本_粗略描述',
        'type': 'text_only',
        'use_image': False, # 现在用 dummy_image 替代
        'prompt': "Hint: A sheep is standing on a road. A car is behind the sheep.\n" + BASE_PROMPT,
    },
    {
        'id': 'C2_L2',
        'name': '无图纯文本_中等描述',
        'type': 'text_only',
        'use_image': False,
        'prompt': "Hint: A white sheep is standing in the middle of a road, facing the camera. A gray car is behind the sheep, facing the same direction as the camera view.\n" + BASE_PROMPT,
    },
    {
        'id': 'C2_L3',
        'name': '无图纯文本_详细描述',
        'type': 'text_only',
        'use_image': False,
        'prompt': "Hint: In this photo taken from the front, a white sheep stands on the road in the foreground, directly facing the camera. Behind the sheep, further down the road, there is a gray car. The car's front is visible, indicating it is facing the sheep (and the camera). The sheep is positioned between the camera and the car.\n" + BASE_PROMPT,
    },

    # --- 条件3：图像 + 补充文本（扶视觉编码器一把）---
    {
        'id': 'C3_H1',
        'name': '有图+弱提示',
        'type': 'image_text',
        'use_image': True,
        'prompt': "Hint: The sheep is closer to the camera than the car.\n" + BASE_PROMPT,
    },
    {
        'id': 'C3_H2',
        'name': '有图+中提示',
        'type': 'image_text',
        'use_image': True,
        'prompt': "Hint: In this image, the sheep is in the foreground and the car is in the background.\n" + BASE_PROMPT,
    },
    {
        'id': 'C3_H3',
        'name': '有图+强提示',
        'type': 'image_text',
        'use_image': True,
        'prompt': "Hint: Spatial context: The sheep is standing in front of the car. The sheep is closer to the viewer, and the car is further away behind the sheep.\n" + BASE_PROMPT,
    },

    # --- 条件4：图像 + 深度信息（验证深度假说）---
    {
        'id': 'C4_D1',
        'name': '有图+绝对距离描述',
        'type': 'image_text',
        'use_image': True,
        'prompt': "Hint: Depth information - The sheep is approximately 5 meters from the camera. The car is approximately 15 meters from the camera.\n" + BASE_PROMPT,
    },
    {
        'id': 'C4_D2',
        'name': '有图+相对比例描述',
        'type': 'image_text',
        'use_image': True,
        'prompt': "Hint: Depth hint - The sheep occupies about 30% of the image height, while the car occupies about 20% of the image height despite being a larger object in reality.\n" + BASE_PROMPT,
    },

    # --- 条件5：增强视觉线索（修改图片）---
    {
        'id': 'C5_V1',
        'name': '图上叠加文字和框',
        'type': 'image_text',
        'use_image': True,
        'image_modifier': 'draw_boxes_and_text',
        'prompt': BASE_PROMPT,
    },
]

def get_dummy_image(out_dir):
    """生成一张纯黑的 10x10 图片作为无图条件的输入，避免 pipeline 崩溃"""
    path = os.path.join(out_dir, "dummy_black.jpg")
    if not os.path.exists(path):
        os.makedirs(out_dir, exist_ok=True)
        img = Image.new('RGB', (10, 10), color=(0, 0, 0))
        img.save(path)
    return path

def modify_image_with_annotations(img_path, out_dir):
    """在图上画框和文字标注（条件 5）"""
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"无法打开原图 {img_path}: {e}")
        return None

    draw = ImageDraw.Draw(img)
    
    width, height = img.size
    
    # 大致的坐标 (针对 3001004.png 羊在中间前方，车在后面偏左)
    # 这些比例可能是粗略估计的
    sheep_box = [width*0.35, height*0.5, width*0.55, height*0.85]
    car_box = [width*0.35, height*0.35, width*0.8, height*0.55]

    # 画框
    draw.rectangle(sheep_box, outline="red", width=3)
    draw.rectangle(car_box, outline="blue", width=3)

    # 画文字
    try:
        # 尝试加载较大字体，如果没有则用默认（默认字体较小）
        font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        font = ImageFont.load_default()

    draw.text((sheep_box[0], sheep_box[1] - 40), "FRONT (Sheep)", fill="red", font=font)
    draw.text((car_box[0], car_box[1] - 40), "BACKGROUND (Car)", fill="blue", font=font)

    # 画引导线
    draw.line([width*0.45, height*0.85, width*0.45, height*0.95], fill="red", width=5)
    draw.text((width*0.45 + 10, height*0.9), "Close to camera", fill="red", font=font)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "annotated_1004.jpg")
    img.save(out_path)
    return out_path


def build_message(condition, img_path, annotated_img_path, dummy_img_path):
    """构建输入给模型的 message"""
    msg = []
    
    if condition['use_image']:
        if condition.get('image_modifier') == 'draw_boxes_and_text' and annotated_img_path:
            msg.append({'type': 'image', 'value': annotated_img_path})
        elif img_path:
            msg.append({'type': 'image', 'value': img_path})
        else:
            print(f"警告：找不到图片 {img_path}，跳过此条件")
            return None
    else:
        # 必须提供 dummy 图片以避免报错
        msg.append({'type': 'image', 'value': dummy_img_path})
            
    msg.append({'type': 'text', 'value': condition['prompt']})
    return msg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='7b', choices=['0.5b', '7b'])
    parser.add_argument('--out-dir', type=str, default='outputs/spatial_hypothesis')
    args = parser.parse_args()

    model_key = f'llava-onevision-qwen2-{args.model}-ov-hf'
    print(f"加载模型: {model_key}")
    try:
        model = supported_VLM[model_key]()
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    
    # 检查原图是否存在
    if not os.path.exists(IMAGE_PATH):
        print(f"【错误】找不到 MMBench 原图: {IMAGE_PATH}")
        print("请确保脚本在能够访问 /root/LMUData 目录的服务器上运行。")
        sys.exit(1)

    # 准备修改后的图片 (用于条件 5)
    annotated_img_path = modify_image_with_annotations(IMAGE_PATH, args.out_dir)
    # 准备 dummy 图片 (用于条件 2)
    dummy_img_path = get_dummy_image(args.out_dir)

    results = []

    print("\n========== 开始测试 ==========\n")
    for cond in CONDITIONS:
        print(f"运行条件: [{cond['id']}] {cond['name']}")
        
        msg = build_message(cond, IMAGE_PATH, annotated_img_path, dummy_img_path)
        if msg is None:
            continue
            
        try:
            response = model.generate(msg)
        except Exception as e:
            response = f"[Error] 推理失败: {e}"
            
        print(f"  模型回答: {response.strip()}")
        
        # 判断正误 (正确答案是 B)
        is_correct = False
        if isinstance(response, str):
            # 简单的选择题判对逻辑
            pred_letter = response.strip()[0].upper()
            if pred_letter == 'B' or 'in front of' in response.lower():
                is_correct = True
        
        print(f"  判对: {'✅' if is_correct else '❌'}\n")

        results.append({
            'id': cond['id'],
            'name': cond['name'],
            'response': response,
            'is_correct': is_correct
        })

    # ========== 生成报告 ==========
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(args.out_dir, f'report_{args.model}_{timestamp}.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f'# 空间理解瓶颈实验报告 (Model: {args.model})\n\n')
        f.write(f'测试时间: {timestamp}\n\n')
        
        f.write('## 实验结果汇总\n\n')
        f.write('| 条件 ID | 条件说明 | 模型预测 | 是否正确 (GT=B) |\n')
        f.write('|---------|----------|----------|----------------|\n')
        
        for r in results:
            short_resp = r['response'].replace('\n', ' ')[:40] + ('...' if len(r['response']) > 40 else '')
            f.write(f"| {r['id']} | {r['name']} | {short_resp} | {'✅' if r['is_correct'] else '❌'} |\n")
            
        f.write('\n## 详细问答\n\n')
        for r, cond in zip(results, CONDITIONS):
            f.write(f"### [{r['id']}] {r['name']}\n\n")
            f.write(f"**图片**：{'原图' if cond['use_image'] and not cond.get('image_modifier') else ('标注图' if cond.get('image_modifier') else '无')}\n\n")
            f.write(f"**文本输入**：\n```text\n{cond['prompt']}\n```\n\n")
            f.write(f"**模型回答**：\n```text\n{r['response']}\n```\n\n")
            f.write(f"**判断**：{'✅' if r['is_correct'] else '❌'}\n\n")
            f.write("---\n\n")

    print(f"测试完成，报告已生成: {report_path}")

if __name__ == '__main__':
    main()
