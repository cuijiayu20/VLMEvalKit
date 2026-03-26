import os
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# 输出目录改为 Linux 服务器路径
OUTPUT_DIR = r"/root/VLMEvalKit/outputs/probing_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_synthetic_batch(batch_size=3):
    """
    构造一批明确指向特定空间区域的测试样本 (Batch)
    这里生成背景白色的图片，在左上角放置明显的红色方块，并构造 Prompt
    """
    dataset = []
    for i in range(batch_size):
        # 强制将原图生成为 384x384 (SigLip 的基础感知尺寸)
        # 防止因哪怕大一点点 (400x400) 就触发 LLaVA-OV AnyRes 的过度切图导致爆显存
        img = Image.new('RGB', (384, 384), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        # 在左上角添加一个显眼的红色特征区域
        d.rectangle([10, 10, 120, 120], fill=(255, 0, 0))
        path = os.path.join(OUTPUT_DIR, f"synthetic_sample_{i}.jpg")
        img.save(path)
        
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "请描述图片左上角的物体"}]}]
        dataset.append({
            "image_path": path, 
            "messages": messages, 
            "target_word": "左"
        })
    return dataset

def extract_full_attention(model, processor, image, messages, target_word):
    """
    核心实验代码：运行 Forward Pass，返回【每一层】、【每个Head】的 Attention 权重
    """
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device, torch.float16)
    input_ids = inputs["input_ids"][0]
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
    attentions = outputs.attentions 
    seq_len = attentions[0].shape[-1]
    input_len = input_ids.shape[0]
    
    tokens_str = [processor.tokenizer.decode([t]) for t in input_ids]
    target_idx_in_input = next((i for i, t in enumerate(tokens_str) if target_word in t), -1)
    if target_idx_in_input == -1: 
        raise ValueError(f"Target word '{target_word}' not found in tokens.")
        
    image_token_index = model.config.image_token_index
    visual_indices = torch.where(input_ids == image_token_index)[0].cpu().tolist()
    target_idx_in_attn = target_idx_in_input
    
    # Visual Tokens 在输入序列中的具体对齐偏移量计算
    if len(visual_indices) > 1:
        blocks = []
        curr = []
        for idx in visual_indices:
            if not curr or idx == curr[-1] + 1: curr.append(idx)
            else:
                blocks.append(curr)
                curr = [idx]
        if curr: blocks.append(curr)
        vis_block = blocks[0]
        vis_start, vis_end = vis_block[0], vis_block[-1] + 1
    else:
        original_image_idx = visual_indices[0] if len(visual_indices) == 1 else input_ids.tolist().index(image_token_index)
        num_visual_tokens = seq_len - input_len + 1
        vis_start = original_image_idx
        vis_end = original_image_idx + num_visual_tokens
        if target_idx_in_input > original_image_idx:
            target_idx_in_attn = target_idx_in_input - 1 + num_visual_tokens
            
    # 为了防止多层巨大的 Attention 矩阵在单张显卡上瞬间堆积导致 24GB 显存 OOM
    # 先将每一层的 Attention 卸载到 CPU 内存上，并且为了绘图保险转换为 float32，然后再 Stack
    cpu_attentions = [layer_attn.cpu().to(torch.float32) for layer_attn in attentions]
    all_layers_attn = torch.stack(cpu_attentions) 
    
    # 切片提取指定的文本 token 到所有视觉 token 的 attention 
    # 结果形状应为: [层数, 独立Head数, 视觉Tokens数量]
    target_attn_tensor = all_layers_attn[:, 0, :, target_idx_in_attn, vis_start:vis_end].numpy()
    
    return target_attn_tensor

def main():
    # HuggingFace 报错 Repo ID 格式错误，其根本原因是因为它在本地硬盘上“找不到这个目录”，所以误以为你输入的是线上 Repo 名字。
    # 结合你之前的描述，真正的绝对路径应该是 /root/models/llava-ov-7b
    model_id = "/root/models/llava-ov-7b"
    
    import os
    if not os.path.exists(model_id):
        print(f"【路径错误】在容器中找不到文件夹: {model_id}")
        print("因为 HuggingFace 找不到本地文件夹，所以报错了 Repo id 格式错误。")
        print("请在终端运行命令 `ls -l /root/models/llava-ov-7b` 确认文件夹存在！")
        return
        
    print(f"Loading Model from local path: {model_id} ...")
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        # 采用多 GPU 均衡分配策略 (balanced)：给主卡 (0卡) 限制一个安全的加载显存，
        # 强制将大量静态模型权重分配到副卡 (1卡)，以保证主卡有足够的显存用来存放 Forward Pass 的激活动态张量
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, 
            device_map="balanced", 
            max_memory={0: "14GiB", 1: "23GiB"},
            torch_dtype=torch.float16,
            attn_implementation="eager"
        )
    except Exception as e:
        print(f"无法加载模型，请检查环境: {e}")
        return
        
    # 1. 构造一批样本 (Batch Data)
    dataset = generate_synthetic_batch(batch_size=2)
    print(f"已构造测试样本批次，总共 {len(dataset)} 个。")
    
    for idx, sample in enumerate(dataset):
        print(f"\n正在处理样本 {idx}...")
        
        # ==========================================================
        # ★★★ 彻底解决 OOM 的核心防护：规避 AnyRes 爆炸机制 ★★★
        # 读取并在传入 Processor 之前强行将图像降阶至 384x384。
        # 只有这样，才能强制模型只提取 1 个 Global Patch（合 729 个 visual tokens）。
        # 此前 400x400 的图片因为长宽超过阈值，被它强行切分成了一个 2x2 网格，
        # 视觉 token 总数生生翻了 5 倍涨到 3645！在 Eager 模式下强行还原并抛出这重达 20GB+ 
        # 的超巨型注意力张量给加速层 (Accelerator) 分发时，你的 3090 显卡就瞬间炸了。
        # ==========================================================
        image = Image.open(sample["image_path"]).convert("RGB").resize((384, 384))
        
        # 2. 提取【每一层】【每个头】的 Attention Weight
        attn_tensor = extract_full_attention(model, processor, image, sample["messages"], sample["target_word"])
        
        num_layers, num_heads, num_vis_tokens = attn_tensor.shape
        
        # 3. Reshape 回 2D 网格 (24x24 或 27x27)
        if num_vis_tokens > 729:
            attn_tensor = attn_tensor[:, :, :729] # LLaVA-OV AnyRes 保底取 Global Patch
            num_vis_tokens = 729
        
        grid_size = int(np.sqrt(num_vis_tokens))
        attn_2d = attn_tensor.reshape((num_layers, num_heads, grid_size, grid_size))
        
        # --- 核心可视化：绘制囊括“所有层和所有Head”的宏观全景热力图 ---
        print(f"正在绘制 {num_layers} x {num_heads} 全景网格对齐热力图...")
        fig, axes = plt.subplots(num_layers, num_heads, figsize=(num_heads * 1.5, num_layers * 1.5))
        
        for l in range(num_layers):
            for h in range(num_heads):
                head_heatmap = attn_2d[l, h]
                # 最大最小值归一化，以便观察该 Head 的焦点是否聚焦
                v_min, v_max = head_heatmap.min(), head_heatmap.max()
                if v_max > v_min:
                    head_heatmap = (head_heatmap - v_min) / (v_max - v_min)
                    
                axes[l, h].imshow(head_heatmap, cmap='jet')
                axes[l, h].axis('off')
                
                if l == 0:
                    axes[l, h].set_title(f"H{h}", fontsize=10)
            axes[l, 0].text(-0.2, 0.5, f"L{l}", va='center', ha='right', transform=axes[l, 0].transAxes, fontsize=12)

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        save_path = os.path.join(OUTPUT_DIR, f"layer_head_grid_sample_{idx}.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"全景图已保存: {save_path}")

        # 4. 把原始的 Numpy Tensor 保存下来，方便之后提取 2D 数组进行定量分析（比如计算落入左上角的比例）
        tensor_path = os.path.join(OUTPUT_DIR, f"attn_tensor_sample_{idx}.npy")
        np.save(tensor_path, attn_2d)
        print(f"原始特征矩阵(Shape: {num_layers}x{num_heads}x{grid_size}x{grid_size})已保存至: {tensor_path}")
        
        # 为了稳定运行多张图片，每跑完一张立即释放 GPU 显存
        import gc
        del attn_tensor
        del attn_2d
        gc.collect()
        torch.cuda.empty_cache()
        
if __name__ == "__main__":
    main()
