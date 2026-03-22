"""
DeepSeek API 连通性测试脚本
用法：
  1. 在项目根目录的 .env 文件中配置：
     OPENAI_API_KEY=sk-你的deepseek的key
     OPENAI_API_BASE=https://api.deepseek.com/v1/chat/completions
     LOCAL_LLM=deepseek-chat
  2. 运行本脚本：
     python scripts/test_deepseek_api.py
"""
import os
import os.path as osp
import sys
import json
import requests

# Add vlmeval to path
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))

def load_env_file():
    """从项目根目录的 .env 文件加载环境变量"""
    project_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    env_path = osp.join(project_root, '.env')
    if osp.exists(env_path):
        print(f"📄 正在从 {env_path} 加载配置...")
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value
        print("✅ .env 文件加载完成")
    else:
        print(f"⚠️  未找到 .env 文件: {env_path}，将使用已有的环境变量。")

def test_raw_request():
    """直接用 requests 测试 DeepSeek API"""
    # 先加载 .env 文件
    load_env_file()

    api_key = os.environ.get('OPENAI_API_KEY', '')
    api_base = os.environ.get('OPENAI_API_BASE', '')
    model = os.environ.get('LOCAL_LLM', 'deepseek-chat')

    print("=" * 60)
    print("DeepSeek API 连通性测试")
    print("=" * 60)
    print(f"  OPENAI_API_KEY:  {api_key[:10]}...{api_key[-4:]}" if len(api_key) > 14 else f"  OPENAI_API_KEY:  {api_key}")
    print(f"  OPENAI_API_BASE: {api_base}")
    print(f"  LOCAL_LLM:       {model}")
    print("=" * 60)

    if not api_key:
        print("❌ 错误: OPENAI_API_KEY 环境变量未设置！")
        return False
    if not api_base:
        print("❌ 错误: OPENAI_API_BASE 环境变量未设置！")
        return False

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': 'Hello, please reply with just "OK".'}],
        'max_tokens': 16,
        'temperature': 0
    }

    print(f"\n>>> 正在向 {api_base} 发送测试请求...")
    try:
        resp = requests.post(api_base, headers=headers, data=json.dumps(payload), timeout=30)
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False

    print(f"<<< HTTP 状态码: {resp.status_code}")

    if 200 <= resp.status_code < 300:
        try:
            data = resp.json()
            answer = data['choices'][0]['message']['content'].strip()
            print(f"<<< 模型回复: {answer}")
            print("\n✅ DeepSeek API 连接成功！可以正常用于 VLMEvalKit 评测。")
            return True
        except Exception as e:
            print(f"❌ 解析响应失败: {e}")
            print(f"<<< 原始响应: {resp.text[:500]}")
            return False
    else:
        print(f"❌ 请求返回错误码 {resp.status_code}")
        print(f"<<< 原始响应: {resp.text[:500]}")
        if resp.status_code == 404:
            print("\n💡 提示: 404 错误通常表示 OPENAI_API_BASE 地址不正确。")
            print("   VLMEvalKit 需要完整的 endpoint 地址，请确认设置为:")
            print("   export OPENAI_API_BASE=\"https://api.deepseek.com/v1/chat/completions\"")
        elif resp.status_code == 401:
            print("\n💡 提示: 401 错误表示 API Key 无效，请检查 OPENAI_API_KEY 是否正确。")
        return False

if __name__ == '__main__':
    success = test_raw_request()
    sys.exit(0 if success else 1)
