"""
测试 LLM API Key 是否有效。
用法: python test_api.py
"""
import json
import os


def test_kimi(api_key, base_url=None, model=None):
    try:
        import anthropic
        client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url or "https://api.kimi.com/coding/",
        )
        response = client.messages.create(
            model=model or "kimi-latest",
            max_tokens=50,
            messages=[{"role": "user", "content": "你好"}],
        )
        print("[OK] KIMI API Key 有效")
        print(f"     模型返回: {response.content[0].text}")
        return True
    except Exception as e:
        print(f"[FAIL] KIMI API Key 无效: {e}")
        return False


def test_anthropic(api_key, base_url=None, model=None):
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        response = client.messages.create(
            model=model or "claude-sonnet-4-6",
            max_tokens=50,
            messages=[{"role": "user", "content": "Hello"}],
        )
        print("[OK] Anthropic API Key 有效")
        return True
    except Exception as e:
        print(f"[FAIL] Anthropic API Key 无效: {e}")
        return False


def main():
    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"未找到 {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    provider = config.get("type") or config.get("llm_provider", "unknown")
    api_key = config.get("api_key", "")
    base_url = config.get("base_url")
    model = config.get("model")

    print(f"配置文件: {config_path}")
    print(f"Provider: {provider}")
    if base_url:
        print(f"Base URL: {base_url}")
    if model:
        print(f"Model: {model}")
    print(f"API Key:  {api_key[:12]}...{api_key[-4:] if len(api_key) > 16 else ''} (长度: {len(api_key)})")
    print()

    if not api_key:
        print("[FAIL] api_key 为空，请在 config.json 中填写")
        return

    if provider == "kimi":
        test_kimi(api_key, base_url, model)
    elif provider == "anthropic":
        test_anthropic(api_key, base_url, model)
    else:
        print(f"不支持测试 provider: {provider}")


if __name__ == "__main__":
    main()
