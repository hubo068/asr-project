"""
测试 LLM API Key 是否有效。
用法: python test_api.py [--provider 名称]
"""
import json
import os
import argparse


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


def test_openai(api_key, base_url=None, model=None):
    try:
        import openai
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
        )
        print("[OK] OpenAI API Key 有效")
        return True
    except Exception as e:
        print(f"[FAIL] OpenAI API Key 无效: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="测试 LLM API Key")
    parser.add_argument("--provider", help="指定测试 config.json 中的哪个 provider")
    parser.add_argument("--config", default="config.json", help="配置文件路径")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"未找到 {args.config}")
        return

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 多 provider 格式
    if "providers" in config:
        provider_name = args.provider or config.get("default")
        if not provider_name:
            print("未指定 provider，可用: --provider " + " ".join(config["providers"].keys()))
            return
        provider_config = config["providers"].get(provider_name)
        if not provider_config:
            print(f"未找到 provider '{provider_name}'")
            return
        provider = provider_config.get("type") or provider_config.get("llm_provider", "unknown")
        api_key = provider_config.get("api_key", "")
        base_url = provider_config.get("base_url")
        model = provider_config.get("model")
        print(f"Provider: {provider_name} ({provider})")
    else:
        # 旧格式
        provider = config.get("type") or config.get("llm_provider", "unknown")
        api_key = config.get("api_key", "")
        base_url = config.get("base_url")
        model = config.get("model")
        print(f"Provider: {provider}")

    if base_url:
        print(f"Base URL: {base_url}")
    if model:
        print(f"Model: {model}")
    print(f"API Key:  {api_key[:12]}...{api_key[-4:] if len(api_key) > 16 else ''} (长度: {len(api_key)})")
    print()

    if not api_key:
        print("[FAIL] api_key 为空")
        return

    if provider == "kimi":
        test_kimi(api_key, base_url, model)
    elif provider == "anthropic":
        test_anthropic(api_key, base_url, model)
    elif provider == "openai":
        test_openai(api_key, base_url, model)
    else:
        print(f"不支持测试 provider: {provider}")


if __name__ == "__main__":
    main()
