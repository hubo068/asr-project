import whisper
import argparse
import os
import re
import json
import shutil


def format_time(seconds):
    """将秒数转换为 HH:MM:SS 格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_config(config_path="config.json"):
    """读取本地配置文件。"""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def normalize_whitespace(text):
    """将多个连续空格压缩为单个空格，并清理首尾空白。"""
    return re.sub(r'\s+', ' ', text).strip()


def needs_punctuation(text):
    """检查文本末尾是否缺少句末标点。"""
    if not text:
        return False
    return text[-1] not in '。！？.,;:!?'


def smart_join(prev_text, next_text, gap):
    """
    智能拼接两句转录文本。
    """
    prev_text = prev_text.rstrip()
    next_text = next_text.lstrip()

    if not prev_text:
        return next_text
    if not next_text:
        return prev_text

    if not needs_punctuation(prev_text):
        return prev_text + ' ' + next_text

    next_first_char = next_text[0]

    if gap > 1.5:
        if '一' <= next_first_char <= '鿿':
            return prev_text + '。' + next_text
        else:
            return prev_text + '. ' + next_text

    if '一' <= next_first_char <= '鿿':
        return prev_text + '，' + next_text
    else:
        return prev_text + ', ' + next_text


def smart_paragraphs(segments, pause_threshold=1.5, max_chars=180):
    """
    根据时间停顿和语义将 segments 合并成自然段落，并智能补全标点。
    """
    paragraphs = []
    current_text = ""
    current_start = None
    current_end = None

    for i, seg in enumerate(segments):
        text = normalize_whitespace(seg["text"])
        if not text:
            continue

        if current_start is None:
            current_start = seg["start"]
            current_end = seg["end"]
            current_text = text
            continue

        prev_end = segments[i - 1]["end"]
        gap = seg["start"] - prev_end
        need_new = False

        if gap > pause_threshold:
            need_new = True

        if len(current_text) >= max_chars and not needs_punctuation(current_text):
            need_new = True

        if "\n" in text:
            parts = text.split("\n")
            current_text = smart_join(current_text, parts[0], gap)
            current_end = seg["end"]
            paragraphs.append({
                "start": current_start,
                "end": current_end,
                "text": current_text.strip()
            })
            for part in parts[1:-1]:
                if part.strip():
                    paragraphs.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": part.strip()
                    })
            current_text = parts[-1].strip() if len(parts) > 1 else ""
            current_start = seg["start"]
            current_end = seg["end"]
            continue

        if need_new:
            if needs_punctuation(current_text):
                current_text += '。'
            paragraphs.append({
                "start": current_start,
                "end": current_end,
                "text": current_text.strip()
            })
            current_text = text
            current_start = seg["start"]
        else:
            current_text = smart_join(current_text, text, gap)

        current_end = seg["end"]

    if current_text.strip():
        if needs_punctuation(current_text):
            current_text += '。'
        paragraphs.append({
            "start": current_start,
            "end": current_end,
            "text": current_text.strip()
        })

    return paragraphs


def refine_text_with_llm(md_content, provider=None, api_key=None, base_url=None, model=None):
    """
    使用 LLM 对转录文本进行后处理，修正同音字、人名、专有名词等错误。
    """
    if provider is None:
        raise RuntimeError("未指定 LLM provider，请在 config.json 中设置 type 或 llm_provider。")

    if not api_key:
        raise RuntimeError(f"未找到 API Key。请在 config.json 中设置 api_key。")

    print(f"  LLM Provider: {provider}")
    if base_url:
        print(f"  Base URL: {base_url}")
    if model:
        print(f"  Model: {model}")

    system_prompt = (
        "你是一位专业的文字校对编辑，擅长语音识别文本的纠错。"
        "你的任务是对转录文本进行审校，严格遵循以下规则：\n"
        "1. 修正同音字错误（如：在/再、做/作、的/得/地、他/她/它等）；\n"
        "2. 修正人名、地名、公司名、品牌名、专业术语、技术名词等专有名词；\n"
        "3. 修正明显的语法和标点错误，但不要过度修改；\n"
        "4. 保持原文的口语风格、语气和表达方式，不要润色或重写；\n"
        "5. 绝对不要添加原文中没有的信息；\n"
        "6. 保留所有 Markdown 格式和时间戳（如 `00:01:23`）。"
    )

    user_prompt = (
        "请对以下语音识别转录文本进行校对纠错。"
        "直接返回修正后的完整 Markdown 文本，不要添加任何解释或总结。\n\n"
        f"{md_content}"
    )

    if provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("请先安装 Anthropic SDK: pip install anthropic")

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model or "claude-sonnet-4-6",
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    elif provider == "openai":
        try:
            import openai
        except ImportError:
            raise RuntimeError("请先安装 OpenAI SDK: pip install openai")

        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model or "gpt-4o",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    elif provider == "kimi":
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("KIMI 当前使用 Anthropic SDK 调用，请先安装: pip install anthropic")

        try:
            client = anthropic.Anthropic(
                api_key=api_key,
                base_url=base_url or "https://api.kimi.com/coding/",
            )
            # KIMI 的 Anthropic 兼容接口对 system 参数敏感，合并到 user prompt 中
            full_prompt = (
                "【系统指令】\n" + system_prompt + "\n\n" +
                "【用户请求】\n" + user_prompt
            )
            response = client.messages.create(
                model=model or "kimi-latest",
                max_tokens=8192,
                messages=[{"role": "user", "content": full_prompt}],
            )
            return response.content[0].text
        except anthropic.AuthenticationError as e:
            raise RuntimeError(
                "KIMI API 认证失败 (401)。可能原因:\n"
                "1. API Key 无效或已过期\n"
                "2. Key 复制时混入了空格或换行，请检查 config.json\n"
                "3. base_url 配置错误\n"
                f"原始错误: {e}"
            ) from e
        except anthropic.BadRequestError as e:
            if "high risk" in str(e).lower():
                raise RuntimeError(
                    "KIMI 安全策略拦截 (400 high risk)。可能原因:\n"
                    "1. 转录文本中包含敏感内容\n"
                    "2. 请求被风控系统误判\n"
                    "建议: 检查转录文本内容，或尝试更换模型\n"
                    f"原始错误: {e}"
                ) from e
            raise

    else:
        raise ValueError(f"不支持的 LLM provider: {provider}")


def transcribe_audio(audio_path, model_size="base", output_path=None):
    """
    只做语音识别，生成 Markdown 文稿。
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    if output_path is None:
        base, _ = os.path.splitext(audio_path)
        output_path = base + ".md"

    print(f"[步骤 1/2] 正在加载 Whisper {model_size} 模型...")
    model = whisper.load_model(model_size)

    print(f"[步骤 1/2] 正在转录: {audio_path}")
    result = model.transcribe(
        audio_path,
        verbose=False,
        initial_prompt="请输出带标点符号的完整句子。",
        condition_on_previous_text=True,
    )

    segments = result.get("segments", [])
    detected_language = result.get("language", "unknown")
    total_duration = segments[-1]["end"] if segments else 0

    print(f"检测到的语言: {detected_language}")
    print(f"总时长: {format_time(total_duration)}")

    paragraphs = smart_paragraphs(segments)
    print(f"共生成 {len(paragraphs)} 个段落")

    lines = [
        "# 语音转录文稿",
        "",
        f"- **源文件**: `{os.path.basename(audio_path)}`",
        f"- **识别语言**: {detected_language}",
        f"- **总时长**: {format_time(total_duration)}",
        "",
        "---",
        "",
    ]

    for para in paragraphs:
        time_tag = f"`{format_time(para['start'])}`"
        lines.append(f"{time_tag} {para['text']}")
        lines.append("")

    md_content = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"文稿已保存: {output_path}")
    return output_path


def refine_md(md_path, output_path=None, llm_provider=None, api_key=None, base_url=None, model=None):
    """
    只做 LLM 优化，读取已有的 .md 文件进行校对纠错。
    """
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Markdown 文件不存在: {md_path}")

    if output_path is None:
        base, ext = os.path.splitext(md_path)
        output_path = base + "_refined" + ext

    print(f"[步骤 2/2] 正在读取: {md_path}")
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    print(f"[步骤 2/2] 正在使用 LLM 进行文字校对...")
    refined_content = refine_text_with_llm(
        md_content, provider=llm_provider, api_key=api_key, base_url=base_url, model=model
    )
    print("校对完成")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(refined_content)

    print(f"优化文稿已保存: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="语音转 Markdown 文稿 (支持分步执行: 识别 +> LLM 优化)"
    )
    parser.add_argument("audio", help="音频文件路径 (或 .md 文件路径，配合 --only-refine)")
    parser.add_argument(
        "--model", default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper 模型大小 (默认: base)"
    )
    parser.add_argument(
        "--output", "-o",
        help="输出文件路径 (默认: 与源文件同名 .md)"
    )
    parser.add_argument(
        "--refine", action="store_true",
        help="启用 LLM 后处理。如果 .md 已存在则直接优化，否则先转录再优化"
    )
    parser.add_argument(
        "--only-refine", action="store_true",
        help="只做 LLM 优化，要求对应的 .md 文件已存在"
    )
    parser.add_argument(
        "--config", default="config.json",
        help="配置文件路径 (默认: config.json)"
    )

    parser.add_argument(
        "--llm-provider",
        help="指定使用 config.json 中配置的哪个 provider（默认使用 default 指定的）"
    )

    args = parser.parse_args()

    # 读取配置，支持多 provider 格式和旧格式
    config = load_config(args.config)

    # 多 provider 格式: { "providers": { "kimi": {...}, "openai": {...} }, "default": "kimi" }
    if "providers" in config:
        provider_name = args.llm_provider or config.get("default", "kimi")
        provider_config = config["providers"].get(provider_name)
        if not provider_config:
            available = list(config["providers"].keys())
            raise RuntimeError(
                f"配置文件中未找到 provider '{provider_name}'。"
                f"可用的 provider: {available}"
            )
        llm_provider = provider_config.get("type") or provider_config.get("llm_provider")
        api_key = provider_config.get("api_key")
        base_url = provider_config.get("base_url")
        model = provider_config.get("model")
        print(f"使用配置: {provider_name} ({llm_provider})")
    else:
        # 旧格式兼容: { "type": "kimi", "api_key": "...", ... }
        llm_provider = config.get("type") or config.get("llm_provider")
        api_key = config.get("api_key")
        base_url = config.get("base_url")
        model = config.get("model")

    # 计算默认路径
    audio_base, _ = os.path.splitext(args.audio)
    md_path = args.output if args.output else audio_base + ".md"
    refined_path = audio_base + "_refined.md"

    # 只做 LLM 优化
    if args.only_refine:
        if not os.path.exists(md_path):
            raise FileNotFoundError(
                f"找不到 {md_path}，请先运行转录生成 .md 文件，"
                f"或指定 --output 指向已有的 .md 文件。"
            )
        refine_md(md_path, output_path=refined_path, llm_provider=llm_provider, api_key=api_key, base_url=base_url, model=model)
        return

    # 转录 + 可选优化
    if args.refine:
        if os.path.exists(md_path):
            print(f"检测到已有文稿: {md_path}，跳过语音识别，直接进行 LLM 优化。")
            refine_md(md_path, output_path=refined_path, llm_provider=llm_provider, api_key=api_key, base_url=base_url, model=model)
        else:
            transcribe_audio(args.audio, model_size=args.model, output_path=md_path)
            refine_md(md_path, output_path=refined_path, llm_provider=llm_provider, api_key=api_key, base_url=base_url, model=model)
    else:
        transcribe_audio(args.audio, model_size=args.model, output_path=md_path)


if __name__ == "__main__":
    main()
