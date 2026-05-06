"""
声音克隆与语音合成模块
基于 CosyVoice 实现中文声音克隆

使用流程：
1. 准备 1-3 段参考音频（建议 3-10 秒，清晰的人声）
2. 注册声音: python voice_clone.py --register my_voice --audio sample.wav
3. 合成语音: python voice_clone.py --text "任意中文文字" --voice my_voice -o output.wav
"""

import os
import argparse
import sys
import json
import re
from datetime import datetime

# 将 CosyVoice 子模块 Matcha-TTS 加入 Python 路径
_MATCHA_TTS_PATH = os.path.join(os.path.dirname(__file__), "CosyVoice", "third_party", "Matcha-TTS")
if os.path.exists(_MATCHA_TTS_PATH) and _MATCHA_TTS_PATH not in sys.path:
    sys.path.insert(0, _MATCHA_TTS_PATH)


def check_cosyvoice():
    """检查 CosyVoice 是否已安装"""
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice
        return True
    except ImportError:
        return False


def check_model(model_path):
    """检查预训练模型是否存在"""
    return os.path.exists(model_path)


VOICES_FILE = "voices.json"


def split_text_into_chunks(text, max_chars=300):
    """
    将长文本按语义切分成多个 chunk，每个不超过 max_chars。
    优先按段落 -> 句子 -> 强制切分的层次处理。
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        # 按句子切分（保留标点）
        sentences = re.findall(r'[^。！？.!?]+[。！？.!?]?', para)
        if not sentences:
            sentences = [para]

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # 超长句子：强制按字数切分
            if len(sent) > max_chars:
                if current.strip():
                    chunks.append(current.strip())
                    current = ""
                for i in range(0, len(sent), max_chars):
                    piece = sent[i:i + max_chars].strip()
                    if len(piece) > 1:
                        chunks.append(piece)
                continue

            # 正常句子
            if current and len(current) + len(sent) > max_chars:
                chunks.append(current.strip())
                current = sent
            else:
                current += sent

        # 段落结束，保存当前 chunk
        if current.strip():
            chunks.append(current.strip())
            current = ""

    if current.strip():
        chunks.append(current.strip())

    # 清理首尾标点和空白
    final = []
    for c in chunks:
        c = re.sub(r'^[。！？.!?；;，,\s]+', '', c)
        c = re.sub(r'[。！？.!?；;，,\s]+$', '', c)
        if len(c) > 1:
            final.append(c)

    return final


def load_text_file(file_path):
    """
    读取文本文件内容，并过滤掉时间轴信息。
    支持 .md、.txt 等纯文本格式。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文本文件不存在: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 过滤 Markdown 元信息行（如 - **源文件**: xxx）
    lines = content.splitlines()
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        # 跳过空行和 Markdown 元信息
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("-") and "**" in stripped:
            continue
        if stripped.startswith("---"):
            continue
        # 移除反引号包裹的时间戳，如 `00:01:23`
        line = re.sub(r"`\d{2}:\d{2}:\d{2}`", "", line)
        # 移除行首的纯时间戳（不带反引号），如 00:01:23
        line = re.sub(r"^\d{2}:\d{2}:\d{2}\s*", "", line)
        filtered_lines.append(line.strip())

    text = "\n".join(filtered_lines)
    # 将多个连续空行压缩为一个
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


class VoiceCloner:
    def __init__(self, model_dir="pretrained_models/CosyVoice-300M"):
        """
        初始化声音克隆器

        Args:
            model_dir: CosyVoice 预训练模型目录路径
        """
        if not check_cosyvoice():
            print("[错误] CosyVoice 未安装，请按以下步骤安装:\n")
            print("1. git clone https://github.com/FunAudioLLM/CosyVoice.git")
            print("2. cd CosyVoice")
            print("3. pip install -e .")
            print("4. 下载预训练模型到 pretrained_models/ 目录")
            print("   模型下载: https://www.modelscope.cn/models/iic/CosyVoice-300M")
            sys.exit(1)

        if not check_model(model_dir):
            print(f"[错误] 模型目录不存在: {model_dir}")
            print("请从 https://github.com/FunAudioLLM/CosyVoice 下载预训练模型")
            sys.exit(1)

        from cosyvoice.cli.cosyvoice import CosyVoice
        import torch

        print(f"正在加载模型: {model_dir}")
        self.cosyvoice = CosyVoice(model_dir)
        self.torch = torch
        self.registered_voices = self._load_voices()
        print("模型加载完成")

    def _load_voices(self):
        """从文件加载已注册的声音列表"""
        if os.path.exists(VOICES_FILE):
            try:
                with open(VOICES_FILE, "r", encoding="utf-8") as f:
                    voices = json.load(f)
                # 过滤掉音频文件已经不存在的记录
                valid_voices = {}
                for name, info in voices.items():
                    if os.path.exists(info.get("prompt_audio", "")):
                        valid_voices[name] = info
                    else:
                        print(f"[警告] 声音 '{name}' 的参考音频已不存在，已忽略")
                return valid_voices
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[警告] 读取 {VOICES_FILE} 失败: {e}")
        return {}

    def _save_voices(self):
        """将已注册的声音列表保存到文件"""
        with open(VOICES_FILE, "w", encoding="utf-8") as f:
            json.dump(self.registered_voices, f, ensure_ascii=False, indent=2)

    def register_voice(self, name, audio_paths):
        """
        注册一个声音（提取声纹）

        Args:
            name: 声音名称，如 "my_voice"
            audio_paths: 参考音频文件路径列表，1-3 段
        """
        if not audio_paths:
            raise ValueError("至少需要提供 1 段参考音频")

        # 只取第一段作为 prompt（CosyVoice 的 zero-shot 模式只需要一段）
        prompt_audio = audio_paths[0]
        if not os.path.exists(prompt_audio):
            raise FileNotFoundError(f"音频文件不存在: {prompt_audio}")

        print(f"正在注册声音 '{name}'，使用参考音频: {prompt_audio}")

        self.registered_voices[name] = {
            "prompt_audio": prompt_audio,
            "audio_count": len(audio_paths),
            "registered_at": datetime.now().isoformat()
        }
        self._save_voices()
        print(f"声音 '{name}' 注册成功（使用了 {len(audio_paths)} 段参考音频）")
        print(f"已保存到 {VOICES_FILE}")

    def list_voices(self):
        """列出所有已注册的声音"""
        if not self.registered_voices:
            print("暂无已注册的声音")
            return
        print(f"已注册的声音（共 {len(self.registered_voices)} 个）:")
        for name, info in self.registered_voices.items():
            print(f"  - {name}: {info['prompt_audio']} ({info['audio_count']} 段音频)")

    def synthesize(self, text, voice_name, output_path="output.wav"):
        """
        使用注册的声音合成语音

        Args:
            text: 要合成的中文文字
            voice_name: 已注册的声音名称
            output_path: 输出音频路径（默认 output.wav）
        """
        import torchaudio

        if voice_name not in self.registered_voices:
            print(f"[错误] 声音 '{voice_name}' 未注册")
            if self.registered_voices:
                print(f"已注册的声音: {list(self.registered_voices.keys())}")
            else:
                print("还没有注册过任何声音，请先用 --register 注册")
            sys.exit(1)

        voice_info = self.registered_voices[voice_name]
        prompt_audio = voice_info["prompt_audio"]

        if not os.path.exists(prompt_audio):
            print(f"[错误] 参考音频已不存在: {prompt_audio}")
            print("请重新注册该声音: python voice_clone.py --register <名称> --audio <音频>")
            sys.exit(1)

        voice_info = self.registered_voices[voice_name]
        prompt_audio = voice_info["prompt_audio"]

        print(f"正在合成语音...")
        print(f"  文字: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"  声音: {voice_name}")
        print(f"  参考音频: {prompt_audio}")

        # 使用 CosyVoice 的 zero-shot（跨语言复刻）模式
        results = list(self.cosyvoice.inference_zero_shot(
            tts_text=text,
            prompt_text="",  # 空字符串表示自动提取参考音频的文本内容
            prompt_wav=prompt_audio
        ))

        if not results:
            raise RuntimeError("合成失败，没有返回结果")

        # 保存最后一段结果
        final_result = results[-1]
        torchaudio.save(output_path, final_result['tts_speech'], 22050)

        print(f"合成完成: {output_path}")
        return output_path

    def synthesize_long_text(self, text, voice_name, output_path="output.wav", max_chars=300):
        """
        长文本分段合成，使用注册的声音。

        Args:
            text: 要合成的长文本
            voice_name: 已注册的声音名称
            output_path: 输出音频路径
            max_chars: 每段最大字符数
        """
        import torchaudio
        import torch

        if voice_name not in self.registered_voices:
            print(f"[错误] 声音 '{voice_name}' 未注册")
            if self.registered_voices:
                print(f"已注册的声音: {list(self.registered_voices.keys())}")
            else:
                print("还没有注册过任何声音，请先用 --register 注册")
            sys.exit(1)

        voice_info = self.registered_voices[voice_name]
        prompt_audio = voice_info["prompt_audio"]

        if not os.path.exists(prompt_audio):
            print(f"[错误] 参考音频已不存在: {prompt_audio}")
            print("请重新注册该声音: python voice_clone.py --register <名称> --audio <音频>")
            sys.exit(1)

        chunks = split_text_into_chunks(text, max_chars)
        print(f"文本共 {len(text)} 字符，切分为 {len(chunks)} 段合成")

        all_audio = []
        sample_rate = 22050

        for i, chunk in enumerate(chunks, 1):
            print(f"  [{i}/{len(chunks)}] {chunk[:50]}{'...' if len(chunk) > 50 else ''}")
            results = list(self.cosyvoice.inference_zero_shot(
                tts_text=chunk,
                prompt_text="",
                prompt_wav=prompt_audio
            ))
            if not results:
                print(f"  第 {i} 段合成失败，跳过")
                continue

            audio = results[-1]['tts_speech']
            all_audio.append(audio)

            # 段间添加 0.3 秒静音
            if i < len(chunks):
                silence = torch.zeros(1, int(sample_rate * 0.3))
                all_audio.append(silence)

        if not all_audio:
            raise RuntimeError("合成失败，没有生成任何音频")

        final_audio = torch.cat(all_audio, dim=1)
        torchaudio.save(output_path, final_audio, sample_rate)
        print(f"合成完成: {output_path}（共 {len(chunks)} 段）")
        return output_path

    def synthesize_simple(self, text, output_path="output.wav"):
        """
        使用预置音色合成（无需注册声音）

        Args:
            text: 要合成的中文文字
            output_path: 输出音频路径
        """
        import torchaudio

        print(f"正在使用预置音色合成...")
        print(f"  文字: {text[:60]}{'...' if len(text) > 60 else ''}")

        # 使用 inference_sft 模式（使用预置说话人）
        results = list(self.cosyvoice.inference_sft(
            tts_text=text,
            spk_id="中文女"  # 预置音色：中文女、中文男、日语男、粤语女、英文女、英文男、韩语女
        ))

        if not results:
            raise RuntimeError("合成失败，没有返回结果")

        final_result = results[-1]
        torchaudio.save(output_path, final_result['tts_speech'], 22050)

        print(f"合成完成: {output_path}")
        return output_path

    def synthesize_simple_long(self, text, output_path="output.wav", max_chars=300):
        """
        长文本分段合成，使用预置音色。

        Args:
            text: 要合成的长文本
            output_path: 输出音频路径
            max_chars: 每段最大字符数
        """
        import torchaudio
        import torch

        chunks = split_text_into_chunks(text, max_chars)
        print(f"文本共 {len(text)} 字符，切分为 {len(chunks)} 段合成")

        all_audio = []
        sample_rate = 22050

        for i, chunk in enumerate(chunks, 1):
            print(f"  [{i}/{len(chunks)}] {chunk[:50]}{'...' if len(chunk) > 50 else ''}")
            results = list(self.cosyvoice.inference_sft(
                tts_text=chunk,
                spk_id="中文女"
            ))
            if not results:
                print(f"  第 {i} 段合成失败，跳过")
                continue

            audio = results[-1]['tts_speech']
            all_audio.append(audio)

            # 段间添加 0.3 秒静音
            if i < len(chunks):
                silence = torch.zeros(1, int(sample_rate * 0.3))
                all_audio.append(silence)

        if not all_audio:
            raise RuntimeError("合成失败，没有生成任何音频")

        final_audio = torch.cat(all_audio, dim=1)
        torchaudio.save(output_path, final_audio, sample_rate)
        print(f"合成完成: {output_path}（共 {len(chunks)} 段）")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="声音克隆与语音合成 (基于 CosyVoice)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 注册声音（使用 1-3 段参考音频，自动保存到 voices.json）
  python voice_clone.py --register my_voice --audio sample1.wav sample2.wav

  # 使用注册的声音合成
  python voice_clone.py --text "你好，这是克隆的声音。" --voice my_voice -o output.wav

  # 使用预置音色直接合成（无需注册）
  python voice_clone.py --text "你好世界" --preset -o output.wav

  # 从文件读取文字合成（自动过滤时间轴）
  python voice_clone.py --text-file meeting.md --voice my_voice -o output.wav

  # 长文本分段合成（自动切分）
  python voice_clone.py --text-file long_article.md --voice my_voice --max-chars 250 -o output.wav

  # 查看已注册的声音列表
  python voice_clone.py --list-voices
        """
    )
    parser.add_argument("--register", help="注册声音的名称")
    parser.add_argument("--audio", nargs="+", help="参考音频文件路径（1-3 段，每段 3-10 秒）")
    parser.add_argument("--text", help="要合成的文字内容")
    parser.add_argument("--text-file", help="从文件读取文字内容（支持 .md/.txt，自动过滤时间轴）")
    parser.add_argument("--voice", help="已注册的声音名称")
    parser.add_argument("--output", "-o", default="output.wav", help="输出音频路径（默认: output.wav）")
    parser.add_argument("--model", default="pretrained_models/CosyVoice-300M",
                        help="CosyVoice 模型目录路径（默认: pretrained_models/CosyVoice-300M）")
    parser.add_argument("--preset", action="store_true",
                        help="使用预置音色合成（无需注册声音）")
    parser.add_argument("--list-voices", action="store_true",
                        help="列出所有已注册的声音")
    parser.add_argument("--max-chars", type=int, default=300,
                        help="长文本分段时每段最大字符数（默认: 300，超过自动分段）")

    args = parser.parse_args()

    # 列出已注册的声音
    if args.list_voices:
        cloner = VoiceCloner(args.model)
        cloner.list_voices()
        return

    # 注册声音模式
    if args.register:
        if not args.audio:
            print("[错误] 注册声音需要提供参考音频，使用 --audio sample.wav")
            sys.exit(1)
        cloner = VoiceCloner(args.model)
        cloner.register_voice(args.register, args.audio)
        return

    # 处理 --text-file，读取文件内容
    text = args.text
    if args.text_file:
        text = load_text_file(args.text_file)
        print(f"已读取文件: {args.text_file} ({len(text)} 字符)")

    # 合成模式
    if text:
        cloner = VoiceCloner(args.model)
        is_long = len(text) > args.max_chars

        if args.preset:
            if is_long:
                cloner.synthesize_simple_long(text, args.output, max_chars=args.max_chars)
            else:
                cloner.synthesize_simple(text, args.output)
        elif args.voice:
            if is_long:
                cloner.synthesize_long_text(text, args.voice, args.output, max_chars=args.max_chars)
            else:
                cloner.synthesize(text, args.voice, args.output)
        else:
            print("[错误] 请指定 --voice 使用克隆声音，或加 --preset 使用预置音色")
            sys.exit(1)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
