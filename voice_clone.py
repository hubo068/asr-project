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
        self.registered_voices = {}
        print("模型加载完成")

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
            "audio_count": len(audio_paths)
        }
        print(f"声音 '{name}' 注册成功（使用了 {len(audio_paths)} 段参考音频）")

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
            raise ValueError(
                f"声音 '{voice_name}' 未注册，请先调用 register_voice\n"
                f"已注册的声音: {list(self.registered_voices.keys())}"
            )

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


def main():
    parser = argparse.ArgumentParser(
        description="声音克隆与语音合成 (基于 CosyVoice)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 注册声音（使用 1-3 段参考音频）
  python voice_clone.py --register my_voice --audio sample1.wav sample2.wav

  # 使用注册的声音合成
  python voice_clone.py --text "你好，这是克隆的声音。" --voice my_voice -o output.wav

  # 使用预置音色直接合成（无需注册）
  python voice_clone.py --text "你好世界" --preset -o output.wav
        """
    )
    parser.add_argument("--register", help="注册声音的名称")
    parser.add_argument("--audio", nargs="+", help="参考音频文件路径（1-3 段，每段 3-10 秒）")
    parser.add_argument("--text", help="要合成的文字内容")
    parser.add_argument("--voice", help="已注册的声音名称")
    parser.add_argument("--output", "-o", default="output.wav", help="输出音频路径（默认: output.wav）")
    parser.add_argument("--model", default="pretrained_models/CosyVoice-300M",
                        help="CosyVoice 模型目录路径（默认: pretrained_models/CosyVoice-300M）")
    parser.add_argument("--preset", action="store_true",
                        help="使用预置音色合成（无需注册声音）")

    args = parser.parse_args()

    # 注册声音模式
    if args.register:
        if not args.audio:
            print("[错误] 注册声音需要提供参考音频，使用 --audio sample.wav")
            sys.exit(1)
        cloner = VoiceCloner(args.model)
        cloner.register_voice(args.register, args.audio)
        return

    # 合成模式
    if args.text:
        cloner = VoiceCloner(args.model)

        if args.preset:
            # 使用预置音色
            cloner.synthesize_simple(args.text, args.output)
        elif args.voice:
            # 使用注册的克隆声音
            cloner.synthesize(args.text, args.voice, args.output)
        else:
            print("[错误] 请指定 --voice 使用克隆声音，或加 --preset 使用预置音色")
            sys.exit(1)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
