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
import subprocess
from datetime import datetime

# 将 CosyVoice 子模块加入 Python 路径
_COSYVOICE_PATH = os.path.join(os.path.dirname(__file__), "CosyVoice")
if os.path.exists(_COSYVOICE_PATH) and _COSYVOICE_PATH not in sys.path:
    sys.path.insert(0, _COSYVOICE_PATH)

# 将 CosyVoice 子模块 Matcha-TTS 加入 Python 路径
_MATCHA_TTS_PATH = os.path.join(os.path.dirname(__file__), "CosyVoice", "third_party", "Matcha-TTS")
if os.path.exists(_MATCHA_TTS_PATH) and _MATCHA_TTS_PATH not in sys.path:
    sys.path.insert(0, _MATCHA_TTS_PATH)


def check_cosyvoice():
    """检查 CosyVoice 是否已安装"""
    try:
        from cosyvoice.cli.cosyvoice import AutoModel
        return True
    except ImportError:
        return False


def check_model(model_path):
    """检查预训练模型是否存在"""
    return os.path.exists(model_path)


VOICES_FILE = "voices.json"


def load_registered_voices(voices_file=VOICES_FILE):
    """Load registered voice metadata without initializing CosyVoice."""
    if os.path.exists(voices_file):
        try:
            with open(voices_file, "r", encoding="utf-8") as f:
                voices = json.load(f)
            valid_voices = {}
            for name, info in voices.items():
                prompt_audio = info.get("prompt_audio", "")
                prompts = info.get("prompts", [])
                valid_prompt = prompt_audio or (prompts[0].get("audio", "") if prompts else "")
                if os.path.exists(valid_prompt):
                    valid_voices[name] = info
                else:
                    print(f"[警告] 声音 '{name}' 的参考音频已不存在，已忽略")
            return valid_voices
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[警告] 读取 {voices_file} 失败: {e}")
    return {}


def print_registered_voices(voices):
    if not voices:
        print("还没有注册任何声音")
        return

    print(f"已注册的声音（共 {len(voices)} 个）:")
    for name, info in voices.items():
        prompts = info.get("prompts")
        if prompts:
            text_count = sum(1 for p in prompts if p.get("text"))
            default_audio = prompts[0].get("audio", "")
            print(f"  - {name}: {default_audio} ({len(prompts)} 段音频，{text_count} 段有参考文本)")
            for i, prompt in enumerate(prompts):
                status = "text=yes" if prompt.get("text") else "text=no"
                print(f"      [{i}] {prompt.get('audio', '')} ({status})")
        else:
            status = "text=yes" if info.get("prompt_text", "") else "text=no"
            print(f"  - {name}: {info['prompt_audio']} ({info['audio_count']} 段音频，{status})")


def get_cuda_arch_status(torch_module):
    if not torch_module.cuda.is_available():
        return {
            "available": False,
            "device_name": "",
            "capability": None,
            "device_arch": "",
            "compiled_arches": [],
            "supported": False,
        }

    capability = torch_module.cuda.get_device_capability(0)
    device_arch = f"sm_{capability[0]}{capability[1]}"
    try:
        compiled_arches = torch_module.cuda.get_arch_list()
    except Exception:
        compiled_arches = []
    return {
        "available": True,
        "device_name": torch_module.cuda.get_device_name(0),
        "capability": capability,
        "device_arch": device_arch,
        "compiled_arches": compiled_arches,
        "supported": (not compiled_arches) or device_arch in compiled_arches,
    }


def print_cuda_diagnostics():
    """Print enough runtime details to diagnose CUDA/PyTorch environment issues."""
    print("Python executable:", sys.executable)
    try:
        import torch
    except Exception as exc:
        print(f"torch import failed: {exc}")
        return False

    print("torch version:", torch.__version__)
    print("torch file:", getattr(torch, "__file__", "unknown"))
    print("torch CUDA build:", torch.version.cuda)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.cuda.device_count:", torch.cuda.device_count())
    arch_status = get_cuda_arch_status(torch)
    if arch_status["available"]:
        print("torch CUDA device:", arch_status["device_name"])
        print("torch CUDA capability:", arch_status["capability"])
        print("torch CUDA device arch:", arch_status["device_arch"])
        print("torch compiled CUDA arch list:", arch_status["compiled_arches"])
        print("torch supports this GPU arch:", arch_status["supported"])
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode == 0:
            first_lines = "\n".join(result.stdout.splitlines()[:12])
            print("nvidia-smi:")
            print(first_lines)
        else:
            print("nvidia-smi failed:", result.stderr.strip() or result.stdout.strip())
    except Exception as exc:
        print(f"nvidia-smi unavailable: {exc}")
    return torch.cuda.is_available() and arch_status["supported"]


def build_cuda_unavailable_message(torch_module):
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    activate_hint = f"  conda activate {env_name}\n" if env_name else "  conda activate <your-env>\n"
    return (
        "CUDA was requested, but torch.cuda.is_available() is False.\n"
        f"Python executable: {sys.executable}\n"
        f"torch version: {getattr(torch_module, '__version__', 'unknown')}\n"
        f"torch file: {getattr(torch_module, '__file__', 'unknown')}\n"
        f"torch CUDA build: {getattr(torch_module.version, 'cuda', None)}\n"
        "Install/use a CUDA-enabled PyTorch build in your intended environment, then run:\n"
        f"{activate_hint}"
        "  python voice_clone.py --text-file llm01_zh.md --voice my_voice --device cuda -o meeting.wav\n"
        "You can inspect the active runtime with:\n"
        "  python voice_clone.py --cuda-diagnostics"
    )


def build_cuda_arch_unsupported_message(torch_module, arch_status):
    return (
        "CUDA is available, but this PyTorch build does not include kernels for your GPU architecture.\n"
        f"Python executable: {sys.executable}\n"
        f"torch version: {getattr(torch_module, '__version__', 'unknown')}\n"
        f"torch CUDA build: {getattr(torch_module.version, 'cuda', None)}\n"
        f"GPU: {arch_status['device_name']}\n"
        f"GPU arch: {arch_status['device_arch']}\n"
        f"PyTorch compiled arch list: {arch_status['compiled_arches']}\n"
        "Install a newer PyTorch CUDA wheel that supports this GPU, for example CUDA 12.8+ builds:\n"
        "  python -m pip uninstall -y torch torchaudio torchvision\n"
        "  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128\n"
        "Then verify with:\n"
        "  python voice_clone.py --cuda-diagnostics"
    )


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


def load_prompt_texts_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"参考文本文件不存在: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.read().splitlines()]


def save_wav(output_path, speech, sample_rate):
    import soundfile as sf

    audio = speech.detach().cpu().float()
    if audio.dim() == 2:
        audio = audio.transpose(0, 1)
    audio = audio.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    mean = audio.mean().item()
    std = audio.std(unbiased=False).item()
    if abs(mean) > 0.05 and std > 1e-4:
        print(f"[warning] output audio has DC offset mean={mean:.4f}; removing it before saving.")
        audio = audio - audio.mean(dim=0, keepdim=True)

    peak = audio.abs().max().item()
    if abs(mean) > 0.1 or std < 1e-4:
        print(f"[warning] unusual output audio stats: mean={mean:.4f}, std={std:.6f}, peak={peak:.4f}")
    if peak >= 0.01:
        if peak < 0.2:
            print(f"[warning] output audio peak {peak:.4f}; normalizing quiet audio.")
        elif peak > 0.99:
            print(f"[warning] output audio peak {peak:.4f}; normalizing to avoid clipping.")
        audio = audio / peak * 0.95
    elif peak > 0:
        print(f"[warning] output audio peak {peak:.6f} is too low; saved without boosting noise.")
    else:
        print("[warning] output audio is completely silent.")

    sf.write(output_path, audio.numpy(), sample_rate, subtype="PCM_16")


def load_wav_with_soundfile(wav, target_sr, min_sr=16000):
    import soundfile as sf
    import torch
    import torchaudio

    speech, sample_rate = sf.read(wav, dtype="float32", always_2d=True)
    speech = torch.from_numpy(speech).transpose(0, 1).mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate >= min_sr, "wav sample rate {} must be greater than {}".format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech


def patch_cosyvoice_audio_io():
    """Avoid torchaudio.load's torchcodec dependency in newer torchaudio builds."""
    try:
        import cosyvoice.utils.file_utils as file_utils
        import cosyvoice.cli.frontend as frontend
    except Exception as exc:
        print(f"[warning] Could not patch CosyVoice audio loader: {exc}")
        return

    file_utils.load_wav = load_wav_with_soundfile
    frontend.load_wav = load_wav_with_soundfile


class VoiceCloner:
    def __init__(
        self,
        model_dir="pretrained_models/CosyVoice-300M",
        device="auto",
        fp16=None,
        load_jit=None,
        load_trt=False,
        trt_concurrent=1,
    ):
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

        from cosyvoice.cli.cosyvoice import AutoModel
        import torch
        patch_cosyvoice_audio_io()

        device = device.lower()
        if device not in {"auto", "cuda"}:
            raise ValueError("device must be 'auto' or 'cuda'")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(build_cuda_unavailable_message(torch))

        use_cuda = torch.cuda.is_available()
        arch_status = get_cuda_arch_status(torch)
        if use_cuda and not arch_status["supported"]:
            raise RuntimeError(build_cuda_arch_unsupported_message(torch, arch_status))

        if use_cuda:
            torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

        enable_fp16 = use_cuda if fp16 is None else fp16
        enable_jit = use_cuda if load_jit is None else load_jit

        self.torch = torch
        self.runtime = {
            "device": "cuda" if use_cuda else "cpu",
            "fp16": enable_fp16,
            "load_jit": enable_jit,
            "load_trt": load_trt,
        }

        print(f"正在加载模型: {model_dir}")
        if use_cuda:
            print(f"Runtime device: cuda ({torch.cuda.get_device_name(torch.cuda.current_device())})")
        else:
            print("Runtime device: cpu")
        print(f"Acceleration: fp16={enable_fp16}, jit={enable_jit}, trt={load_trt}")
        self._warn_if_onnx_cuda_missing(use_cuda)

        try:
            self.cosyvoice = AutoModel(
                model_dir=model_dir,
                load_jit=enable_jit,
                load_trt=load_trt,
                fp16=enable_fp16,
                trt_concurrent=trt_concurrent,
            )
        except Exception:
            if enable_jit and load_jit is None:
                print("[warning] JIT model load failed; retrying without JIT.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                enable_jit = False
                self.runtime["load_jit"] = False
                self.cosyvoice = AutoModel(
                    model_dir=model_dir,
                    load_jit=False,
                    load_trt=load_trt,
                    fp16=enable_fp16,
                    trt_concurrent=trt_concurrent,
                )
            else:
                raise

        model_device = getattr(getattr(self.cosyvoice, "model", None), "device", None)
        if model_device is not None:
            print(f"Model device: {model_device}")
        self.sample_rate = getattr(self.cosyvoice, "sample_rate", 22050)
        self.registered_voices = self._load_voices()
        print("模型加载完成")

    def _warn_if_onnx_cuda_missing(self, use_cuda):
        if not use_cuda:
            return
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
        except Exception as exc:
            print(f"[warning] Could not inspect onnxruntime providers: {exc}")
            return
        if "CUDAExecutionProvider" not in providers:
            print(
                "[warning] onnxruntime CUDAExecutionProvider is not installed; "
                "PyTorch TTS runs on GPU, but ONNX frontend steps will stay on CPU. "
                "Install onnxruntime-gpu for full GPU acceleration."
            )

    def _load_voices(self):
        """从文件加载已注册的声音列表"""
        return load_registered_voices()

    def _save_voices(self):
        """将已注册的声音列表保存到文件"""
        with open(VOICES_FILE, "w", encoding="utf-8") as f:
            json.dump(self.registered_voices, f, ensure_ascii=False, indent=2)

    def _clone_inference(self, tts_text, prompt_audio, prompt_text):
        if prompt_text:
            return list(self.cosyvoice.inference_zero_shot(
                tts_text=tts_text,
                prompt_text=prompt_text,
                prompt_wav=prompt_audio,
            ))
        print("[warning] this voice has no prompt_text; using cross_lingual fallback.")
        return list(self.cosyvoice.inference_cross_lingual(
            tts_text=tts_text,
            prompt_wav=prompt_audio,
        ))

    def _get_voice_prompt(self, voice_info, prompt_index=0):
        prompts = voice_info.get("prompts")
        if prompts:
            if prompt_index < 0 or prompt_index >= len(prompts):
                raise IndexError(f"prompt_index {prompt_index} 超出范围，当前共有 {len(prompts)} 段参考音频")
            prompt = prompts[prompt_index]
            return prompt.get("audio", ""), prompt.get("text", "")

        if prompt_index not in (0, None):
            raise IndexError("旧版 voices.json 只保存了 1 段参考音频，prompt_index 只能为 0")
        return voice_info.get("prompt_audio", ""), voice_info.get("prompt_text", "")

    def register_voice(self, name, audio_paths, prompt_text="", prompt_texts=None):
        """
        注册一个声音（提取声纹）

        Args:
            name: 声音名称，如 "my_voice"
            audio_paths: 参考音频文件路径列表，1-3 段
        """
        if not audio_paths:
            raise ValueError("至少需要提供 1 段参考音频")

        prompt_texts = prompt_texts or []
        if prompt_text and not prompt_texts:
            prompt_texts = [prompt_text]

        prompts = []
        for i, audio_path in enumerate(audio_paths):
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"音频文件不存在: {audio_path}")
            prompts.append({
                "audio": audio_path,
                "text": prompt_texts[i].strip() if i < len(prompt_texts) else "",
            })

        prompt_audio = prompts[0]["audio"]
        print(f"正在注册声音 '{name}'，使用 {len(prompts)} 段参考音频，默认参考音频: {prompt_audio}")

        self.registered_voices[name] = {
            "prompt_audio": prompt_audio,
            "prompt_text": prompts[0]["text"],
            "prompts": prompts,
            "audio_count": len(audio_paths),
            "registered_at": datetime.now().isoformat()
        }
        self._save_voices()
        print(f"声音 '{name}' 注册成功（使用了 {len(audio_paths)} 段参考音频）")
        print(f"已保存到 {VOICES_FILE}")

    def list_voices(self):
        """列出所有已注册的声音"""
        print_registered_voices(self.registered_voices)

    def synthesize(self, text, voice_name, output_path="output.wav", prompt_index=0):
        """
        使用注册的声音合成语音

        Args:
            text: 要合成的中文文字
            voice_name: 已注册的声音名称
            output_path: 输出音频路径（默认 output.wav）
        """
        if voice_name not in self.registered_voices:
            print(f"[错误] 声音 '{voice_name}' 未注册")
            if self.registered_voices:
                print(f"已注册的声音: {list(self.registered_voices.keys())}")
            else:
                print("还没有注册过任何声音，请先用 --register 注册")
            sys.exit(1)

        voice_info = self.registered_voices[voice_name]
        prompt_audio, prompt_text = self._get_voice_prompt(voice_info, prompt_index)

        if not os.path.exists(prompt_audio):
            print(f"[错误] 参考音频已不存在: {prompt_audio}")
            print("请重新注册该声音: python voice_clone.py --register <名称> --audio <音频>")
            sys.exit(1)

        print(f"正在合成语音...")
        print(f"  文字: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"  声音: {voice_name}")
        print(f"  参考音频[{prompt_index}]: {prompt_audio}")

        results = self._clone_inference(text, prompt_audio, prompt_text)

        if not results:
            raise RuntimeError("合成失败，没有返回结果")

        # 保存最后一段结果
        final_result = results[-1]
        save_wav(output_path, final_result['tts_speech'], self.sample_rate)

        print(f"合成完成: {output_path}")
        return output_path

    def synthesize_long_text(self, text, voice_name, output_path="output.wav", max_chars=300, prompt_index=0):
        """
        长文本分段合成，使用注册的声音。

        Args:
            text: 要合成的长文本
            voice_name: 已注册的声音名称
            output_path: 输出音频路径
            max_chars: 每段最大字符数
        """
        import torch

        if voice_name not in self.registered_voices:
            print(f"[错误] 声音 '{voice_name}' 未注册")
            if self.registered_voices:
                print(f"已注册的声音: {list(self.registered_voices.keys())}")
            else:
                print("还没有注册过任何声音，请先用 --register 注册")
            sys.exit(1)

        voice_info = self.registered_voices[voice_name]
        prompt_audio, prompt_text = self._get_voice_prompt(voice_info, prompt_index)

        if not os.path.exists(prompt_audio):
            print(f"[错误] 参考音频已不存在: {prompt_audio}")
            print("请重新注册该声音: python voice_clone.py --register <名称> --audio <音频>")
            sys.exit(1)

        chunks = split_text_into_chunks(text, max_chars)
        print(f"文本共 {len(text)} 字符，切分为 {len(chunks)} 段合成")
        print(f"使用参考音频[{prompt_index}]: {prompt_audio}")

        all_audio = []
        sample_rate = self.sample_rate

        for i, chunk in enumerate(chunks, 1):
            print(f"  [{i}/{len(chunks)}] {chunk[:50]}{'...' if len(chunk) > 50 else ''}")
            results = self._clone_inference(chunk, prompt_audio, prompt_text)
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
        save_wav(output_path, final_audio, sample_rate)
        print(f"合成完成: {output_path}（共 {len(chunks)} 段）")
        return output_path

    def synthesize_simple(self, text, output_path="output.wav"):
        """
        使用预置音色合成（无需注册声音）

        Args:
            text: 要合成的中文文字
            output_path: 输出音频路径
        """
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
        save_wav(output_path, final_result['tts_speech'], self.sample_rate)

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
        import torch

        chunks = split_text_into_chunks(text, max_chars)
        print(f"文本共 {len(text)} 字符，切分为 {len(chunks)} 段合成")

        all_audio = []
        sample_rate = self.sample_rate

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
        save_wav(output_path, final_audio, sample_rate)
        print(f"合成完成: {output_path}（共 {len(chunks)} 段）")
        return output_path

    def synthesize_prompt_candidates(self, text, voice_name, output_path="candidate.wav"):
        if voice_name not in self.registered_voices:
            print(f"[错误] 声音 '{voice_name}' 未注册")
            sys.exit(1)

        voice_info = self.registered_voices[voice_name]
        prompts = voice_info.get("prompts") or [{
            "audio": voice_info.get("prompt_audio", ""),
            "text": voice_info.get("prompt_text", ""),
        }]
        root, ext = os.path.splitext(output_path)
        ext = ext or ".wav"
        outputs = []
        for i, prompt in enumerate(prompts):
            prompt_audio = prompt.get("audio", "")
            prompt_text = prompt.get("text", "")
            if not os.path.exists(prompt_audio):
                print(f"[warning] skip prompt[{i}], file not found: {prompt_audio}")
                continue
            candidate_path = f"{root}_p{i}{ext}"
            print(f"[candidate {i}] {prompt_audio} -> {candidate_path}")
            results = self._clone_inference(text, prompt_audio, prompt_text)
            if not results:
                print(f"[warning] prompt[{i}] synthesis failed")
                continue
            save_wav(candidate_path, results[-1]["tts_speech"], self.sample_rate)
            outputs.append(candidate_path)
        if outputs:
            print("候选样本已生成:")
            for path in outputs:
                print(f"  - {path}")
        return outputs


def main():
    parser = argparse.ArgumentParser(
        description="声音克隆与语音合成 (基于 CosyVoice)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查 CUDA/PyTorch 环境
  python voice_clone.py --cuda-diagnostics

  # 注册声音（参考文本要和音频内容一致，自动保存到 voices.json）
  python voice_clone.py --register my_voice --audio sample1.wav --prompt-text "sample1.wav 中真实说出的文字"

  # 多段参考音频注册，每行参考文本对应一个 --audio
  python voice_clone.py --register my_voice --audio sample1.wav sample2.wav --prompt-texts-file prompt_texts.txt

  # 生成每段参考音频的候选试听
  python voice_clone.py --text "你好，这是克隆的声音。" --voice my_voice --prompt-candidates -o candidate.wav

  # 使用预置音色直接合成（无需注册）
  python voice_clone.py --text "你好世界" --preset -o output.wav

  # 从文件读取文字合成（自动过滤时间轴）
  python voice_clone.py --text-file meeting.md --voice my_voice --prompt-index 0 --device cuda --no-fp16 --no-load-jit -o output.wav

  # 长文本分段合成（自动切分）
  python voice_clone.py --text-file long_article.md --voice my_voice --max-chars 120 -o output.wav

  # 查看已注册的声音列表
  python voice_clone.py --list-voices
        """
    )
    parser.add_argument("--register", help="注册声音的名称")
    parser.add_argument("--audio", nargs="+", help="参考音频文件路径（1-3 段，每段 3-10 秒）")
    parser.add_argument("--prompt-text", help="参考音频中真实说出的文字，用于 zero-shot 克隆")
    parser.add_argument("--prompt-text-file", help="从文本文件读取参考音频对应文字")
    parser.add_argument("--prompt-texts-file", help="多段参考音频对应文本，每行对应一个 --audio")
    parser.add_argument("--prompt-index", type=int, default=0,
                        help="合成时选择第几段参考音频（从 0 开始，默认 0）")
    parser.add_argument("--prompt-candidates", action="store_true",
                        help="为每段参考音频各生成一个候选试听样本")
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

    parser.add_argument("--device", choices=["auto", "cuda"], default="auto",
                        help="Inference device. auto uses CUDA when available; cuda fails if no CUDA GPU is visible.")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None,
                        help="Use FP16 autocast on CUDA. Default: enabled when CUDA is available.")
    parser.add_argument("--load-jit", action=argparse.BooleanOptionalAction, default=None,
                        help="Load JIT optimized modules. Default: enabled when CUDA is available.")
    parser.add_argument("--load-trt", action="store_true",
                        help="Use TensorRT for flow decoder when TensorRT is installed and engine can be built.")
    parser.add_argument("--trt-concurrent", type=int, default=1,
                        help="TensorRT concurrent context count.")
    parser.add_argument("--cuda-diagnostics", action="store_true",
                        help="Print Python/PyTorch/CUDA diagnostics and exit.")

    args = parser.parse_args()
    if args.cuda_diagnostics:
        ok = print_cuda_diagnostics()
        sys.exit(0 if ok else 1)

    cloner_options = {
        "model_dir": args.model,
        "device": args.device,
        "fp16": args.fp16,
        "load_jit": args.load_jit,
        "load_trt": args.load_trt,
        "trt_concurrent": args.trt_concurrent,
    }

    # 列出已注册的声音
    if args.list_voices:
        print_registered_voices(load_registered_voices())
        return

    # 注册声音模式
    if args.register:
        if not args.audio:
            print("[错误] 注册声音需要提供参考音频，使用 --audio sample.wav")
            sys.exit(1)
        prompt_text = args.prompt_text or ""
        if args.prompt_text_file:
            prompt_text = load_text_file(args.prompt_text_file)
        prompt_texts = load_prompt_texts_file(args.prompt_texts_file) if args.prompt_texts_file else None
        cloner = VoiceCloner(**cloner_options)
        cloner.register_voice(args.register, args.audio, prompt_text=prompt_text, prompt_texts=prompt_texts)
        return

    # 处理 --text-file，读取文件内容
    text = args.text
    if args.text_file:
        text = load_text_file(args.text_file)
        print(f"已读取文件: {args.text_file} ({len(text)} 字符)")

    # 合成模式
    if text:
        cloner = VoiceCloner(**cloner_options)
        is_long = len(text) > args.max_chars

        if args.preset:
            if is_long:
                cloner.synthesize_simple_long(text, args.output, max_chars=args.max_chars)
            else:
                cloner.synthesize_simple(text, args.output)
        elif args.voice:
            if args.prompt_candidates:
                cloner.synthesize_prompt_candidates(text, args.voice, args.output)
            elif is_long:
                cloner.synthesize_long_text(text, args.voice, args.output, max_chars=args.max_chars, prompt_index=args.prompt_index)
            else:
                cloner.synthesize(text, args.voice, args.output, prompt_index=args.prompt_index)
        else:
            print("[错误] 请指定 --voice 使用克隆声音，或加 --preset 使用预置音色")
            sys.exit(1)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
