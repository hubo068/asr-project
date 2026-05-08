# 语音转 Markdown 文稿 (Whisper)

基于 OpenAI Whisper 的语音转录工具，支持中英文自动识别、智能分段、标点补全，可选的 LLM 后处理纠错，以及英文音频翻译为中文。

**核心设计**：语音识别 和 LLM 优化分为两步执行，识别过的文件可以直接优化，不用重新识别。

## 环境准备

### 1. 安装 FFmpeg

Whisper 依赖 FFmpeg 处理音频：

- **Windows**: `winget install Gyan.FFmpeg`
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 LLM

复制 `config_demo.json` 为 `config.json`，填入你的 API Key：

```bash
cp config_demo.json config.json
```

**单 Provider 格式（旧格式，仍兼容）：**

```json
{
  "type": "kimi",
  "base_url": "https://api.kimi.com/coding/",
  "api_key": "sk-你的密钥",
  "model": "Kimi Code"
}
```

**多 Provider 格式（推荐，支持命令行切换）：**

```json
{
  "providers": {
    "kimi": {
      "type": "kimi",
      "base_url": "https://api.kimi.com/coding/",
      "api_key": "sk-你的KIMI密钥",
      "model": "Kimi Code"
    },
    "openai": {
      "type": "openai",
      "api_key": "sk-你的OpenAI密钥",
      "model": "gpt-4o"
    }
  },
  "default": "kimi"
}
```

支持的 `type`：`kimi`、`anthropic`、`openai`

## 使用方法

### 第一步：语音识别

```bash
python transcribe.py your_audio.m4a
```

输出：`your_audio.md`

### 第二步：LLM 优化（分步执行）

如果已经生成了 `.md` 文件，直接优化，**不用重新识别**：

```bash
# 方式1：--refine 检测到 .md 已存在，自动跳过识别
python transcribe.py your_audio.m4a --refine

# 方式2：--only-refine 只做优化（要求 .md 必须已存在）
python transcribe.py your_audio.m4a --only-refine
```

优化后的文件保存为：`your_audio_refined.md`

### 翻译为中文（英文音频）

对于英文音频，可以使用 LLM 将转录结果翻译成中文：

```bash
# 方式1：--translate-to-zh 检测到 .md 已存在，自动跳过识别
python transcribe.py your_audio.m4a --translate-to-zh

# 方式2：已有英文文稿，直接翻译
python transcribe.py your_audio.m4a --only-refine --translate-to-zh
```

翻译后的文件保存为：`your_audio_zh.md`

也可以先校对英文，再翻译中文：

```bash
python transcribe.py your_audio.m4a --refine --translate-to-zh
```

### 切换 LLM Provider（多 Provider 配置时）

```bash
# 使用默认 provider（config.json 中 default 指定的）
python transcribe.py meeting.m4a --only-refine

# 手动指定 provider
python transcribe.py meeting.m4a --only-refine --llm-provider openai
```

### 一步完成（识别 + 优化）

如果 `.md` 不存在，`--refine` 会先识别再优化：

```bash
python transcribe.py your_audio.m4a --refine
```

### 完整示例（中文音频）

```bash
# 1. 识别
python transcribe.py meeting.m4a
# -> 生成 meeting.md

# 2. 检查 meeting.md，发现有几处同音字/人名错误

# 3. 直接优化（秒级完成，不重新识别）
python transcribe.py meeting.m4a --only-refine
# -> 生成 meeting_refined.md
```

### 完整示例（英文音频翻译）

```bash
# 1. 识别英文音频
python transcribe.py podcast.mp3
# -> 生成 podcast.md（英文原文）

# 2. 翻译成中文
python transcribe.py podcast.mp3 --translate-to-zh
# -> 生成 podcast_zh.md（中文翻译）

# 或一步完成：识别 + 翻译
python transcribe.py podcast.mp3 --translate-to-zh
```

## 分步工作流示意

### 中文音频

```
音频文件
    |
[步骤 1] Whisper 识别  ->  your_audio.md
    |  （人工检查，或发现错误）
[步骤 2] LLM 优化      ->  your_audio_refined.md
```

### 英文音频（翻译）

```
音频文件
    |
[步骤 1] Whisper 识别  ->  your_audio.md（英文原文）
    |
[步骤 2] LLM 翻译      ->  your_audio_zh.md（中文翻译）
    |  （可选：先校对再翻译）
[可选] LLM 优化        ->  your_audio_refined.md（英文校对稿）
                         ->  your_audio_zh.md（中文翻译）
```

## 模型说明

| 模型 | 速度 | 准确率 | 内存 |
|------|------|--------|------|
| tiny | 最快 | 一般 | ~1 GB |
| base | 快 | 较好 | ~1 GB |
| small | 中等 | 好 | ~2 GB |
| medium | 较慢 | 很好 | ~5 GB |
| large | 最慢 | 最佳 | ~10 GB |

## 输出示例

`your_audio.md`（原始识别）：

```markdown
# 语音转录文稿

- **源文件**: `meeting.m4a`
- **识别语言**: zh
- **总时长**: 00:32:15

---

`00:00:05` 大家好，今天我们讨论一下下个季度的产品规划。

`00:00:22` 首先由设计团队介绍一下最新的交互方案，这个方案在用户测试中的反馈非常好。
```

`your_audio_refined.md`（LLM 优化后）：

```markdown
# 语音转录文稿

- **源文件**: `meeting.m4a`
- **识别语言**: zh
- **总时长**: 00:32:15

---

`00:00:05` 大家好，今天我们讨论一下下个季度的产品规划。

`00:00:22` 首先由设计团队介绍一下最新的交互方案，这个方案在用户测试中的反馈非常好。
```

`your_audio_zh.md`（LLM 翻译后，英文音频）：

```markdown
# 语音转录文稿

- **源文件**: `podcast.mp3`
- **识别语言**: en
- **总时长**: 00:32:15

---

`00:00:05` 大家好，欢迎来到本期播客。今天我们来讨论人工智能领域的最新进展。

`00:00:22` 首先，让我们看看大语言模型在过去几个月中的发展，以及它们如何改变我们的工作方式。
```

## 声音克隆（实验功能）

基于 **CosyVoice** 实现中文声音克隆。推荐使用带真实参考文本的 zero-shot 克隆：参考音频越干净、参考文本越准确，声音相似度通常越好。

### 安装 CosyVoice

声音克隆需要额外安装 CosyVoice 及其预训练模型（约 3GB）：

```bash
# 建议先进入项目使用的虚拟环境
conda activate asr

# 1. 安装 CosyVoice
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
pip install -e .
cd ..

# 2. 下载预训练模型
# 方式 A：使用 modelscope 下载（推荐国内用户）
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')"

# 方式 B：从 HuggingFace 下载（需要网络通畅）
# git clone https://huggingface.co/FunAudioLLM/CosyVoice-300M pretrained_models/CosyVoice-300M
```

**GPU 加速**：建议 NVIDIA GPU（显存 4GB+）。RTX 50 系列等新显卡需要支持对应架构的 PyTorch CUDA 包，例如 CUDA 12.8+：

```bash
python -m pip uninstall -y torch torchaudio torchvision
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 检查当前环境、PyTorch、CUDA、显卡架构是否匹配
python voice_clone.py --cuda-diagnostics
```

如果日志提示 `onnxruntime CUDAExecutionProvider is not installed`，说明 PyTorch TTS 主体会走 GPU，但部分 ONNX 前端仍在 CPU 上运行；需要更完整加速时再安装 `onnxruntime-gpu`。

### 使用流程

准备参考音频时，优先选 6-10 秒、单人、无背景音乐、无混响、语气稳定的 WAV。`--prompt-text` / `--prompt-texts-file` 必须和参考音频里真实说出的内容尽量逐字一致。

```bash
# 1. 单段参考音频注册
python voice_clone.py --register my_voice --audio sample1.wav --prompt-text "这里填写 sample1.wav 里真实说出的文字"

# 2. 多段参考音频注册：prompt_texts.txt 每行对应一个 --audio
python voice_clone.py --register my_voice --audio sample1.wav sample2.wav sample3.wav --prompt-texts-file prompt_texts.txt

# 3. 为每段参考音频生成候选试听，选最像的一段
python voice_clone.py --text "你好，这是一次声音克隆测试。" --voice my_voice --device cuda --no-fp16 --no-load-jit --prompt-candidates -o candidate.wav

# 4. 查看已注册声音和每段参考音频的编号
python voice_clone.py --list-voices
```

**注册声音自动持久化**：注册信息自动保存到 `voices.json`，下次直接使用 `--voice` 即可，无需重新注册。

### 给文稿配音（支持长文本）

```bash
# 使用第 0 段参考音频给 Markdown 文稿配音
python voice_clone.py --text-file meeting.md --voice my_voice --prompt-index 0 --device cuda --no-fp16 --no-load-jit -o meeting.wav

# 给本项目中的 llm01_zh.md 配音
python voice_clone.py --text-file "D:\work\07ai-app\asr_project\llm01_zh.md" --voice my_voice --prompt-index 0 --device cuda --no-fp16 --no-load-jit --max-chars 120 -o llm01.wav

# 长文本自动分段合成；相似度不稳时可把分段调短一些
python voice_clone.py --text-file article.md --voice my_voice --prompt-index 0 --device cuda --no-fp16 --no-load-jit --max-chars 100 -o article.wav
```

`--text-file` 会自动过滤以下内容：
- Markdown 标题、元信息、分隔线
- 时间轴（如 `` `00:01:23` `` 或 `00:01:23`）

长文本（超过 `--max-chars`）会自动分段合成，段间插入 0.3 秒静音，最终拼接为完整音频。

### 提升克隆相似度

- 给每段参考音频都提供准确参考文本，优先使用 `--prompt-texts-file`。
- 使用 `--prompt-candidates` 试听不同参考音频，选最像的一段后用 `--prompt-index` 固定它。
- 参考音频不要太长，6-10 秒通常比几十秒杂音频更稳定。
- 参考音频的语速、情绪、语言尽量接近目标文稿。
- 生成爆音、静音、滴答声时，优先使用 `--no-fp16 --no-load-jit` 重新生成。
- 长文配音建议 `--max-chars 80-150`，每段过长时音色和韵律更容易漂移。

### 音频格式要求

CosyVoice 的参考音频**仅支持 WAV 格式**（16kHz、单声道最佳）。如果手边是 m4a/mp3，先用 FFmpeg 转换：

```bash
ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav
```

### 预置音色

使用 `--preset` 时，内置音色包括：`中文女`、`中文男`、`日语男`、`粤语女`、`英文女`、`英文男`、`韩语女`。

## 常见问题

**Q: 转录结果中同音字/人名错误较多怎么办？**
A: 使用 `--only-refine` 对已有 `.md` 进行 LLM 优化，不用重新跑 Whisper。

**Q: 不想用 LLM，还有其他方法吗？**
A: 可以在 `transcribe.py` 的 `initial_prompt` 中预先放入常见人名和术语，Whisper 会参考这些词汇进行识别。

**Q: 支持哪些音频格式？**
A: 所有 FFmpeg 支持的格式（mp3、wav、m4a、flac、ogg、wma、aac 等）。

**Q: 声音克隆报错 `Format not recognised`？**
A: CosyVoice 的参考音频仅支持 WAV 格式。请先用 FFmpeg 转换：`ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav`。

**Q: 给长文章配音会截断吗？**
A: 不会。`voice_clone.py` 会自动将长文本按语义切分，逐段合成后拼接为完整音频。通过 `--max-chars` 控制每段长度（默认 300 字）。

**Q: 注册的声音下次还要重新注册吗？**
A: 不需要。注册信息自动保存到 `voices.json`，下次直接用 `--voice` 即可。如果参考音频文件被删除，对应的声音会自动从列表中移除。
