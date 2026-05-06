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

基于 **CosyVoice** 实现中文声音克隆：只需 1-3 段参考音频，即可克隆声音给任意文字配音。

### 安装 CosyVoice

声音克隆需要额外安装 CosyVoice 及其预训练模型（约 3GB）：

```bash
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

**硬件要求**：建议 NVIDIA GPU（显存 4GB+），CPU 也可运行但速度较慢。

### 使用流程

```bash
# 1. 注册声音（准备 1-3 段 3-10 秒的清晰人声）
python voice_clone.py --register my_voice --audio sample1.wav sample2.wav

# 2. 使用克隆的声音合成语音
python voice_clone.py --text "你好，这是用我克隆的声音合成的。" --voice my_voice -o output.wav

# 3. 也可以直接用预置音色（无需注册）
python voice_clone.py --text "你好世界" --preset -o output.wav
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
