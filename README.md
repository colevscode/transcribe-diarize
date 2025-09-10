# Audio Transcription with Speaker Diarization

A Python tool for transcribing audio files with automatic speaker identification using Whisper and pyannote.audio.

## Features

- 🎤 **High-quality transcription** using OpenAI's Whisper models
- 👥 **Automatic speaker diarization** with pyannote.audio
- 🔧 **Flexible workflows** - test snippets, step-by-step processing, or one-command full pipeline
- 📝 **Customizable prompts** for different audio types (meetings, interviews, calls, etc.)
- ⚡ **Performance optimized** speaker diarization with memory-efficient processing
- 🏷️ **Generic speaker labels** - automatically assigns Speaker1, Speaker2, etc.

## Installation

```bash
# Install dependencies using Poetry
poetry install

# Or install with pip (requires Python 3.11+)
pip install faster-whisper pyannote.audio torch torchaudio
```

## Requirements

- **Python 3.11+**
- **FFmpeg** (for audio processing)
- **Hugging Face token** (optional, for gated diarization models) - save in `.hf` file

## Usage

### Quick Start

```bash
# Full transcription with speaker diarization
poetry run python transcribe.py audio.m4a --num-speakers 2

# Test with a 60-second snippet first
poetry run python transcribe.py test audio.m4a --duration 60 --num-speakers 2
```

## Commands

### 1. `test` - Test on Audio Snippet

Process a short snippet for quick testing and parameter validation.

```bash
poetry run python transcribe.py test <audio_file> [options]
```

**Options:**
- `--duration SECONDS` - Snippet length in seconds (default: 60)
- `--num-speakers N` - Expected number of speakers for faster processing
- `--prompt "TEXT"` - Custom transcription prompt
- `--no-diarize` - Skip speaker identification (faster)
- `-o FILE` - Output file path

**Examples:**
```bash
# Test 90-second snippet with 2 speakers
poetry run python transcribe.py test meeting.wav --duration 90 --num-speakers 2

# Test with custom prompt
poetry run python transcribe.py test interview.m4a --duration 30 --prompt "This is a job interview"
```

### 2. `default` - Full Pipeline (One Command)

Complete transcription + speaker diarization in one step.

```bash
poetry run python transcribe.py [default] <audio_file> [options]
```

**Options:**
- `--num-speakers N` - Expected number of speakers
- `--prompt "TEXT"` - Custom transcription prompt
- `--no-diarize` - Skip speaker identification
- `-o FILE` - Output file path

**Examples:**
```bash
# Simple usage (can omit 'default')
poetry run python transcribe.py call.wav --num-speakers 2

# Business meeting with custom prompt
poetry run python transcribe.py meeting.wav --prompt "This is a business meeting recording" --num-speakers 4
```

### 3. `transcribe` - Transcription Only

Generate raw transcription without speaker identification.

```bash
poetry run python transcribe.py transcribe <audio_file> [options]
```

**Options:**
- `--prompt "TEXT"` - Custom transcription prompt

**Example:**
```bash
poetry run python transcribe.py transcribe lecture.wav --prompt "This is an educational lecture"
```

### 4. `diarize` - Speaker Identification Only

Apply speaker diarization to existing transcription data.

```bash
poetry run python transcribe.py diarize <transcription_file> [options]
```

**Options:**
- `--num-speakers N` - Expected number of speakers
- `-o FILE` - Output file path

**Example:**
```bash
poetry run python transcribe.py diarize audio.transcript-segments.txt --num-speakers 2
```

## Global Options

- `--model {tiny,base,small,medium,large-v2,large-v3}` - Whisper model size (default: base)

## Custom Prompts

Prompts help Whisper understand the audio context for better transcription accuracy.

**Default prompts:**
- With speakers: `"This is a voice recording of a conversation between N people."`
- Without speakers: `"This is a voice recording."`

**Custom prompt examples:**
```bash
--prompt "This is a business meeting recording"
--prompt "This is a phone call between a client and contractor"  
--prompt "This is an interview between a journalist and expert"
--prompt "This is a medical consultation"
```

## Output Formats

### Transcript Files
- **`.transcript-segments.txt`** - Intermediate format with timestamps
- **`.transcript.txt`** - Final format with speaker labels

### Example Output
```
[0:00:04] Hello. Hey, can you hear me? This is Cole. Oh, okay.
[0:00:10 Speaker1] Yeah, I called you. I couldn't figure out how to record it on my normal phone call.
[0:00:15 Speaker2] That's okay. How are you doing?
[0:00:20 Speaker1] I'm doing alright. I just got back from this trip so I'm a little exhausted.
```

## Workflows

### 1. Quick Test Workflow
```bash
# Test parameters on a snippet
poetry run python transcribe.py test audio.wav --duration 120 --num-speakers 3

# If results look good, process full file
poetry run python transcribe.py audio.wav --num-speakers 3
```

### 2. Step-by-Step Workflow
```bash
# Step 1: Transcribe only
poetry run python transcribe.py transcribe audio.wav --prompt "This is a meeting"

# Step 2: Add speaker identification
poetry run python transcribe.py diarize audio.transcript-segments.txt --num-speakers 4
```

### 3. One-Command Workflow
```bash
# Everything in one step
poetry run python transcribe.py audio.wav --num-speakers 2 --prompt "This is a client call"
```

## Performance Tips

1. **Use `--num-speakers`** to constrain diarization for faster processing
2. **Test with snippets** first to validate parameters before processing long files
3. **Choose appropriate Whisper models**:
   - `tiny` - Fastest, lowest accuracy
   - `base` - Good balance (default)
   - `large-v3` - Highest accuracy, slowest

## File Support

Supports any audio format that FFmpeg can handle:
- `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.mp4`, etc.

## Troubleshooting

### Common Issues

1. **"No speakers detected"** - Try without `--num-speakers` constraint
2. **Poor speaker separation** - Ensure audio has clear speaker distinction
3. **Slow processing** - Use smaller Whisper model (`--model tiny`) or shorter test snippets
4. **FFmpeg errors** - Ensure FFmpeg is installed and audio file is not corrupted

### Requirements
- Ensure `.hf` file contains valid Hugging Face token for speaker diarization
- Audio files should have clear speech with minimal background noise
- For best results, use audio with distinct speakers

## Examples by Use Case

### Business Meetings
```bash
poetry run python transcribe.py meeting.wav \
  --prompt "This is a business meeting recording" \
  --num-speakers 4
```

### Phone Calls
```bash
poetry run python transcribe.py call.m4a \
  --prompt "This is a phone call between client and contractor" \
  --num-speakers 2
```

### Interviews
```bash
poetry run python transcribe.py interview.wav \
  --prompt "This is an interview between journalist and expert" \
  --num-speakers 2
```

### Lectures/Presentations
```bash
poetry run python transcribe.py lecture.wav \
  --prompt "This is an educational lecture" \
  --no-diarize
```

## License

MIT License - feel free to use and modify as needed.
