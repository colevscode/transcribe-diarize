#!/usr/bin/env python3
"""
Transcription script with speaker diarization using faster-whisper and pyannote-audio
"""

import os
import sys
import argparse
import tempfile
import subprocess
import json
import pickle
import time
from pathlib import Path
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torch

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"

def transcribe_audio(audio_path: str, model_size: str = "base", num_speakers: int = None, 
                    speaker_names: list = None, silent: bool = False):
    """
    Audio transcription with speaker-aware prompts
    
    Args:
        audio_path: Path to the audio file  
        model_size: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
        num_speakers: Expected number of speakers (affects prompt)
        speaker_names: List of speaker names (affects prompt)
        silent: If True, don't print progress messages
        
    Returns:
        tuple: (segments_list, language_info) - raw Whisper segments and detection info
    """
    
    if not os.path.exists(audio_path):
        if not silent:
            print(f"Error: Audio file not found: {audio_path}")
        return [], {}
    
    if not silent:
        print(f"Loading Whisper model: {model_size}")
    whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    # Create dynamic initial prompt based on speaker count
    initial_prompt = "This is a phone conversation"
    if num_speakers:
        if num_speakers == 2:
            initial_prompt += " between two people"
        else:
            initial_prompt += f" between {num_speakers} people"
    elif speaker_names and len(speaker_names) > 0:
        if len(speaker_names) == 2:
            initial_prompt += " between two people"
        else:
            initial_prompt += f" between {len(speaker_names)} people"
    initial_prompt += "."
    
    if not silent:
        print(f"Transcribing audio...")
    
    # Use temperature and natural breaks to encourage segmentation
    transcribe_params = {
        'beam_size': 1,                    # Lower beam size for more diverse outputs
        'initial_prompt': initial_prompt,
        'vad_filter': False,               # Disable VAD completely
        'word_timestamps': True,           # Enable word-level timestamps 
        'temperature': 0.2,                # Slight temperature for variation
        'condition_on_previous_text': False, # Don't condition - allows fresh starts
        'compression_ratio_threshold': 2.0,  # Lower threshold encourages shorter segments
        'log_prob_threshold': -0.8,        # Stricter confidence threshold
        'no_speech_threshold': 0.8,        # Higher threshold for speech detection
        'patience': 1.0,                   # Less patience = more frequent breaks
        'length_penalty': 0.8              # Slight penalty for longer segments
    }
        
    segments, info = whisper_model.transcribe(audio_path, **transcribe_params)
    segments_list = list(segments)
    
    if not silent:
        print(f"✅ Detected language '{info.language}' with probability {info.language_probability}")
        print(f"📊 Generated {len(segments_list)} segments")
    
    return segments_list, info

def extract_audio_snippet(audio_path: str, duration: int = 15, start_time: int = 0) -> str:
    """Extract a snippet of audio using ffmpeg to a temporary file"""
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Create temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='speaker_id_')
    os.close(temp_fd)  # Close the file descriptor, we just need the path
    
    try:
        # Use ffmpeg to extract snippet
        cmd = [
            'ffmpeg', '-i', audio_path, 
            '-ss', str(start_time),
            '-t', str(duration),
            '-y',  # overwrite output files
            '-loglevel', 'error',  # Reduce ffmpeg output
            temp_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if not os.path.exists(temp_path):
            raise RuntimeError(f"Failed to create audio snippet at {temp_path}")
        
        return temp_path
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Error extracting audio snippet: {e}")
        if e.stderr:
            print(f"FFmpeg stderr: {e.stderr}")
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
    except Exception as e:
        print(f"❌ Unexpected error extracting snippet: {e}")
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise

def identify_speakers_interactively(audio_path: str, model_size: str = "base", duration: int = 60) -> list:
    """
    Extract a snippet, transcribe with speaker diarization, 
    and ask user to identify speakers interactively
    """
    
    print("🎙️  Speaker Identification Mode")
    print("=" * 50)
    print(f"Extracting first {duration} seconds for speaker identification...")
    
    # Extract snippet
    snippet_path = None
    try:
        snippet_path = extract_audio_snippet(audio_path, duration=duration)
        
        print("Transcribing snippet with speaker diarization...")
        
        # Verify snippet was created successfully
        if not snippet_path or not os.path.exists(snippet_path):
            print("❌ Failed to extract audio snippet.")
            return []
        
        # Use public API
        transcription_segments, info = transcribe_audio(
            snippet_path, 
            model_size=model_size,
            silent=False
        )
        
        # Do speaker diarization
        try:
            # Read HF token
            hf_token = None
            try:
                with open('.hf', 'r') as f:
                    hf_token = f.read().strip()
            except FileNotFoundError:
                print("Warning: .hf file not found.")
            
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", 
                use_auth_token=hf_token
            )
            
            diarization = diarization_pipeline(snippet_path)
            
            # Create speaker timeline
            speaker_timeline = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_timeline.append({
                    'start': turn.start,
                    'end': turn.end, 
                    'speaker': speaker
                })
            
            # Get unique speakers
            unique_speakers = sorted(list(set([s['speaker'] for s in speaker_timeline])))
            
            print(f"\n🔍 Found {len(unique_speakers)} speaker(s) in the snippet:")
            
            # Show transcription with speakers
            print("\nTranscription with detected speakers:")
            print("-" * 40)
            
            for segment in transcription_segments:
                start_time = segment.start
                end_time = segment.end
                text = segment.text.strip()
                timestamp = format_timestamp(start_time)
                
                # Find speaker for this segment
                speaker_label = None
                segment_center = (start_time + end_time) / 2
                
                for speaker_segment in speaker_timeline:
                    if (speaker_segment['start'] <= segment_center <= speaker_segment['end']):
                        speaker_label = speaker_segment['speaker']
                        break
                
                if speaker_label:
                    # Convert to readable format
                    try:
                        speaker_idx = unique_speakers.index(speaker_label)
                        readable_speaker = f"Speaker {speaker_idx + 1}"
                    except ValueError:
                        readable_speaker = "Unknown Speaker"
                    
                    line = f"[{timestamp} {readable_speaker}] {text}"
                else:
                    line = f"[{timestamp}] {text}"
                
                print(line)
            
            if not unique_speakers:
                print("❌ No speakers detected in snippet.")
                return []
            
            print("\n" + "=" * 50)
            print("Please identify each speaker:")
            
            speaker_mapping = {}
            for i, speaker in enumerate(unique_speakers):
                readable_speaker = f"Speaker {i + 1}"
                while True:
                    name = input(f"\nWho is '{readable_speaker}'? Enter name (or 'skip' to leave as-is): ").strip()
                    if name and name.lower() != 'skip':
                        speaker_mapping[speaker] = name
                        break
                    elif name.lower() == 'skip':
                        speaker_mapping[speaker] = readable_speaker
                        break
                    else:
                        print("Please enter a name or 'skip'")
            
            # Return list of names in order of speaker detection
            speaker_names = [speaker_mapping[speaker] for speaker in unique_speakers]
            
            print(f"\n✅ Speaker mapping confirmed:")
            for i, (original, mapped) in enumerate(speaker_mapping.items()):
                readable = f"Speaker {unique_speakers.index(original) + 1}"
                print(f"   {readable} → {mapped}")
            
            return speaker_names
            
        except Exception as e:
            print(f"❌ Speaker diarization failed: {e}")
            print("Showing basic transcription without speakers:")
            
            for segment in transcription_segments:
                timestamp = format_timestamp(segment.start)
                print(f"[{timestamp}] {segment.text.strip()}")
            
            return []
        
    finally:
        # Clean up temporary file
        if snippet_path and os.path.exists(snippet_path):
            os.unlink(snippet_path)


def save_transcription_data(segments, audio_path: str, model_size: str = "base"):
    """Save transcription segments to intermediate file"""
    audio_path_obj = Path(audio_path)
    output_path = audio_path_obj.with_suffix('.transcript-segments.txt')
    
    # Create compact segment format
    lines = []
    lines.append(f"# Audio: {audio_path}")
    lines.append(f"# Model: {model_size}")
    lines.append(f"# Segments: {len(segments)}")
    lines.append("")
    
    for segment in segments:
        start_time = format_timestamp(segment.start)
        end_time = format_timestamp(segment.end)
        text = segment.text.strip()
        lines.append(f"[{start_time}-{end_time}] {text}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"💾 Transcription segments saved to: {output_path}")
    return str(output_path)

def load_transcription_data(transcription_file: str):
    """Load transcription data from intermediate file"""
    if transcription_file.endswith('.transcript-segments.txt'):
        # Parse new text format
        segments = []
        audio_file = None
        
        with open(transcription_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('# Audio: '):
                    audio_file = line[9:]  # Remove '# Audio: '
                elif line.startswith('[') and '] ' in line:
                    # Parse: [0:00:04-0:00:10] Text content
                    bracket_end = line.find('] ')
                    if bracket_end > 0:
                        time_part = line[1:bracket_end]  # Remove [ and ]
                        text_part = line[bracket_end + 2:]  # Remove ] and space
                        
                        if '-' in time_part:
                            start_str, end_str = time_part.split('-', 1)
                            start_time = parse_timestamp(start_str)
                            end_time = parse_timestamp(end_str)
                            
                            segments.append({
                                'start': start_time,
                                'end': end_time,
                                'text': text_part
                            })
        
        return {
            'audio_file': audio_file or transcription_file,
            'segments': segments
        }
    else:
        # Legacy JSON format
        with open(transcription_file, 'r', encoding='utf-8') as f:
            return json.load(f)

def parse_timestamp(timestamp_str: str) -> float:
    """Convert HH:MM:SS timestamp string to seconds"""
    parts = timestamp_str.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    return 0.0

def apply_speaker_diarization(audio_file: str, transcription_data: dict, speaker_names: list = None, num_speakers: int = None):
    """Apply speaker diarization to existing transcription data"""
    
    overall_start = time.time()
    
    try:
        print("⚡ Optimizing speaker diarization for performance...")
        
        # Read Hugging Face token from .hf file
        hf_token = None
        try:
            with open('.hf', 'r') as f:
                hf_token = f.read().strip()
        except FileNotFoundError:
            print("Warning: .hf file not found. Speaker diarization may fail for gated models.")
        
        model_start = time.time()
        print("Loading diarization model with optimizations...")
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", 
            use_auth_token=hf_token
        )
        print(f"⏱️  Model loaded in {time.time() - model_start:.1f}s")
        
        # Optimize pipeline parameters for speed
        # These parameters help speed up processing
        import torch
        if hasattr(diarization_pipeline, '_segmentation'):
            # Set smaller batch size to reduce memory usage and potentially speed up processing
            if hasattr(diarization_pipeline._segmentation, 'batch_size'):
                diarization_pipeline._segmentation.batch_size = 16
        
        if hasattr(diarization_pipeline, '_embedding'):
            # Set smaller embedding batch size
            if hasattr(diarization_pipeline._embedding, 'batch_size'):
                diarization_pipeline._embedding.batch_size = 16
        
        audio_load_start = time.time()
        print("Loading audio into memory for faster processing...")
        # Load audio into memory first to avoid I/O bottlenecks
        import torchaudio
        try:
            waveform, sample_rate = torchaudio.load(audio_file)
            print(f"📊 Loaded audio in {time.time() - audio_load_start:.1f}s: {waveform.shape[1]/sample_rate:.1f}s, {sample_rate}Hz")
        except Exception as e:
            print(f"⚠️  Could not preload audio, using file path: {e}")
            waveform = audio_file  # Fallback to file path
        
        diarization_start = time.time()
        print("🔍 Performing speaker diarization...")
        
        # Set number of speakers if specified
        diarization_params = {}
        if num_speakers:
            diarization_params['num_speakers'] = num_speakers
            print(f"🎯 Constraining to {num_speakers} speakers for faster processing")
        elif speaker_names:
            diarization_params['num_speakers'] = len(speaker_names)
            print(f"🎯 Constraining to {len(speaker_names)} speakers based on provided names")
        
        # Use the optimized audio input - pyannote expects specific format
        if isinstance(waveform, torch.Tensor):
            # Convert to the format pyannote expects: {"waveform": tensor, "sample_rate": int}
            audio_input = {"waveform": waveform, "sample_rate": sample_rate}
        else:
            # Fallback to file path
            audio_input = audio_file
        
        diarization = diarization_pipeline(audio_input, **diarization_params)
        print(f"⏱️  Diarization completed in {time.time() - diarization_start:.1f}s")
        
        # Create speaker timeline
        speaker_timeline = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_timeline.append({
                'start': turn.start,
                'end': turn.end, 
                'speaker': speaker
            })
        
        unique_speakers = sorted(list(set([s['speaker'] for s in speaker_timeline])))
        print(f"✅ Found {len(unique_speakers)} speakers")
        
        # Process transcription segments with speaker information
        result_lines = []
        
        for segment_data in transcription_data['segments']:
            start_time = segment_data['start']
            end_time = segment_data['end']
            text = segment_data['text']
            timestamp = format_timestamp(start_time)
            
            # Find speaker for this segment
            speaker_label = None
            segment_center = (start_time + end_time) / 2
            
            for speaker_segment in speaker_timeline:
                if (speaker_segment['start'] <= segment_center <= speaker_segment['end']):
                    speaker_label = speaker_segment['speaker']
                    break
            
            if speaker_label:
                # Map speaker ID to name if names provided
                if speaker_names:
                    try:
                        speaker_idx = unique_speakers.index(speaker_label)
                        if speaker_idx < len(speaker_names):
                            speaker_name = speaker_names[speaker_idx]
                        else:
                            speaker_name = f"Speaker {speaker_idx + 1}"
                    except ValueError:
                        speaker_name = "Unknown Speaker"
                else:
                    try:
                        speaker_idx = unique_speakers.index(speaker_label)
                        speaker_name = f"Speaker {speaker_idx + 1}"
                    except ValueError:
                        speaker_name = "Unknown Speaker"
                
                line = f"[{timestamp} {speaker_name}] {text}"
            else:
                line = f"[{timestamp}] {text}"
            
            result_lines.append(line)
        
        total_time = time.time() - overall_start
        print(f"🏁 Speaker diarization completed in {total_time:.1f}s total")
        
        return result_lines
        
    except Exception as e:
        print(f"❌ Speaker diarization failed: {e}")
        print("Returning transcription without speaker labels...")
        
        # Fallback: return transcription without speakers
        result_lines = []
        for segment_data in transcription_data['segments']:
            timestamp = format_timestamp(segment_data['start'])
            line = f"[{timestamp}] {segment_data['text']}"
            result_lines.append(line)
        
        return result_lines

def cmd_transcribe(args):
    """Transcribe audio and save intermediate data"""
    print(f"🎤 Transcribing: {args.audio_file}")
    
    if not os.path.exists(args.audio_file):
        print(f"❌ Audio file not found: {args.audio_file}")
        return
    
    # Extract test snippet if requested
    audio_path = args.audio_file
    if hasattr(args, 'test') and args.test:
        duration = getattr(args, 'duration', 60)
        print(f"📝 Test mode: extracting first {duration} seconds...")
        audio_path = extract_audio_snippet(args.audio_file, duration=duration)
    
    try:
        # Use public API - cmd_transcribe is just a CLI wrapper
        num_speakers = getattr(args, 'num_speakers', None)
        speakers_list = args.speakers.split(',') if hasattr(args, 'speakers') and args.speakers else None
        
        segments_list, info = transcribe_audio(
            audio_path,
            model_size=args.model,
            num_speakers=num_speakers,
            speaker_names=speakers_list,
            silent=False
        )
        
        # Save transcription data
        # Use the actual audio path that was transcribed (snippet in test mode, original otherwise)
        transcription_file = save_transcription_data(segments_list, audio_path, args.model)
        
        # Show preview
        print("\n📋 Transcription preview:")
        print("-" * 40)
        for i, segment in enumerate(segments_list[:5]):  # Show first 5 segments
            timestamp = format_timestamp(segment.start)
            print(f"[{timestamp}] {segment.text.strip()}")
        
        if len(segments_list) > 5:
            print(f"... and {len(segments_list) - 5} more segments")
        
        return transcription_file, audio_path if hasattr(args, 'test') and args.test else None
        
    finally:
        # Don't clean up test snippet here - let the test workflow handle it
        # so diarization can use the same snippet
        pass

def cmd_diarize(args):
    """Apply speaker diarization to transcription data"""
    print(f"👥 Diarizing: {args.transcription_file}")
    
    if not os.path.exists(args.transcription_file):
        print(f"❌ Transcription file not found: {args.transcription_file}")
        return
    
    # Load transcription data
    transcription_data = load_transcription_data(args.transcription_file)
    audio_file = transcription_data['audio_file']
    
    if not os.path.exists(audio_file):
        print(f"❌ Original audio file not found: {audio_file}")
        return
    
    # Parse speaker names
    speaker_names = None
    if args.speakers:
        speaker_names = [name.strip() for name in args.speakers.split(',')]
    
    # Apply speaker diarization
    result = apply_speaker_diarization(
        audio_file=audio_file,
        transcription_data=transcription_data,
        speaker_names=speaker_names,
        num_speakers=args.num_speakers
    )
    
    # Save final output
    output_file = args.output
    if not output_file:
        audio_path = Path(audio_file)
        output_file = audio_path.with_suffix('.transcript.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))
    
    print(f"💾 Final transcription saved to: {output_file}")
    return output_file

def cmd_test(args):
    """Test transcription and diarization on variable-length snippet"""
    duration = getattr(args, 'duration', 60)
    print(f"🧪 Test Mode: Processing {duration}-second snippet")
    
    # Set test flag
    args.test = True
    
    if args.identify_speakers:
        # Interactive speaker identification
        try:
            speaker_names = identify_speakers_interactively(args.audio_file, args.model, duration)
            if speaker_names:
                print(f"\n✅ Identified speakers: {', '.join(speaker_names)}")
                
                # Now run a complete test transcription with the identified speakers
                print(f"\n🔄 Running full test transcription with identified speakers...")
                
                # Set speakers for the test run
                args.speakers = ','.join(speaker_names)
                args.num_speakers = len(speaker_names)
                
                # Run the full test workflow
                return run_test_workflow(args, speaker_names)
            else:
                print("\n❌ No speakers identified.")
                return []
        except Exception as e:
            print(f"❌ Speaker identification failed: {e}")
            return []
    else:
        # Regular test workflow
        return run_test_workflow(args, None)

def run_test_workflow(args, speaker_names):
    """Run the complete test workflow and show results"""
    
    # 1. Transcribe snippet
    duration = getattr(args, 'duration', 60)
    print(f"📝 Step 1: Transcribing {duration}-second test snippet...")
    result = cmd_transcribe(args)
    
    if not result:
        return
        
    # Handle new return format from cmd_transcribe
    if isinstance(result, tuple):
        transcription_file, test_snippet_path = result
    else:
        # Backward compatibility
        transcription_file, test_snippet_path = result, None
    
    # 2. Apply diarization if requested
    test_output_file = None
    if args.speakers or args.num_speakers:
        print(f"👥 Step 2: Applying speaker diarization...")
        
        args.transcription_file = transcription_file
        
        # Set test output file  
        audio_path = Path(args.audio_file)
        test_output_file = audio_path.with_suffix('.test-transcript.txt')
        args.output = str(test_output_file)
        
        # Run diarization and get results
        final_output_file = cmd_diarize(args)
        
        if final_output_file and os.path.exists(final_output_file):
            print(f"\n📋 Complete Test Transcript with Speaker Identification:")
            print("=" * 60)
            
            # Read and display the complete test transcript
            with open(final_output_file, 'r', encoding='utf-8') as f:
                test_content = f.read()
                print(test_content)
            
            print("=" * 60)
            print(f"📁 Test transcript saved to: {final_output_file}")
        else:
            print("⚠️  Could not generate test transcript file")
    else:
        print(f"📋 Basic test transcription (no speaker diarization):")
        if transcription_file and os.path.exists(transcription_file):
            transcription_data = load_transcription_data(transcription_file)
            for segment_data in transcription_data['segments'][:10]:  # Show first 10 segments
                timestamp = format_timestamp(segment_data['start'])
                print(f"[{timestamp}] {segment_data['text']}")
    
    # Show commands for full processing
    print(f"\n🚀 To process the complete audio file, run:")
    print(f"   poetry run python transcribe.py transcribe '{args.audio_file}'")
    
    if args.speakers or args.num_speakers:
        speakers_arg = f" --speakers {args.speakers}" if args.speakers else ""
        num_speakers_arg = f" --num-speakers {args.num_speakers}" if args.num_speakers else ""
        print(f"   poetry run python transcribe.py diarize <audio-file>.transcript-segments.txt{speakers_arg}{num_speakers_arg}")
        
        if speaker_names:
            print(f"\n💡 Or use the default command for one-step processing:")
            print(f"   poetry run python transcribe.py default '{args.audio_file}' --speakers {','.join(speaker_names)} --num-speakers {len(speaker_names)}")
    
    # Clean up temporary files
    if transcription_file and os.path.exists(transcription_file):
        try:
            os.unlink(transcription_file)
        except Exception as e:
            print(f"⚠️  Could not clean up transcription file: {e}")
    
    # Clean up test snippet if it was created
    if test_snippet_path and os.path.exists(test_snippet_path):
        try:
            os.unlink(test_snippet_path)
            print(f"🧹 Cleaned up test snippet")
        except Exception as e:
            print(f"⚠️  Could not clean up test snippet: {e}")
    
    return speaker_names if speaker_names else []

def cmd_default(args):
    """Default command: transcribe + diarize in one go"""
    print("🎯 Default Mode: Transcribe + Diarize")
    
    # Step 1: Transcribe
    result = cmd_transcribe(args)
    if isinstance(result, tuple):
        transcription_file, _ = result  # Ignore snippet path in default mode
    else:
        transcription_file = result
    
    if not transcription_file:
        return
    
    # Step 2: Diarize
    args.transcription_file = transcription_file
    
    # Set default speaker names if none provided
    if not args.speakers and not hasattr(args, 'no_diarize'):
        args.speakers = "Speaker 1,Speaker 2"
        print("ℹ️  No speakers specified, using default: Speaker 1, Speaker 2")
    
    if args.speakers or args.num_speakers:
        cmd_diarize(args)

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with speaker diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with snippet and interactive speaker identification (custom duration)
  python transcribe.py test audio.m4a --identify-speakers --duration 90
  
  # Full workflow with known speakers  
  python transcribe.py audio.m4a --speakers Cole,Roman
  
  # Step-by-step workflow
  python transcribe.py transcribe audio.m4a
  python transcribe.py diarize audio.transcript-segments.txt --speakers Cole,Roman --num-speakers 2
  
  # Test specific parameters with custom duration
  python transcribe.py test audio.m4a --speakers Cole,Roman --num-speakers 2 --duration 30
        """
    )
    
    # Global arguments
    parser.add_argument("--model", 
                       default="base",
                       choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                       help="Whisper model size (default: base)")
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test on audio snippet (configurable duration)')
    test_parser.add_argument('audio_file', help='Path to audio file')
    test_parser.add_argument('--duration', type=int, default=60, help='Test clip duration in seconds (default: 60)')
    test_parser.add_argument('--identify-speakers', action='store_true', help='Interactive speaker identification (slow)')
    test_parser.add_argument('--speakers', help='Comma-separated speaker names (e.g., Cole,Roman)')
    test_parser.add_argument('--num-speakers', type=int, help='Expected number of speakers')
    test_parser.add_argument('--no-diarize', action='store_true', help='Skip speaker diarization (faster)')
    test_parser.add_argument('--manual-speakers', action='store_true', help='Manual speaker assignment (fast)')
    test_parser.add_argument('-o', '--output', help='Output file path')
    
    # Transcribe command  
    transcribe_parser = subparsers.add_parser('transcribe', help='Transcribe audio to intermediate format')
    transcribe_parser.add_argument('audio_file', help='Path to audio file')
    
    # Diarize command
    diarize_parser = subparsers.add_parser('diarize', help='Apply speaker diarization to transcription')
    diarize_parser.add_argument('transcription_file', help='Path to .transcript-segments.txt file (or legacy .transcription.json)')
    diarize_parser.add_argument('--speakers', help='Comma-separated speaker names (e.g., Cole,Roman)')
    diarize_parser.add_argument('--num-speakers', type=int, help='Expected number of speakers')
    diarize_parser.add_argument('-o', '--output', help='Output file path')
    
    # Default command (no subcommand) - add as separate subparser
    default_parser = subparsers.add_parser('default', help='Full transcribe + diarize workflow (can omit "default")')
    default_parser.add_argument('audio_file', help='Path to audio file')
    default_parser.add_argument('--speakers', help='Comma-separated speaker names (e.g., Cole,Roman)')
    default_parser.add_argument('--num-speakers', type=int, help='Expected number of speakers')  
    default_parser.add_argument('-o', '--output', help='Output file path')
    default_parser.add_argument('--no-diarize', action='store_true', help='Skip diarization (transcribe only)')
    
    # Make default subcommand if no subcommand provided but audio file is given
    # This is a bit hacky but handles the common case
    if len(sys.argv) > 1 and not sys.argv[1] in ['test', 'transcribe', 'diarize', 'default', '-h', '--help']:
        # Assume first argument is audio file, insert 'default' command
        sys.argv.insert(1, 'default')
    
    args = parser.parse_args()
    
    # Route to appropriate command
    if args.command == 'test':
        cmd_test(args)
    elif args.command == 'transcribe':
        cmd_transcribe(args)
    elif args.command == 'diarize':
        cmd_diarize(args)
    elif args.command == 'default':
        cmd_default(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
