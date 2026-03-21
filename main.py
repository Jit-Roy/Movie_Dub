import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import librosa

from Vocal_Music_Separation import vocal_music_separator
from Speech_Overlap import detect_overlaps
from Speaker_Diarization import perform_diarization_and_extract
from Speaker_Separation import separate_speakers
from Speaker_Identification import match_and_merge_speaker
from ASR import transcribe_audio, WHISPER_ID
from Qwen3llm import translate_fragment
from Qwen3tts import load_tts_model, generate_voice_clone
from helper import ensure_wav, load_mono, save_wav

DEFAULT_TEMP_DIR = "temp"
DEFAULT_SEPARATOR_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEFAULT_SEPARATION_MODEL = "lichenda/wsj0_2mix_skim_noncausal"
DEFAULT_MATCH_THRESHOLD = 0.60
DEFAULT_SEPARATION_DEVICE = "cuda"
DEFAULT_SR = 16000


def time_stretch_to_duration(audio: np.ndarray, sr: int, target_duration: float) -> np.ndarray:
    if len(audio) == 0 or target_duration <= 0:
        return audio
    current_duration = len(audio) / sr
    if current_duration <= 0:
        return audio
    rate = current_duration / target_duration
    if rate <= 0:
        return audio
    return librosa.effects.time_stretch(audio, rate=rate)


def overlay_audio(base: np.ndarray, overlay: np.ndarray, start_sample: int) -> np.ndarray:
    if overlay.size == 0:
        return base
    end_sample = start_sample + len(overlay)
    if end_sample > len(base):
        base = np.pad(base, (0, end_sample - len(base)))
    base[start_sample:end_sample] += overlay
    return base


def find_speaker_audio(diarization_dir: Path) -> List[Tuple[str, Path]]:
    speakers = []
    for spk_dir in sorted(diarization_dir.glob("speaker_*")):
        if not spk_dir.is_dir():
            continue
        merged = spk_dir / f"{spk_dir.name}_combined_merged.wav"
        combined = spk_dir / f"{spk_dir.name}_combined.wav"
        if merged.exists():
            speakers.append((spk_dir.name, merged))
        elif combined.exists():
            speakers.append((spk_dir.name, combined))
    return speakers


def collect_separation_outputs(separation_dir: Path) -> Dict[int, List[Path]]:
    mapping: Dict[int, List[Path]] = {}
    for segment_dir in separation_dir.glob("segment*"):
        if not segment_dir.is_dir():
            continue
        name = segment_dir.name
        if not name.startswith("segment"):
            continue
        try:
            idx = int(name.replace("segment", ""))
        except ValueError:
            continue
        voices = sorted(segment_dir.glob("voice*.wav"))
        if voices:
            mapping[idx] = voices
    return mapping


def synthesize_speaker_segments(
    speaker_audio_path: Path,
    segments: List[Dict],
    target_language: str,
    ref_audio: str | None,
    ref_text: str | None,
    tts_model,
) -> Tuple[np.ndarray, List[Dict]]:
    base_audio, sr = load_mono(speaker_audio_path, sr=DEFAULT_SR)
    dubbed = np.zeros_like(base_audio)
    segment_outputs = []

    if not ref_audio:
        ref_audio = str(speaker_audio_path)
    if not ref_text:
        ref_text = " ".join(seg.get("text", "").strip() for seg in segments).strip()

    for seg_idx, seg in enumerate(segments, start=1):
        text = seg.get("text", "").strip()
        if not text:
            continue

        translated = translate_fragment(text, target_language)
        if not ref_text:
            continue

        wavs, tts_sr = generate_voice_clone(
            text=translated,
            language=target_language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            model=tts_model,
        )
        if not wavs:
            continue

        tts_audio = wavs[0]
        if isinstance(tts_audio, np.ndarray) and tts_audio.ndim > 1:
            tts_audio = tts_audio.mean(axis=1)

        if tts_sr != DEFAULT_SR:
            tts_audio = librosa.resample(tts_audio.astype(np.float32), orig_sr=tts_sr, target_sr=DEFAULT_SR)

        target_duration = float(seg["end"] - seg["start"])
        tts_audio = time_stretch_to_duration(tts_audio.astype(np.float32), DEFAULT_SR, target_duration)

        start_sample = int(float(seg["start"]) * DEFAULT_SR)
        dubbed = overlay_audio(dubbed, tts_audio, start_sample)

        segment_outputs.append(
            {
                "segment_index": seg_idx,
                "start": seg["start"],
                "end": seg["end"],
                "source_text": text,
                "translated_text": translated,
            }
        )

    return dubbed, segment_outputs


def mix_audio_tracks(tracks: List[np.ndarray]) -> np.ndarray:
    if not tracks:
        return np.array([], dtype=np.float32)

    max_len = max(len(t) for t in tracks)
    mixed = np.zeros(max_len, dtype=np.float32)
    for t in tracks:
        if len(t) < max_len:
            t = np.pad(t, (0, max_len - len(t)))
        mixed += t

    peak = np.max(np.abs(mixed)) if mixed.size else 0.0
    if peak > 1.0:
        mixed = mixed / peak
    return mixed


def main() -> None:
    parser = argparse.ArgumentParser(description="Movie dubbing pipeline")
    parser.add_argument("--input-audio", required=True, help="Path to input mp3 or wav")
    parser.add_argument("--target-language", default="Chinese", help="Target translation and TTS language")
    parser.add_argument("--hf-token", default="", help="Hugging Face token for overlap detection")
    args = parser.parse_args()

    input_path = Path(args.input_audio).resolve()
    temp_dir = Path(DEFAULT_TEMP_DIR).resolve()
    output_dir = Path.cwd().resolve()

    temp_dir.mkdir(parents=True, exist_ok=True)

    wav_path = ensure_wav(input_path, temp_dir)

    print("[1/7] Separating vocals and music...")
    vocal_music_separator(
        str(wav_path),
        vocal_dir=str(temp_dir / "vocal"),
        music_dir=str(temp_dir / "music"),
        model_name=DEFAULT_SEPARATOR_MODEL,
    )

    vocal_path = temp_dir / "vocal" / "vocal.wav"
    music_path = temp_dir / "music" / "music.wav"
    if not vocal_path.exists():
        raise FileNotFoundError(f"Vocal output not found: {vocal_path}")

    print("[2/7] Detecting overlapping speech...")
    if args.hf_token:
        overlaps = detect_overlaps(
            hf_token=args.hf_token,
            audio_file=str(vocal_path),
            plot=False,
        )
    else:
        overlaps = []
        print("No HF token provided. Skipping overlap detection.")

    print("[3/7] Running diarization...")
    diarization_dir = temp_dir / "diarization"
    perform_diarization_and_extract(
        audio_path=str(vocal_path),
        output_base_dir=str(diarization_dir),
        remove_segments=overlaps if overlaps else None,
    )

    if overlaps:
        print("[4/7] Separating overlapping speakers...")
        separation_dir = temp_dir / "separation"
        separate_speakers(
            audio_file=str(vocal_path),
            output_dir=str(separation_dir),
            segments=overlaps,
            model=DEFAULT_SEPARATION_MODEL,
            device=DEFAULT_SEPARATION_DEVICE,
        )

        print("[5/7] Matching separated voices to diarization speakers...")
        separation_mapping = collect_separation_outputs(separation_dir)
        for seg_idx, voices in separation_mapping.items():
            if seg_idx - 1 >= len(overlaps):
                continue
            segment = overlaps[seg_idx - 1]
            for voice_path in voices:
                result = match_and_merge_speaker(
                    input_audio_path=str(voice_path),
                    segment=segment,
                    diarization_dir=str(diarization_dir),
                    threshold=DEFAULT_MATCH_THRESHOLD,
                )
                print(result)
    else:
        print("[4/7] Skipping separation/identification (no overlap segments).")

    print("[6/7] Running ASR per speaker...")
    speaker_files = find_speaker_audio(diarization_dir)
    if not speaker_files:
        raise RuntimeError("No speaker audio files found after diarization.")

    tts_model = load_tts_model()
    dubbed_tracks = []
    for speaker_name, speaker_audio_path in speaker_files:
        print(f"  ASR -> {speaker_name}")
        result = transcribe_audio(
            str(speaker_audio_path),
            whisper_model_id=WHISPER_ID,
            device=None,
        )

        segments = result.get("segments", [])
        if not segments:
            continue

        dubbed_audio, _segment_outputs = synthesize_speaker_segments(
            speaker_audio_path=speaker_audio_path,
            segments=segments,
            target_language=args.target_language,
            ref_audio=None,
            ref_text=None,
            tts_model=tts_model,
        )
        dubbed_tracks.append(dubbed_audio)

    print("[7/7] Mixing music and dubbed speakers...")
    tracks = []
    if music_path.exists():
        music_audio, _ = load_mono(music_path, sr=DEFAULT_SR)
        tracks.append(music_audio)

    tracks.extend(dubbed_tracks)
    final_mix = mix_audio_tracks(tracks)
    final_path = output_dir / "final_mix.wav"
    save_wav(final_path, final_mix, DEFAULT_SR)

    print(f"Done. Final mix: {final_path}")


if __name__ == "__main__":
    main()
