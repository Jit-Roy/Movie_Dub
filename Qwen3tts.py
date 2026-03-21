import torch
from qwen_tts import Qwen3TTSModel


def load_tts_model(device: str | None = None, dtype: torch.dtype | None = None) -> Qwen3TTSModel:
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device_map=device,
        dtype=dtype,
    )


def generate_voice_clone(
    text: str,
    language: str,
    ref_audio: str,
    ref_text: str,
    model: Qwen3TTSModel | None = None,
):
    if model is None:
        model = load_tts_model()
    return model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text,
    )


if __name__ == "__main__":
    ref_audio = "D:\\Personal Projects\\Movie_Dub\\Sample_English_Audio.mp3"
    ref_text = "This week we got a open weak model which is really really impressive."
    model = load_tts_model()
    wavs, sr = generate_voice_clone(
        text=(
            "Ciao, bellissima! Hai quel look che mi fa girare la testa-"
            "sempre impeccabile. Stasera c'e un nuovo lounge in centro... "
            "ci vai con me? Prometto: niente di noioso, solo stile e buona musica."
        ),
        language="Italian",
        ref_audio=ref_audio,
        ref_text=ref_text,
        model=model,
    )
    print(f"Generated {len(wavs)} clip(s) at {sr} Hz")