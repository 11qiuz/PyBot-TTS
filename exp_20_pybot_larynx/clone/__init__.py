"""Zero-ML reference sample voice cloning package for PyBot-Larynx Stage 5."""
from exp_20_pybot_larynx.clone.adapter import apply_clone_to_control, estimate_clone_vtln
from exp_20_pybot_larynx.clone.audio_io import decode_wav_bytes_to_24k_mono, load_wav_file_to_24k_mono
from exp_20_pybot_larynx.clone.profile import (
    BASE_SPEAKER_PROFILE_PATH,
    VoiceCloneProfile,
    build_base_speaker_profile,
    extract_clone_profile,
    extract_clone_profile_from_wavs,
    load_base_speaker_profile,
)

__all__ = [
    "BASE_SPEAKER_PROFILE_PATH",
    "VoiceCloneProfile",
    "apply_clone_to_control",
    "build_base_speaker_profile",
    "decode_wav_bytes_to_24k_mono",
    "estimate_clone_vtln",
    "extract_clone_profile",
    "extract_clone_profile_from_wavs",
    "load_base_speaker_profile",
    "load_wav_file_to_24k_mono",
]
