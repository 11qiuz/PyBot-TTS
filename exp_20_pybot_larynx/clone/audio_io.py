"""Audio decoding, resampling to 24 kHz mono, high-pass filtering, and silence trimming."""
from __future__ import annotations

from functools import lru_cache
import io
import math
from pathlib import Path
from typing import Any
import wave

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfiltfilt

from exp_20_pybot_larynx import SR


@lru_cache(maxsize=4)
def _hp45_sos(sr: int = SR) -> np.ndarray:
    return butter(2, 45.0, btype="high", fs=float(sr), output="sos")


def _read_wav_bytes_raw(data: bytes) -> tuple[np.ndarray, int, int]:
    """Parse PCM/IEEE-float WAV bytes into float64 in [-1, 1], returning `(mono_x, sr, n_channels)`."""
    if not data or len(data) < 44:
        raise ValueError("音频文件为空或不是有效的 WAV 格式。")

    raw_arr: np.ndarray | None = None
    sr: int = 0
    try:
        sr_read, arr = wavfile.read(io.BytesIO(data))
        sr = int(sr_read)
        if arr.dtype == np.uint8:
            raw_arr = (arr.astype(np.float64) - 128.0) / 128.0
        elif arr.dtype == np.int16:
            raw_arr = arr.astype(np.float64) / 32768.0
        elif arr.dtype == np.int32:
            raw_arr = arr.astype(np.float64) / 2147483648.0
        elif np.issubdtype(arr.dtype, np.floating):
            raw_arr = arr.astype(np.float64)
        else:
            raw_arr = None
    except Exception:
        raw_arr = None

    if raw_arr is None:
        try:
            with wave.open(io.BytesIO(data), "rb") as wf:
                sr = int(wf.getframerate())
                nch = int(wf.getnchannels())
                sw = int(wf.getsampwidth())
                frames = wf.readframes(wf.getnframes())
        except Exception as exc:
            raise ValueError(f"无法解析参考音频 WAV 数据: {exc}") from exc

        if sw == 1:
            arr_u8 = np.frombuffer(frames, dtype=np.uint8)
            raw_arr = (arr_u8.astype(np.float64) - 128.0) / 128.0
        elif sw == 2:
            raw_arr = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
        elif sw == 3:
            b = np.frombuffer(frames, dtype=np.uint8)
            n_samp = len(b) // 3
            if n_samp == 0:
                raise ValueError("WAV 音频内容为空。")
            tri = b[: n_samp * 3].reshape(-1, 3)
            vals = (
                tri[:, 0].astype(np.int32)
                | (tri[:, 1].astype(np.int32) << 8)
                | (tri[:, 2].astype(np.int32) << 16)
            )
            vals = np.where(vals >= 0x800000, vals - 0x1000000, vals)
            raw_arr = vals.astype(np.float64) / 8388608.0
        elif sw == 4:
            raw_arr = np.frombuffer(frames, dtype="<i4").astype(np.float64) / 2147483648.0
        else:
            raise ValueError(f"不支持的 WAV 采样位宽: {sw * 8}-bit")

        if nch > 1:
            raw_arr = raw_arr.reshape(-1, nch)

    if raw_arr.ndim == 1:
        n_ch = 1
        mono = raw_arr
    else:
        n_ch = int(raw_arr.shape[1])
        mono = np.mean(raw_arr, axis=1)

    return np.asarray(mono, dtype=np.float64).ravel(), sr, n_ch


def _trim_silence(x: np.ndarray, sr: int = SR, margin_ms: float = 80.0, thresh_db: float = -34.0) -> np.ndarray:
    """Trim leading and trailing silence below `thresh_db` relative to p95 frame RMS."""
    win = max(120, int(round(0.020 * sr)))
    hop = max(60, int(round(0.005 * sr)))
    if len(x) <= win + hop:
        return x

    n_fr = 1 + (len(x) - win) // hop
    idx = np.arange(n_fr)[:, None] * hop + np.arange(win)[None, :]
    frames = x[idx]
    fr_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-18)
    ref_rms = float(np.percentile(fr_rms, 95))
    if ref_rms <= 1e-6:
        return x[:0]

    thr = max(1e-4, ref_rms * (10.0 ** (thresh_db / 20.0)))
    active = np.where(fr_rms >= thr)[0]
    if active.size == 0:
        return x[:0]

    margin_samp = int(round(margin_ms * sr / 1000.0))
    s0 = max(0, int(active[0]) * hop - margin_samp)
    s1 = min(len(x), int(active[-1]) * hop + win + margin_samp)
    return x[s0:s1]


def decode_wav_bytes_to_24k_mono(
    data: bytes,
    *,
    max_duration_s: float = 30.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode WAV bytes to 24 kHz mono float64 waveform with rumble removal & silence trimming."""
    x, orig_sr, n_ch = _read_wav_bytes_raw(data)
    if orig_sr < 8000 or orig_sr > 96000:
        raise ValueError(f"不支持的采样率 {orig_sr} Hz，请提供 8–96 kHz 音频。")
    if len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("参考音频为空或包含无效采样点。")

    max_orig = int(round(max_duration_s * orig_sr))
    if len(x) > max_orig:
        x = x[:max_orig]

    was_resampled = orig_sr != SR
    if was_resampled:
        g = math.gcd(SR, orig_sr)
        x = resample_poly(x, SR // g, orig_sr // g).astype(np.float64)

    if len(x) > 32:
        x = sosfiltfilt(_hp45_sos(SR), x)

    x_trim = _trim_silence(x, sr=SR, margin_ms=80.0, thresh_db=-34.0)
    dur_s = len(x_trim) / float(SR)
    rms = float(np.sqrt(np.mean(x_trim * x_trim))) if x_trim.size else 0.0
    if dur_s < 0.6 or rms < 1e-4:
        raise ValueError("参考音频有效语音过短（需 >= 0.6 秒）或音量过小。")

    peak = float(np.max(np.abs(x_trim)))
    if peak > 0.99:
        x_trim = x_trim * (0.98 / peak)

    return np.ascontiguousarray(x_trim, dtype=np.float64), {
        "orig_sr": int(orig_sr),
        "orig_channels": int(n_ch),
        "duration_s": round(dur_s, 3),
        "was_resampled": bool(was_resampled),
    }


def load_wav_file_to_24k_mono(
    path: Path | str,
    *,
    max_duration_s: float = 30.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a WAV file from disk and decode it to 24 kHz mono."""
    return decode_wav_bytes_to_24k_mono(Path(path).read_bytes(), max_duration_s=max_duration_s)
