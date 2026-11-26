"""Rule 3: Place-specific nasal onset ('m' vs 'n') and coda ('-n', '-ng', '-ong/-iong') anti-resonance shaping."""
from __future__ import annotations

import numpy as np
from scipy.fft import rfftfreq

from exp_20_pybot_larynx import N_BAP, N_BINS, N_FFT, SR
from exp_20_pybot_larynx.arbiter import (
    NEPER_TO_DB,
    RuleProposal,
    UtteranceContext,
    classify_onset_manner,
    has_nasal_coda,
)
from exp_20_pybot_larynx.larynx.synth import LarynxControl

_FREQS = rfftfreq(N_FFT, 1.0 / SR)
_DB_TO_NEPER = 1.0 / NEPER_TO_DB
_NASAL_SCALE = 0.80


def _gauss(center_hz: float, sigma_hz: float) -> np.ndarray:
    return np.exp(-0.5 * ((_FREQS - center_hz) / sigma_hz) ** 2)


# Empirically calibrated from TRAIN_IDS (real - Stage2_C2) residuals:
# Bilabial nasal onset 'm': oral cavity side-branch antiresonance notch ~1.35 kHz + HF rolloff
_ONSET_M_TEMPLATE = _DB_TO_NEPER * (
    -1.00
    + 0.45 * _gauss(800.0, 250.0)
    - 1.55 * _gauss(1350.0, 320.0)
    - 0.40 * _gauss(3000.0, 700.0)
)
# Alveolar nasal onset 'n': stronger 0.7-1.5 kHz dip while preserving ~2.0 kHz alveolar transition
_ONSET_N_TEMPLATE = _DB_TO_NEPER * (
    -0.95
    - 0.45 * _gauss(850.0, 420.0)
    + 0.90 * _gauss(2000.0, 450.0)
)
# Alveolar coda '-n': low murmur ~220 Hz preserved, antiresonance zero ~2.8 kHz
_CODA_N_TEMPLATE = _DB_TO_NEPER * (
    -1.05
    + 0.55 * _gauss(220.0, 180.0)
    - 0.40 * _gauss(2800.0, 850.0)
)
# Velar coda '-ng': low/mid murmur preserved, velar antiresonance dip ~1.65 kHz
_CODA_NG_TEMPLATE = _DB_TO_NEPER * (
    -0.85
    + 0.45 * _gauss(250.0, 220.0)
    - 0.40 * _gauss(1650.0, 650.0)
)
# Rounded velar coda '-ong/-iong': low murmur ~200 Hz preserved, deeper ~1.2 kHz + ~3.9 kHz zeros
_CODA_ONG_TEMPLATE = _DB_TO_NEPER * (
    -1.10
    + 0.85 * _gauss(200.0, 170.0)
    - 0.70 * _gauss(1200.0, 400.0)
    - 0.50 * _gauss(3900.0, 750.0)
)


def classify_nasal_coda(final: str) -> str:
    """Classify Mandarin final into 'coda_ong', 'coda_ng', 'coda_n', or 'none'."""
    f = str(final).strip()
    if not has_nasal_coda(f):
        return "none"
    if f in ("ong", "iong"):
        return "coda_ong"
    if f.endswith("ng"):
        return "coda_ng"
    return "coda_n"


class RuleNasalAntiresonance:
    """Applies separate m/n onset and -n/-ng/-ong coda anti-resonance and nasal clarity shaping."""

    name = "nasal_antiresonance"

    def propose(self, ctx: UtteranceContext, base: LarynxControl) -> RuleProposal | None:
        n_fr = len(base.f0_hz)
        if n_fr == 0:
            return None

        mask = np.zeros(n_fr, dtype=bool)
        d_env = np.zeros((n_fr, N_BINS), dtype=np.float64)
        d_bap = np.zeros((n_fr, N_BAP), dtype=np.float64)

        for s in ctx.syllables:
            a = max(0, min(n_fr, int(s.start_fr)))
            b = max(a, min(n_fr, int(s.end_fr)))
            sp = max(a, min(b, int(s.split_fr)))
            if b <= a:
                continue

            if classify_onset_manner(s.onset) == "nasal" and sp > a:
                mask[a:sp] = True
                tmpl = _ONSET_M_TEMPLATE if s.onset == "m" else _ONSET_N_TEMPLATE
                d_env[a:sp] += _NASAL_SCALE * tmpl[None, :]
                v_on = base.voiced[a:sp]
                if np.any(v_on):
                    d_bap[a:sp, 1:4][v_on] -= 0.008

            coda_kind = classify_nasal_coda(s.final)
            if coda_kind != "none":
                f_a = sp if (s.onset and sp > a) else a
                if b > f_a:
                    u_f = np.asarray(ctx.phase_u[f_a:b], dtype=np.float64)
                    if coda_kind == "coda_ong":
                        coda_w = np.clip((u_f - 0.82) / 0.16, 0.0, 1.0)
                        tmpl = _CODA_ONG_TEMPLATE
                    elif coda_kind == "coda_ng":
                        coda_w = np.clip((u_f - 0.84) / 0.14, 0.0, 1.0)
                        tmpl = _CODA_NG_TEMPLATE
                    else:
                        coda_w = np.clip((u_f - 0.84) / 0.14, 0.0, 1.0)
                        tmpl = _CODA_N_TEMPLATE

                    coda_m = coda_w > 0.0
                    if np.any(coda_m):
                        idx_c = np.arange(f_a, b)[coda_m]
                        mask[idx_c] = True
                        d_env[idx_c] += (_NASAL_SCALE * coda_w[coda_m, None]) * tmpl[None, :]

        if not np.any(mask):
            return None

        return RuleProposal(
            module_name=self.name,
            mask=mask,
            delta_log_env=d_env,
            delta_bap=d_bap,
        )
