"""Decode published MP3 donors to the 24 kHz mono PCM the bank expects.

The bank ships PCM WAV so the runtime needs no decoder. Extending the bank is a
build-time step, and this module is that step. It shells out to a user-supplied
ffmpeg binary and verifies the result; it never guesses at audio it could not
decode, and it refuses to write a file whose hash does not match the pinned
manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SR = 24000

# Candidates are tried only when the caller did not pass an explicit path. They
# are conveniences for this machine, not a requirement of the format.
FFMPEG_CANDIDATES = (
    'ffmpeg',
    r'C:\Users\MECHREVO\Documents\deepseek-harness\default-workspace\.tools\ffmpeg'
    r'\ffmpeg-9.0.2-essentials_build\bin\ffmpeg.exe',
    r'C:\Users\MECHREVO\Documents\deepseek-harness\default-workspace\tools\ffmpeg.exe',
)


def find_ffmpeg(explicit=None):
    """Return a usable ffmpeg path, or raise with an actionable message."""
    tried = []
    for candidate in ((explicit,) if explicit else ()) + FFMPEG_CANDIDATES:
        found = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
        if not found:
            tried.append(f'{candidate} (missing)')
            continue
        try:
            probe = subprocess.run([found, '-version'], capture_output=True,
                                   text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            tried.append(f'{found} ({type(exc).__name__})')
            continue
        if probe.returncode == 0 and probe.stdout.startswith('ffmpeg version'):
            return found
        tried.append(f'{found} (not ffmpeg)')
    raise RuntimeError('no usable ffmpeg found; pass --ffmpeg or extend the bank '
                       'with pre-decoded WAV. Tried: ' + '; '.join(tried))


def decode(source, target, ffmpeg, sr=SR):
    """Decode one audio file to mono 16-bit PCM WAV at `sr`; return target."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
               '-i', str(source), '-ac', '1', '-ar', str(sr),
               '-sample_fmt', 's16', '-f', 'wav', str(target)]
    run = subprocess.run(command, capture_output=True, text=True)
    if run.returncode != 0:
        target.unlink(missing_ok=True)
        detail = (run.stderr or run.stdout or '').strip().splitlines()[-3:]
        raise ValueError(f'ffmpeg failed on {Path(source).name}: ' + ' | '.join(detail))
    if not target.is_file() or target.stat().st_size <= 44:
        raise ValueError(f'decode produced no audio for {Path(source).name}')
    return target


def decode_all(manifest_path, out_dir, ffmpeg=None, sr=SR):
    """Decode every record in a manifest; return a report of what happened.

    The manifest is the JSON written by the selection step: it maps recording id
    to the source MP3 hash so a re-decode can be proven to start from the same
    bytes the licence notice describes.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    tool = find_ffmpeg(ffmpeg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(manifest_path).resolve().parent / 'mp3'

    rows, failures = [], []
    for ident, record in sorted(manifest.items()):
        source = source_dir / f'{ident}.mp3'
        if not source.is_file():
            failures.append(dict(id=ident, reason='source mp3 missing'))
            continue
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != record.get('sha256'):
            failures.append(dict(id=ident, reason='source hash mismatch',
                                 expected=record.get('sha256'), actual=digest))
            continue
        target = out_dir / f'{ident}.wav'
        try:
            decode(source, target, tool, sr=sr)
        except ValueError as exc:
            failures.append(dict(id=ident, reason=str(exc)))
            continue
        rows.append(dict(id=ident, wav=str(target.relative_to(HERE)) if
                         target.is_relative_to(HERE) else str(target),
                         wav_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                         mp3_sha256=digest, text=record.get('text'),
                         pinyin=record.get('pinyin')))
    return dict(sr=sr, ffmpeg=tool, decoded=rows, failures=failures)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('-o', '--out-dir', type=Path, required=True)
    parser.add_argument('--ffmpeg')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = decode_all(args.manifest, args.out_dir, args.ffmpeg)
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.report:
        args.report.write_text(text, encoding='utf-8')
    print(f"decoded {len(result['decoded'])}, failed {len(result['failures'])}")
    for row in result['failures']:
        print('  FAIL', row['id'], row['reason'])
    if result['failures']:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
