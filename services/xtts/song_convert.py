from __future__ import annotations

import argparse
import gc
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from TTS.api import TTS

MODEL_NAME = "voice_conversion_models/multilingual/vctk/freevc24"
CHUNK_SECONDS = 15.0
CHUNK_OVERLAP_SECONDS = 0.5
MIN_FINAL_CHUNK_SECONDS = 2.0


def available_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return requested if requested in {"cpu", "cuda", "mps"} else "cpu"


def run_ffmpeg(arguments: list[str], failure: str) -> None:
    completed = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *arguments],
        capture_output=True,
        check=False,
        timeout=1800,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{failure}: {detail[-1000:]}")


def audio_duration(source: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source),
        ],
        capture_output=True,
        check=False,
        timeout=60,
    )
    try:
        duration = float(completed.stdout.decode("utf-8").strip())
    except ValueError as error:
        raise RuntimeError("Unable to read vocal duration.") from error
    if completed.returncode != 0 or duration <= 0:
        raise RuntimeError("Unable to read vocal duration.")
    return duration


def chunk_ranges(duration: float) -> list[tuple[float, float]]:
    chunks: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(start + CHUNK_SECONDS, duration)
        if duration - (end - CHUNK_OVERLAP_SECONDS) < MIN_FINAL_CHUNK_SECONDS:
            end = duration
        chunks.append((start, end - start))
        if end >= duration:
            break
        start = end - CHUNK_OVERLAP_SECONDS
    return chunks


def extract_chunk(source: Path, destination: Path, start: float, duration: float) -> None:
    run_ffmpeg(
        [
            "-ss",
            f"{start:.6f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.6f}",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        "Unable to prepare a vocal chunk",
    )


def join_chunks(chunks: list[Path], output: Path, duration: float) -> None:
    if len(chunks) == 1:
        shutil.move(chunks[0], output)
        return

    arguments: list[str] = []
    for chunk in chunks:
        arguments.extend(["-i", str(chunk)])

    filters: list[str] = []
    previous = "[0:a]"
    for index in range(1, len(chunks)):
        next_label = f"[joined{index}]"
        filters.append(
            f"{previous}[{index}:a]acrossfade="
            f"d={CHUNK_OVERLAP_SECONDS}:c1=tri:c2=tri{next_label}"
        )
        previous = next_label
    filters.append(f"{previous}apad,atrim=0:{duration:.6f}[out]")
    arguments.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )
    run_ffmpeg(arguments, "Unable to join converted vocal chunks")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()

    source = Path(arguments.source).resolve(strict=True)
    target = Path(arguments.target).resolve(strict=True)
    output = Path(arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    device = available_device(arguments.device)
    converter = TTS(model_name=MODEL_NAME, progress_bar=False).to(device)
    duration = audio_duration(source)

    with tempfile.TemporaryDirectory(prefix="freevc-chunks-") as temporary_dir:
        temporary_path = Path(temporary_dir)
        converted_chunks: list[Path] = []
        ranges = chunk_ranges(duration)
        for index, (start, chunk_duration) in enumerate(ranges):
            print(
                f"Converting vocal chunk {index + 1}/{len(ranges)}",
                file=sys.stderr,
                flush=True,
            )
            source_chunk = temporary_path / f"source-{index:03d}.wav"
            converted_chunk = temporary_path / f"converted-{index:03d}.wav"
            extract_chunk(source, source_chunk, start, chunk_duration)
            converter.voice_conversion_to_file(
                source_wav=str(source_chunk),
                target_wav=str(target),
                file_path=str(converted_chunk),
            )
            if not converted_chunk.is_file() or converted_chunk.stat().st_size == 0:
                raise RuntimeError("Voice conversion produced an empty chunk.")
            converted_chunks.append(converted_chunk)
            source_chunk.unlink(missing_ok=True)
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        join_chunks(converted_chunks, output, duration)


if __name__ == "__main__":
    main()
