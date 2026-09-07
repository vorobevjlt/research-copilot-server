from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from TTS.api import TTS


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()

    text = sys.stdin.read().strip()
    if not text:
        raise ValueError("Speech text is empty.")

    source = Path(arguments.source).resolve(strict=True)
    output = Path(arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    model = TTS(model_name=arguments.model, progress_bar=False).to(
        available_device(arguments.device)
    )
    model.tts_to_file(
        text=text,
        file_path=str(output),
        speaker_wav=[str(source)],
        language=arguments.language,
        split_sentences=True,
    )


if __name__ == "__main__":
    main()
