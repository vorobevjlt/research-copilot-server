from __future__ import annotations

import argparse
from pathlib import Path

import torch
from TTS.api import TTS

MODEL_NAME = "voice_conversion_models/multilingual/vctk/freevc24"


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
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()

    source = Path(arguments.source).resolve(strict=True)
    target = Path(arguments.target).resolve(strict=True)
    output = Path(arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    converter = TTS(model_name=MODEL_NAME, progress_bar=False).to(
        available_device(arguments.device)
    )
    converter.voice_conversion_to_file(
        source_wav=str(source),
        target_wav=str(target),
        file_path=str(output),
    )


if __name__ == "__main__":
    main()
