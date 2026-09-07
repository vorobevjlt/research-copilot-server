from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import re
import subprocess
import tempfile
import threading
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from starlette.background import BackgroundTask

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_TEXT_LENGTH = 4096
MIN_SAMPLE_SECONDS = 3.0
MAX_SAMPLE_SECONDS = 120.0
UPLOAD_CHUNK_BYTES = 1024 * 1024
VOICE_ID_PATTERN = re.compile(r"^voice_[0-9a-f]{32}$")
SUPPORTED_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".ogg",
    ".wav",
    ".webm",
}
SUPPORTED_LANGUAGES = {
    "de",
    "en",
    "es",
    "fr",
    "it",
    "ja",
    "ko",
    "nl",
    "pl",
    "pt",
    "ru",
    "zh",
}

VOICE_DATA_DIR = Path(os.getenv("VOICE_DATA_DIR", "/data/voices")).resolve()
MODEL_NAME = os.getenv(
    "XTTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2"
).strip()
MODEL: Any | None = None
MODEL_DEVICE = "not-loaded"
MODEL_LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()
LOGGER = logging.getLogger("xtts-service")


class SpeechRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    voice_id: str = Field(pattern=r"^voice_[0-9a-f]{32}$")
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    language: str

    @field_validator("user_id", "text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("language")
    @classmethod
    def supported_language(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_LANGUAGES:
            raise ValueError("language is not supported by this XTTS service")
        return normalized


class VoiceResponse(BaseModel):
    id: str


def require_service_key(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    expected = os.getenv("XTTS_SERVICE_API_KEY", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="XTTS_SERVICE_API_KEY is not set.")
    supplied = authorization.removeprefix("Bearer ") if authorization else ""
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid service credentials.")


def user_voice_dir(user_id: str) -> Path:
    user_hash = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    return VOICE_DATA_DIR / user_hash


def voice_path(user_id: str, voice_id: str) -> Path:
    if not VOICE_ID_PATTERN.fullmatch(voice_id):
        raise HTTPException(status_code=400, detail="Invalid voice reference.")
    return user_voice_dir(user_id) / f"{voice_id}.wav"


def configured_device(torch_module: Any) -> str:
    requested = os.getenv("XTTS_DEVICE", "auto").strip().lower()
    if requested == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        if torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("XTTS_DEVICE=cuda, but CUDA is not available.")
    if requested == "mps" and not torch_module.backends.mps.is_available():
        raise RuntimeError("XTTS_DEVICE=mps, but Apple Metal is not available.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise RuntimeError("XTTS_DEVICE must be auto, cpu, cuda, or mps.")
    return requested


def get_model() -> Any:
    global MODEL, MODEL_DEVICE
    if MODEL is not None:
        return MODEL

    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL
        if os.getenv("COQUI_TOS_AGREED", "").strip() != "1":
            raise RuntimeError(
                "Set COQUI_TOS_AGREED=1 after reviewing the XTTS-v2 CPML license."
            )

        import torch
        from TTS.api import TTS

        MODEL_DEVICE = configured_device(torch)
        MODEL = TTS(MODEL_NAME).to(MODEL_DEVICE)
        return MODEL


async def save_upload(upload: UploadFile, destination: Path) -> int:
    size = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413, detail="Voice sample must be 10 MiB or smaller."
                )
            output.write(chunk)
    await upload.close()
    if size == 0:
        raise HTTPException(status_code=400, detail="Voice sample is empty.")
    return size


def normalize_audio(source: Path, destination: Path) -> None:
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-t",
                str(MAX_SAMPLE_SECONDS + 1),
                "-ac",
                "1",
                "-ar",
                "24000",
                "-c:a",
                "pcm_s16le",
                str(destination),
            ],
            capture_output=True,
            check=False,
            timeout=90,
        )
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503, detail="ffmpeg is not installed in the XTTS service."
        ) from error
    except subprocess.TimeoutExpired as error:
        raise HTTPException(
            status_code=400, detail="Voice sample took too long to decode."
        ) from error

    if completed.returncode != 0 or not destination.exists():
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise HTTPException(
            status_code=400,
            detail=detail[-500:] if detail else "Voice sample could not be decoded.",
        )

    try:
        with wave.open(str(destination), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
    except (wave.Error, ZeroDivisionError) as error:
        raise HTTPException(
            status_code=400, detail="Normalized voice sample is invalid."
        ) from error

    if duration < MIN_SAMPLE_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Voice sample must be at least {MIN_SAMPLE_SECONDS:g} seconds.",
        )
    if duration > MAX_SAMPLE_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Voice sample must be {MAX_SAMPLE_SECONDS:g} seconds or shorter.",
        )


def synthesize(request: SpeechRequest, reference_path: Path, output_path: Path) -> None:
    try:
        with INFERENCE_LOCK:
            if not reference_path.is_file():
                raise HTTPException(status_code=404, detail="Saved voice was not found.")
            model = get_model()
            model.tts_to_file(
                text=request.text,
                file_path=str(output_path),
                speaker_wav=[str(reference_path)],
                language=request.language,
                split_sentences=True,
            )
    except HTTPException:
        raise
    except Exception as error:
        LOGGER.exception("XTTS inference failed")
        raise HTTPException(status_code=502, detail="XTTS inference failed.") from error

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise HTTPException(status_code=502, detail="XTTS produced no audio.")


def install_voice(temporary_path: Path, destination: Path) -> None:
    with INFERENCE_LOCK:
        temporary_path.replace(destination)
        for previous_voice in destination.parent.glob("voice_*.wav"):
            if previous_voice != destination:
                previous_voice.unlink(missing_ok=True)


def remove_voice(path: Path) -> None:
    with INFERENCE_LOCK:
        path.unlink(missing_ok=True)
        try:
            path.parent.rmdir()
        except OSError:
            pass


def remove_file(path: Path) -> None:
    path.unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    VOICE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if os.getenv("XTTS_PRELOAD_MODEL", "true").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        await asyncio.to_thread(get_model)
    yield


app = FastAPI(
    title="Research Copilot XTTS-v2 service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_loaded": MODEL is not None,
        "device": MODEL_DEVICE,
    }


@app.post(
    "/v1/voices",
    response_model=VoiceResponse,
    dependencies=[Depends(require_service_key)],
)
async def create_voice(
    user_id: Annotated[str, Form(min_length=1, max_length=128)],
    name: Annotated[str, Form(min_length=1, max_length=64)],
    audio_sample: Annotated[UploadFile, File()],
) -> VoiceResponse:
    del name  # The display name remains in the signed Next.js token.
    suffix = Path(audio_sample.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported voice sample format.")

    voice_id = f"voice_{uuid.uuid4().hex}"
    directory = user_voice_dir(user_id)
    directory.mkdir(parents=True, exist_ok=True)
    destination = voice_path(user_id, voice_id)
    temporary_destination = directory / f".{voice_id}.tmp.wav"

    try:
        with tempfile.TemporaryDirectory(prefix="xtts-upload-") as temporary_dir:
            source = Path(temporary_dir) / f"sample{suffix}"
            await save_upload(audio_sample, source)
            await asyncio.to_thread(normalize_audio, source, temporary_destination)
        await asyncio.to_thread(
            install_voice, temporary_destination, destination
        )
    except Exception:
        temporary_destination.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise

    return VoiceResponse(id=voice_id)


@app.delete(
    "/v1/voices/{voice_id}",
    status_code=204,
    response_class=Response,
    dependencies=[Depends(require_service_key)],
)
async def delete_voice(
    voice_id: str,
    user_id: Annotated[str, Query(min_length=1, max_length=128)],
) -> Response:
    path = voice_path(user_id, voice_id)
    await asyncio.to_thread(remove_voice, path)
    return Response(status_code=204)


@app.post(
    "/v1/speech",
    dependencies=[Depends(require_service_key)],
    response_class=FileResponse,
)
async def generate_speech(request: SpeechRequest) -> FileResponse:
    reference_path = voice_path(request.user_id, request.voice_id)
    output = tempfile.NamedTemporaryFile(prefix="xtts-speech-", suffix=".wav", delete=False)
    output_path = Path(output.name)
    output.close()
    output_path.unlink(missing_ok=True)

    try:
        await asyncio.to_thread(synthesize, request, reference_path, output_path)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise

    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="speech.wav",
        background=BackgroundTask(remove_file, output_path),
    )
