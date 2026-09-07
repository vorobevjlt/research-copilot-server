from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
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
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from starlette.background import BackgroundTask

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_SONG_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_TEXT_LENGTH = 4096
MIN_SAMPLE_SECONDS = 3.0
MAX_SAMPLE_SECONDS = 120.0
MAX_SONG_SECONDS = 5 * 60.0
SONG_JOB_RETENTION_SECONDS = 6 * 60 * 60
SONG_CLEANUP_INTERVAL_SECONDS = 15 * 60
UPLOAD_CHUNK_BYTES = 1024 * 1024
VOICE_ID_PATTERN = re.compile(r"^voice_[0-9a-f]{32}$")
SONG_JOB_ID_PATTERN = re.compile(r"^song_[0-9a-f]{32}$")
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
SONG_JOB_DIR = Path(os.getenv("SONG_JOB_DIR", "/data/song-jobs")).resolve()
MODEL_NAME = os.getenv(
    "XTTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2"
).strip()
INFERENCE_LOCK = threading.Lock()
SONG_JOBS_LOCK = threading.Lock()
SONG_JOBS: dict[str, "SongJob"] = {}
SONG_TASKS: set[asyncio.Task[None]] = set()
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


class SongJobResponse(BaseModel):
    id: str
    status: str
    phase: str
    progress: int = Field(ge=0, le=100)
    error: str | None = None


@dataclass(frozen=True)
class SongAccess:
    user_id: str
    voice_id: str
    origin: str


@dataclass
class SongJob:
    id: str
    user_id: str
    voice_id: str
    directory: Path
    source_path: Path
    created_at: float = field(default_factory=time.time)
    status: str = "queued"
    phase: str = "queued"
    progress: int = 5
    error: str | None = None
    result_path: Path | None = None
    finished_at: float | None = None


def require_service_key(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    expected = os.getenv("XTTS_SERVICE_API_KEY", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="XTTS_SERVICE_API_KEY is not set.")
    supplied = authorization.removeprefix("Bearer ") if authorization else ""
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid service credentials.")


def decode_base64url(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}")


def require_song_access(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> SongAccess:
    secret = os.getenv("XTTS_SERVICE_API_KEY", "").strip()
    if not secret:
        raise HTTPException(status_code=503, detail="XTTS_SERVICE_API_KEY is not set.")

    token = authorization.removeprefix("Bearer ") if authorization else ""
    if len(token) > 4096:
        raise HTTPException(status_code=401, detail="Invalid song access token.")
    parts = token.split(".")
    if len(parts) != 2 or not all(parts):
        raise HTTPException(status_code=401, detail="Invalid song access token.")

    encoded, supplied_signature = parts
    expected_signature = base64.urlsafe_b64encode(
        hmac.new(secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256).digest()
    ).rstrip(b"=").decode("ascii")
    if not hmac.compare_digest(supplied_signature, expected_signature):
        raise HTTPException(status_code=401, detail="Invalid song access token.")

    try:
        payload = json.loads(decode_base64url(encoded))
        user_id = payload["user_id"]
        voice_id = payload["voice_id"]
        origin = payload["origin"]
        expires_at = payload["expires_at"]
        nonce = payload["nonce"]
        if (
            payload["version"] != 1
            or not isinstance(user_id, str)
            or not 1 <= len(user_id) <= 128
            or not isinstance(voice_id, str)
            or not VOICE_ID_PATTERN.fullmatch(voice_id)
            or not isinstance(origin, str)
            or not origin.startswith(("http://", "https://"))
            or not isinstance(expires_at, int)
            or expires_at < int(time.time())
            or not isinstance(nonce, str)
            or not re.fullmatch(r"[0-9a-f]{32}", nonce)
        ):
            raise ValueError("Invalid claims")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise HTTPException(status_code=401, detail="Invalid song access token.") from None

    request_origin = request.headers.get("origin")
    if not request_origin or not hmac.compare_digest(request_origin, origin):
        raise HTTPException(status_code=403, detail="Song access origin does not match.")

    return SongAccess(user_id=user_id, voice_id=voice_id, origin=origin)


def user_voice_dir(user_id: str) -> Path:
    user_hash = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    return VOICE_DATA_DIR / user_hash


def voice_path(user_id: str, voice_id: str) -> Path:
    if not VOICE_ID_PATTERN.fullmatch(voice_id):
        raise HTTPException(status_code=400, detail="Invalid voice reference.")
    return user_voice_dir(user_id) / f"{voice_id}.wav"


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


async def save_song_upload(upload: UploadFile, destination: Path) -> int:
    size = 0
    try:
        with destination.open("wb") as output:
            while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_SONG_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413, detail="Song must be 50 MiB or smaller."
                    )
                output.write(chunk)
    finally:
        await upload.close()
    if size == 0:
        raise HTTPException(status_code=400, detail="Song file is empty.")
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
    with INFERENCE_LOCK:
        if not reference_path.is_file():
            raise HTTPException(status_code=404, detail="Saved voice was not found.")
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    os.getenv("SPEECH_CONVERTER_SCRIPT", "/app/speech_convert.py"),
                    "--source",
                    str(reference_path),
                    "--output",
                    str(output_path),
                    "--language",
                    request.language,
                    "--model",
                    MODEL_NAME,
                    "--device",
                    os.getenv("XTTS_DEVICE", "cpu"),
                ],
                input=request.text.encode("utf-8"),
                capture_output=True,
                check=False,
                timeout=3600,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            raise HTTPException(status_code=502, detail="XTTS inference failed.") from error
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            if detail:
                LOGGER.error("XTTS inference failed: %s", detail[-2000:])
            raise HTTPException(status_code=502, detail="XTTS inference failed.")

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


class SongConversionError(Exception):
    pass


def update_song_job(job_id: str, **changes: Any) -> None:
    with SONG_JOBS_LOCK:
        job = SONG_JOBS.get(job_id)
        if job is None:
            return
        for key, value in changes.items():
            setattr(job, key, value)


def song_job_response(job: SongJob) -> SongJobResponse:
    return SongJobResponse(
        id=job.id,
        status=job.status,
        phase=job.phase,
        progress=job.progress,
        error=job.error,
    )


def run_audio_command(
    arguments: list[str],
    *,
    timeout: int,
    failure_message: str,
) -> None:
    try:
        completed = subprocess.run(
            arguments,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as error:
        raise SongConversionError(failure_message) from error
    except subprocess.TimeoutExpired as error:
        raise SongConversionError(failure_message) from error

    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        if detail:
            LOGGER.error("Song command failed: %s", detail[-2000:])
        raise SongConversionError(failure_message)


def song_duration(path: Path) -> float:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            check=False,
            timeout=60,
        )
        duration = float(completed.stdout.decode("utf-8").strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, TypeError, ValueError) as error:
        raise SongConversionError("The song could not be decoded.") from error

    if completed.returncode != 0 or not 0 < duration <= MAX_SONG_SECONDS:
        if duration > MAX_SONG_SECONDS:
            raise SongConversionError("Song must be five minutes or shorter.")
        raise SongConversionError("The song could not be decoded.")
    return duration


def retain_song_result(job: SongJob, result_path: Path) -> None:
    for child in job.directory.iterdir():
        if child == result_path:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def process_song_job(job_id: str) -> None:
    with SONG_JOBS_LOCK:
        job = SONG_JOBS.get(job_id)
    if job is None:
        return

    reference_path = voice_path(job.user_id, job.voice_id)
    normalized_song = job.directory / "song.wav"
    separation_dir = job.directory / "separated"
    converted_vocals = job.directory / "converted-vocals.wav"
    result_path = job.directory / "converted-song.mp3"

    try:
        update_song_job(
            job_id, status="processing", phase="preparing", progress=10
        )
        song_duration(job.source_path)

        with INFERENCE_LOCK:
            if not reference_path.is_file():
                raise SongConversionError("Saved voice was not found.")
            run_audio_command(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(job.source_path),
                    "-map",
                    "0:a:0",
                    "-ac",
                    "2",
                    "-ar",
                    "44100",
                    "-c:a",
                    "pcm_s16le",
                    str(normalized_song),
                ],
                timeout=600,
                failure_message="The song could not be decoded.",
            )

            update_song_job(job_id, phase="separating", progress=20)
            run_audio_command(
                [
                    sys.executable,
                    "-m",
                    "demucs",
                    "--two-stems=vocals",
                    "-n",
                    "htdemucs",
                    "--device",
                    "cpu",
                    "-j",
                    "1",
                    "-o",
                    str(separation_dir),
                    str(normalized_song),
                ],
                timeout=3600,
                failure_message="Unable to separate the song vocals.",
            )

            stem_dir = separation_dir / "htdemucs" / normalized_song.stem
            vocals_path = stem_dir / "vocals.wav"
            accompaniment_path = stem_dir / "no_vocals.wav"
            if not vocals_path.is_file() or not accompaniment_path.is_file():
                raise SongConversionError("Unable to separate the song vocals.")

            update_song_job(job_id, phase="converting", progress=55)
            run_audio_command(
                [
                    sys.executable,
                    os.getenv("SONG_CONVERTER_SCRIPT", "/app/song_convert.py"),
                    "--source",
                    str(vocals_path),
                    "--target",
                    str(reference_path),
                    "--output",
                    str(converted_vocals),
                    "--device",
                    os.getenv("XTTS_DEVICE", "cpu"),
                ],
                timeout=7200,
                failure_message="Unable to convert the singing voice.",
            )

            update_song_job(job_id, phase="remixing", progress=90)
            run_audio_command(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(accompaniment_path),
                    "-i",
                    str(converted_vocals),
                    "-filter_complex",
                    "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.95",
                    "-ar",
                    "44100",
                    "-ac",
                    "2",
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    "256k",
                    str(result_path),
                ],
                timeout=900,
                failure_message="Unable to remix the converted song.",
            )

        if not result_path.is_file() or result_path.stat().st_size == 0:
            raise SongConversionError("Song conversion produced no audio.")
        retain_song_result(job, result_path)
        update_song_job(
            job_id,
            status="ready",
            phase="ready",
            progress=100,
            result_path=result_path,
            finished_at=time.time(),
        )
    except SongConversionError as error:
        LOGGER.warning("Song conversion %s failed: %s", job_id, error)
        retain_song_result(job, Path("/__no_song_result__"))
        update_song_job(
            job_id,
            status="failed",
            phase="failed",
            progress=100,
            error=str(error),
            finished_at=time.time(),
        )
    except Exception:
        LOGGER.exception("Song conversion %s failed", job_id)
        retain_song_result(job, Path("/__no_song_result__"))
        update_song_job(
            job_id,
            status="failed",
            phase="failed",
            progress=100,
            error="Song conversion failed.",
            finished_at=time.time(),
        )


async def run_song_job(job_id: str) -> None:
    await asyncio.to_thread(process_song_job, job_id)


def require_matching_song_job(job_id: str, access: SongAccess) -> SongJob:
    if not SONG_JOB_ID_PATTERN.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="Song conversion was not found.")
    with SONG_JOBS_LOCK:
        job = SONG_JOBS.get(job_id)
    if (
        job is None
        or job.user_id != access.user_id
        or job.voice_id != access.voice_id
    ):
        raise HTTPException(status_code=404, detail="Song conversion was not found.")
    return job


def remove_song_job(job: SongJob) -> None:
    shutil.rmtree(job.directory, ignore_errors=True)
    with SONG_JOBS_LOCK:
        SONG_JOBS.pop(job.id, None)


async def clean_expired_song_jobs() -> None:
    while True:
        await asyncio.sleep(SONG_CLEANUP_INTERVAL_SECONDS)
        cutoff = time.time() - SONG_JOB_RETENTION_SECONDS
        with SONG_JOBS_LOCK:
            expired = [
                job
                for job in SONG_JOBS.values()
                if job.finished_at is not None and job.finished_at < cutoff
            ]
        for job in expired:
            await asyncio.to_thread(remove_song_job, job)


@asynccontextmanager
async def lifespan(_: FastAPI):
    VOICE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SONG_JOB_DIR.mkdir(parents=True, exist_ok=True)
    for stale_job in SONG_JOB_DIR.glob("song_*"):
        if stale_job.is_dir():
            shutil.rmtree(stale_job, ignore_errors=True)
    cleanup_task = asyncio.create_task(clean_expired_song_jobs())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task


app = FastAPI(
    title="Research Copilot XTTS-v2 service",
    version="1.1.0",
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "XTTS_ALLOWED_ORIGINS",
        "https://research-copilot-iota.vercel.app,http://localhost:3000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=600,
)


@app.get("/health")
async def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_loaded": False,
        "device": os.getenv("XTTS_DEVICE", "cpu"),
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


@app.post(
    "/v1/song-conversions",
    response_model=SongJobResponse,
    status_code=202,
)
async def create_song_conversion(
    song: Annotated[UploadFile, File()],
    access: Annotated[SongAccess, Depends(require_song_access)],
) -> SongJobResponse:
    suffix = Path(song.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported song format.")
    if not voice_path(access.user_id, access.voice_id).is_file():
        raise HTTPException(status_code=404, detail="Saved voice was not found.")

    with SONG_JOBS_LOCK:
        active_jobs = [
            job
            for job in SONG_JOBS.values()
            if job.status in {"uploading", "queued", "processing"}
        ]
        if any(job.user_id == access.user_id for job in active_jobs):
            raise HTTPException(
                status_code=409,
                detail="A song conversion is already running for this user.",
            )
        if len(active_jobs) >= 3:
            raise HTTPException(
                status_code=429,
                detail="The song conversion queue is full. Try again later.",
            )

        job_id = f"song_{uuid.uuid4().hex}"
        directory = SONG_JOB_DIR / job_id
        directory.mkdir(parents=True, exist_ok=False)
        source_path = directory / f"upload{suffix}"
        job = SongJob(
            id=job_id,
            user_id=access.user_id,
            voice_id=access.voice_id,
            directory=directory,
            source_path=source_path,
            status="uploading",
            phase="uploading",
            progress=1,
        )
        SONG_JOBS[job_id] = job

    try:
        await save_song_upload(song, source_path)
    except Exception:
        await asyncio.to_thread(remove_song_job, job)
        raise

    update_song_job(job_id, status="queued", phase="queued", progress=5)

    task = asyncio.create_task(run_song_job(job_id))
    SONG_TASKS.add(task)
    task.add_done_callback(SONG_TASKS.discard)
    return song_job_response(job)


@app.get(
    "/v1/song-conversions/{job_id}",
    response_model=SongJobResponse,
)
async def get_song_conversion(
    job_id: str,
    access: Annotated[SongAccess, Depends(require_song_access)],
) -> SongJobResponse:
    return song_job_response(require_matching_song_job(job_id, access))


@app.get(
    "/v1/song-conversions/{job_id}/result",
    response_class=FileResponse,
)
async def get_song_conversion_result(
    job_id: str,
    access: Annotated[SongAccess, Depends(require_song_access)],
) -> FileResponse:
    job = require_matching_song_job(job_id, access)
    if job.status != "ready" or job.result_path is None:
        raise HTTPException(status_code=409, detail="Song conversion is not ready.")
    if not job.result_path.is_file():
        raise HTTPException(status_code=404, detail="Converted song was not found.")

    return FileResponse(
        job.result_path,
        media_type="audio/mpeg",
        filename="converted-song.mp3",
        background=BackgroundTask(remove_song_job, job),
    )


@app.delete(
    "/v1/song-conversions/{job_id}",
    status_code=204,
    response_class=Response,
)
async def delete_song_conversion(
    job_id: str,
    access: Annotated[SongAccess, Depends(require_song_access)],
) -> Response:
    job = require_matching_song_job(job_id, access)
    if job.status in {"uploading", "queued", "processing"}:
        raise HTTPException(
            status_code=409, detail="A running song conversion cannot be deleted."
        )
    await asyncio.to_thread(remove_song_job, job)
    return Response(status_code=204)
