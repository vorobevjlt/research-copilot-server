from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from typing import Optional, List
from enum import Enum


MAX_DOCUMENT_FILE_SIZE = 50 * 1024 * 1024
DOCUMENT_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain",
    ".md": "text/markdown",
}

DOCUMENT_ACCEPTED_MIME_TYPES = {
    ".pdf": {"application/pdf"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    },
    ".pptx": {
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    },
    ".txt": {"text/plain"},
    ".md": {"text/markdown", "text/x-markdown", "text/plain"},
}


class ProjectCreate(BaseModel):
    name: str = Field(..., description="The name of the project")
    description: Optional[str] = Field(None, description="Project description")


class ChatCreate(BaseModel):
    title: str = Field(..., description="The title of the chat")
    project_id: str = Field(..., description="The ID of the project")


class ProjectSettings(BaseModel):
    embedding_model: str = Field(..., description="The embedding model to use")
    rag_strategy: str = Field(..., description="The RAG strategy to use")
    agent_type: str = Field(..., description="The agent type to use")
    chunks_per_search: int = Field(..., description="The number of chunks per search")
    final_context_size: int = Field(..., description="The final context size")
    similarity_threshold: float = Field(..., description="The similarity threshold")
    number_of_queries: int = Field(..., description="The number of queries")
    reranking_enabled: bool = Field(..., description="Whether reranking is enabled")
    reranking_model: str = Field(..., description="The reranking model to use")
    vector_weight: float = Field(..., description="The vector weight")
    keyword_weight: float = Field(..., description="The keyword weight")


class FileUploadRequest(BaseModel):
    filename: str = Field(
        ..., min_length=1, max_length=255, description="The name of the file"
    )
    file_type: str = Field(
        default="", max_length=255, description="The browser-reported MIME type"
    )
    file_size: int = Field(
        ...,
        gt=0,
        le=MAX_DOCUMENT_FILE_SIZE,
        description="The size of the file in bytes",
    )

    @model_validator(mode="after")
    def validate_document(self):
        self.filename = self.filename.strip()
        if (
            not self.filename
            or Path(self.filename).name != self.filename
            or "/" in self.filename
            or "\\" in self.filename
            or "\x00" in self.filename
        ):
            raise ValueError("filename must be a plain file name")

        extension = Path(self.filename).suffix.lower()
        if extension not in DOCUMENT_MIME_TYPES:
            supported = ", ".join(DOCUMENT_MIME_TYPES)
            raise ValueError(f"Unsupported file extension. Supported types: {supported}")

        reported_type = self.file_type.lower().strip()
        accepted_types = DOCUMENT_ACCEPTED_MIME_TYPES[extension]
        if reported_type and reported_type not in accepted_types | {
            "application/octet-stream"
        }:
            raise ValueError(
                f"File type {reported_type!r} does not match the {extension} extension"
            )

        # Use a trusted, extension-derived type for the signed upload and stored record.
        self.file_type = DOCUMENT_MIME_TYPES[extension]
        return self


class ConfirmFileUploadRequest(BaseModel):
    s3_key: str = Field(..., min_length=1, max_length=1024)


class ProcessingStatus(str, Enum):
    UPLOADING = "uploading"
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    PARTITIONING = "partitioning"
    CHUNKING = "chunking"
    SUMMARISING = "summarising"
    VECTORIZATION = "vectorization"
    COMPLETED = "completed"
    FAILED = "failed"


class UrlRequest(BaseModel):
    url: str = Field(..., description="The URL to process")


class MessageCreate(BaseModel):
    content: str = Field(..., description="The content of the message")


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class QueryVariations(BaseModel):
    queries: List[str] = Field(..., description="The variations of the query")

class InputGuardrailCheck(BaseModel):
    """Schema for input safety check"""
    is_safe: bool = Field(description="Whether the input is safe to process")
    is_toxic: bool = Field(description="Contains toxic or harmful content")
    is_prompt_injection: bool = Field(description="Appears to be a prompt injection attempt")
    contains_pii: bool = Field(description="Contains personal identifiable information")
    reason: str = Field(description="Brief explanation if unsafe, empty string if safe")
