import unittest

from pydantic import ValidationError

from src.models.index import (
    DOCUMENT_MIME_TYPES,
    MAX_DOCUMENT_FILE_SIZE,
    FileUploadRequest,
    ProcessingStatus,
)


class FileUploadRequestTests(unittest.TestCase):
    def test_accepts_and_normalizes_supported_files(self):
        cases = [
            ("report.pdf", "application/pdf", DOCUMENT_MIME_TYPES[".pdf"]),
            ("notes.md", "text/plain", DOCUMENT_MIME_TYPES[".md"]),
            ("notes.md", "", DOCUMENT_MIME_TYPES[".md"]),
            (
                "deck.PPTX",
                "application/octet-stream",
                DOCUMENT_MIME_TYPES[".pptx"],
            ),
        ]

        for filename, reported_type, expected_type in cases:
            with self.subTest(filename=filename, reported_type=reported_type):
                request = FileUploadRequest(
                    filename=filename,
                    file_type=reported_type,
                    file_size=1024,
                )
                self.assertEqual(request.file_type, expected_type)

    def test_rejects_invalid_metadata(self):
        cases = [
            {
                "filename": "archive.zip",
                "file_type": "application/zip",
                "file_size": 1,
            },
            {
                "filename": "../report.pdf",
                "file_type": "application/pdf",
                "file_size": 1,
            },
            {
                "filename": "folder\\report.pdf",
                "file_type": "application/pdf",
                "file_size": 1,
            },
            {
                "filename": "report.pdf",
                "file_type": "text/plain",
                "file_size": 1,
            },
            {
                "filename": "report.pdf",
                "file_type": "application/pdf",
                "file_size": 0,
            },
            {
                "filename": "report.pdf",
                "file_type": "application/pdf",
                "file_size": MAX_DOCUMENT_FILE_SIZE + 1,
            },
        ]

        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    FileUploadRequest(**payload)

    def test_processing_status_includes_failed_state(self):
        self.assertEqual(ProcessingStatus.FAILED.value, "failed")


if __name__ == "__main__":
    unittest.main()
