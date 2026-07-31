import os
import tempfile
import unittest
import uuid
import zipfile
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from src.models.index import ProcessingStatus
from src.rag.ingestion import index as ingestion
from src.rag.ingestion.utils import validate_document_content


class DocumentIngestionFailureTests(unittest.TestCase):
    def test_process_document_persists_failed_status(self):
        query = MagicMock()
        query.select.return_value = query
        query.eq.return_value = query
        query.execute.return_value = SimpleNamespace(data=[])
        fake_supabase = MagicMock()
        fake_supabase.table.return_value = query

        with (
            patch.object(ingestion, "supabase", fake_supabase),
            patch.object(ingestion, "update_status_in_database") as update_status,
        ):
            with self.assertRaises(Exception):
                ingestion.process_document("missing-document")

        self.assertEqual(
            update_status.call_args_list[0],
            call("missing-document", ProcessingStatus.PROCESSING),
        )
        self.assertEqual(
            update_status.call_args_list[-1].args[:2],
            ("missing-document", ProcessingStatus.FAILED),
        )

    def test_partition_failure_removes_temporary_file(self):
        document_id = f"upload-test-{uuid.uuid4()}"
        expected_temp_path = f"/tmp/{document_id}.pdf"
        fake_s3 = MagicMock()

        def write_downloaded_file(_bucket, _key, filename):
            with open(filename, "wb") as temporary_file:
                temporary_file.write(b"not-a-real-pdf")

        fake_s3.download_file.side_effect = write_downloaded_file
        document = {
            "source_type": "file",
            "s3_key": "documents/report.pdf",
            "filename": "report.pdf",
        }

        with (
            patch.object(ingestion, "s3_client", fake_s3),
            patch.object(ingestion, "validate_document_content"),
            patch.object(
                ingestion,
                "partition_document",
                side_effect=RuntimeError("partition failed"),
            ),
        ):
            with self.assertRaises(Exception):
                ingestion.download_content_and_partition(document_id, document)

        self.assertFalse(os.path.exists(expected_temp_path))

    def test_rejects_spoofed_pdf_content(self):
        with tempfile.NamedTemporaryFile() as uploaded_file:
            uploaded_file.write(b"this is not a PDF")
            uploaded_file.flush()

            with self.assertRaisesRegex(ValueError, "valid PDF"):
                validate_document_content(uploaded_file.name, "pdf")

    def test_accepts_openxml_document_with_expected_structure(self):
        with tempfile.NamedTemporaryFile(suffix=".docx") as uploaded_file:
            with zipfile.ZipFile(uploaded_file.name, "w") as archive:
                archive.writestr("[Content_Types].xml", "<Types />")
                archive.writestr("word/document.xml", "<document />")

            validate_document_content(uploaded_file.name, "docx")

    def test_rejects_binary_text_file(self):
        with tempfile.NamedTemporaryFile() as uploaded_file:
            uploaded_file.write(b"text\x00binary")
            uploaded_file.flush()

            with self.assertRaisesRegex(ValueError, "binary data"):
                validate_document_content(uploaded_file.name, "txt")


if __name__ == "__main__":
    unittest.main()
