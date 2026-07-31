import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from src.models.index import ConfirmFileUploadRequest, FileUploadRequest
from src.routes import projectFilesRoutes as routes


def make_supabase_query(*results):
    query = MagicMock()
    for method_name in ("select", "update", "delete", "eq", "order", "insert"):
        getattr(query, method_name).return_value = query
    query.execute.side_effect = [SimpleNamespace(data=result) for result in results]
    supabase = MagicMock()
    supabase.table.return_value = query
    return supabase


class ConfirmUploadRouteTests(unittest.TestCase):
    def setUp(self):
        self.document = {
            "id": "document-id",
            "project_id": "project-id",
            "clerk_id": "user-id",
            "filename": "report.pdf",
            "s3_key": "projects/project-id/documents/report.pdf",
            "file_size": 1024,
            "file_type": "application/pdf",
            "processing_status": "uploading",
            "processing_details": {},
        }
        self.request = ConfirmFileUploadRequest(s3_key=self.document["s3_key"])

    def call_confirm(self):
        return asyncio.run(
            routes.confirm_file_upload_to_s3(
                "project-id", self.request, "user-id"
            )
        )

    def test_verifies_s3_before_queueing(self):
        queued_document = {**self.document, "processing_status": "queued"}
        task_document = {**queued_document, "task_id": "task-id"}
        fake_supabase = make_supabase_query(
            [self.document], [queued_document], [task_document]
        )
        fake_s3 = MagicMock()
        fake_s3.head_object.return_value = {
            "ContentLength": 1024,
            "ContentType": "application/pdf",
        }
        fake_task = MagicMock()
        fake_task.delay.return_value = SimpleNamespace(id="task-id")

        with (
            patch.object(routes, "supabase", fake_supabase),
            patch.object(routes, "s3_client", fake_s3),
            patch.object(routes, "perform_rag_ingestion_task", fake_task),
        ):
            response = self.call_confirm()

        self.assertEqual(response["data"]["task_id"], "task-id")
        fake_s3.head_object.assert_called_once()
        fake_task.delay.assert_called_once_with("document-id")

    def test_rejects_incomplete_s3_object_without_queueing(self):
        fake_supabase = make_supabase_query([self.document])
        fake_s3 = MagicMock()
        fake_s3.head_object.return_value = {
            "ContentLength": 512,
            "ContentType": "application/pdf",
        }
        fake_task = MagicMock()

        with (
            patch.object(routes, "supabase", fake_supabase),
            patch.object(routes, "s3_client", fake_s3),
            patch.object(routes, "perform_rag_ingestion_task", fake_task),
            patch.object(routes, "mark_upload_failed") as mark_failed,
        ):
            with self.assertRaises(HTTPException) as raised:
                self.call_confirm()

        self.assertEqual(raised.exception.status_code, 422)
        mark_failed.assert_called_once()
        fake_task.delay.assert_not_called()

    def test_confirmation_is_idempotent_after_queueing(self):
        queued_document = {**self.document, "processing_status": "queued"}
        fake_supabase = make_supabase_query([queued_document])
        fake_s3 = MagicMock()
        fake_task = MagicMock()

        with (
            patch.object(routes, "supabase", fake_supabase),
            patch.object(routes, "s3_client", fake_s3),
            patch.object(routes, "perform_rag_ingestion_task", fake_task),
        ):
            response = self.call_confirm()

        self.assertEqual(response["data"]["processing_status"], "queued")
        fake_s3.head_object.assert_not_called()
        fake_task.delay.assert_not_called()


class CreateUploadRouteTests(unittest.TestCase):
    def test_presigned_upload_binds_size_and_trusted_content_type(self):
        created_document = {
            "id": "document-id",
            "processing_status": "uploading",
        }
        fake_supabase = make_supabase_query(
            [{"id": "project-id"}], [created_document]
        )
        fake_s3 = MagicMock()
        fake_s3.generate_presigned_url.return_value = "https://upload.example"
        request = FileUploadRequest(
            filename="report.pdf",
            file_type="application/octet-stream",
            file_size=4096,
        )

        with (
            patch.object(routes, "supabase", fake_supabase),
            patch.object(routes, "s3_client", fake_s3),
        ):
            response = asyncio.run(
                routes.get_upload_presigned_url("project-id", request, "user-id")
            )

        params = fake_s3.generate_presigned_url.call_args.kwargs["Params"]
        self.assertEqual(params["ContentLength"], 4096)
        self.assertEqual(params["ContentType"], "application/pdf")
        self.assertEqual(response["data"]["content_type"], "application/pdf")
        self.assertEqual(response["data"]["document"], created_document)


class StaleUploadTests(unittest.TestCase):
    def test_expired_upload_is_marked_failed(self):
        stale_document = {
            "id": "stale-id",
            "processing_status": "uploading",
            "processing_details": {},
            "created_at": (
                datetime.now(timezone.utc) - timedelta(hours=2)
            ).isoformat(),
        }

        with patch.object(routes, "mark_upload_failed") as mark_failed:
            result = routes.expire_stale_uploads([stale_document])

        self.assertEqual(result[0]["processing_status"], "failed")
        self.assertIn("expired", result[0]["processing_details"]["error"]["message"])
        mark_failed.assert_called_once()


if __name__ == "__main__":
    unittest.main()
