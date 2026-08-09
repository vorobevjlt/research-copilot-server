import logging
import tempfile
import unittest
from pathlib import Path

from src.config.logging import configure_file_handler
from src.services import celery as celery_service


class WorkerLoggingTests(unittest.TestCase):
    def test_worker_logging_is_configured_by_celery_startup_signal(self):
        original_configure_logging = celery_service.configure_logging
        calls = []
        celery_service.configure_logging = lambda **kwargs: calls.append(kwargs)
        try:
            celery_service.setup_worker_logging()
        finally:
            celery_service.configure_logging = original_configure_logging

        self.assertEqual(calls, [{"log_filename": "worker.log"}])

    def test_file_handler_uses_the_requested_log_directory(self):
        logger = logging.Logger("worker-file-test", level=logging.INFO)

        with tempfile.TemporaryDirectory() as temporary_directory:
            handler = configure_file_handler(
                logger, "worker.log", Path(temporary_directory)
            )
            try:
                logger.info("task started")
                handler.flush()
            finally:
                handler.close()

            worker_log = Path(temporary_directory) / "worker.log"
            self.assertEqual(worker_log.read_text(), "task started\n")


if __name__ == "__main__":
    unittest.main()
