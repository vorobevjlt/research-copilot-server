## Document processing

By default, uploaded documents and websites are processed by FastAPI after the
response is sent. This works locally without Redis.

For a production Celery worker, set:

```env
DOCUMENT_PROCESSING_MODE=celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

Then start a worker with:

```bash
celery -A tasks.celery_app worker --loglevel=info
```
