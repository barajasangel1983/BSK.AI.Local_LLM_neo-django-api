# ingestion/urls.py
#
# URL patterns for the ingestion app.
# Mounted under /api/ in the project-level urls.py, so:
#
# - POST /api/rag/ingest/
# - GET  /api/rag/jobs/
# - GET  /api/rag/jobs/<uuid:job_id>/
# - GET  /api/rag/health/

from django.urls import path
from . import views

urlpatterns = [
    path("rag/ingest/", views.ingest_document, name="ingest-document"),
    path("rag/jobs/", views.list_jobs, name="job-list"),
    path("rag/jobs/<uuid:job_id>/", views.job_detail, name="job-detail"),
    path("rag/health/", views.ingestion_health, name="ingestion-health"),
]
