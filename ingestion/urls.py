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
from . import library_views, views

urlpatterns = [
    # Document library (shared by RAG Lab and GraphLab)
    path("documents/", library_views.documents, name="library-documents"),
    path("documents/embed/", library_views.documents_embed, name="library-documents-embed"),
    path("documents/<uuid:doc_id>/", library_views.document_detail, name="library-document"),
    path("documents/<uuid:doc_id>/embed/", library_views.document_embed, name="library-document-embed"),
    path("documents/<uuid:doc_id>/embeddings/", library_views.document_embeddings, name="library-document-embeddings"),
    path("jobs/", library_views.jobs, name="library-jobs"),
    path("jobs/<uuid:job_id>/", library_views.job_detail, name="library-job"),
    path("jobs/<uuid:job_id>/cancel/", library_views.job_cancel, name="library-job-cancel"),
    # Legacy ingestion endpoints
    path("rag/ingest/", views.ingest_document, name="ingest-document"),
    path("rag/jobs/", views.list_jobs, name="job-list"),
    path("rag/jobs/<uuid:job_id>/", views.job_detail, name="job-detail"),
    path("rag/health/", views.ingestion_health, name="ingestion-health"),
]
