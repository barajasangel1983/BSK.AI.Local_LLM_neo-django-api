# chat/urls.py

# URL patterns for the chat app.
# These are all mounted under /api/ in the project-level urls.py, so:
#
# - /api/ping/
# - /api/chat/
# - /api/conversations/
# - /api/conversations/<uuid>/
# - /api/models/


from django.urls import path
from . import views

urlpatterns = [
    # Simple health check
    path("ping/", views.ping, name="ping"),

    # Main chat endpoint (create/continue a conversation)
    path("chat/", views.chat_view, name="chat"),

    # Conversation list and detail
    path("conversations/", views.list_conversations, name="conversation-list"),
    path("conversations/<uuid:pk>/", views.conversation_detail, name="conversation-detail"),

    # Static list of models/backends
    path("models/", views.list_models, name="models-list"),
    path("rag/docs/", views.rag_docs, name="rag-docs"),
    path("rag/docs/<str:name>/", views.rag_delete_doc, name="rag-delete-doc"),
    path("rag/query/", views.rag_query, name="rag-query"),
    path("rag/upload/", views.rag_upload, name="rag-upload"),

    # Health monitoring
    path("health/status/", views.health_status, name="health-status"),
    path("health/check/<str:endpoint_id>/", views.health_check_one, name="health-check-one"),
    path("health/restart/<str:endpoint_id>/", views.health_restart, name="health-restart"),
]
