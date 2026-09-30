from django.contrib import admin

from .models import IngestionJob


@admin.register(IngestionJob)
class IngestionJobAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "source_filename",
        "asset_id",
        "document_revision",
        "status",
        "chunk_count",
        "config_version",
        "created_at",
        "completed_at",
    )
    list_filter = ("status", "config_version", "created_at")
    search_fields = ("source_filename", "source_sha256", "asset_id")
    readonly_fields = ("id", "created_at", "completed_at")
