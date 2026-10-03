from django.db import models


class ModelCall(models.Model):
    """One call from the backend to a model or AI service (Analytics page).

    Generation calls record prompt / completion tokens (from the provider's
    usage, or estimated when it reports none); embedding, reranking and
    parsing calls record requests, latency and result only.
    """

    class Status(models.TextChoices):
        OK = "ok", "OK"
        ERROR = "error", "Error"
        TIMEOUT = "timeout", "Timeout"

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    purpose = models.CharField(max_length=20, db_index=True)   # chat, compare, regenerate, title, extract, embed, rerank, parse
    model_id = models.CharField(max_length=128)                 # hub model id or service, e.g. dgx-qwen38-27b-fp8, docling
    conversation_id = models.UUIDField(null=True, blank=True)
    prompt_tokens = models.PositiveIntegerField(null=True, blank=True)
    completion_tokens = models.PositiveIntegerField(null=True, blank=True)
    tokens_estimated = models.BooleanField(default=False)
    latency_ms = models.PositiveIntegerField(default=0)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.OK)
    error = models.CharField(max_length=255, blank=True, default="")

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.created_at:%Y-%m-%d %H:%M} {self.purpose} {self.model_id} {self.status}"
