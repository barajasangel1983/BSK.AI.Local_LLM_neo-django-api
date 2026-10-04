# chat/models.py
import uuid
from django.db import models
from django.contrib.auth.models import User


class Conversation(models.Model):
    # UUID as primary key so we don't rely on integer IDs.
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Owner of the conversation (per-user scoping; currently uses a stub 'admin' user
    # until real auth is wired and request.user is available).
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="conversations", null=True, blank=True)

    # Optional human-readable title (can be empty at first).
    title = models.CharField(max_length=255, blank=True)

    # Asset the conversation is about (canonical id, e.g. bsk:asset:EXTR01); empty = not scoped.
    asset_id = models.CharField(max_length=512, blank=True, default="")

    # Model that was used for this conversation's last turn (optional)
    model_id = models.CharField(max_length=64, blank=True)

    # Auto timestamps: set when created, and each time updated.
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        # What shows in admin / shell when you print a Conversation.
        return self.title or f"Conversation {self.id}"


class Message(models.Model):
    # Only three allowed roles for now.
    ROLE_CHOICES = [
        ("user", "User"),
        ("assistant", "Assistant"),
        ("system", "System"),
    ]

    # UUID primary key for messages too.
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Many messages belong to one conversation.
    # related_name="messages" lets us do: conversation.messages.all()
    conversation = models.ForeignKey(
        Conversation, related_name="messages", on_delete=models.CASCADE  # if conversation is deleted, delete its messages
    )
    role = models.CharField(max_length=16, choices=ROLE_CHOICES)
    content = models.TextField()
    # RAG citations for assistant messages: [{source, asset_id, section_path,
    # page_start, page_end, snippet, score, vector_score, rerank_score}, ...]
    sources = models.JSONField(default=list, blank=True)
    # File the user attached to this message (image or PDF); `attachment_page` is the PDF
    # page shown to the vision model (empty = the PDF was read as text).
    attachment = models.ForeignKey("ChatAttachment", null=True, blank=True, on_delete=models.SET_NULL,
                                   related_name="messages")
    attachment_page = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # Default ordering: oldest → newest.
    class Meta:
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"{self.role}: {self.content[:50]}"


class ChatAttachment(models.Model):
    """An image or PDF attached in a chat. It stays with its conversation: it is never
    added to the document library, Chroma or the Context Graph, and is deleted with the
    conversation. Files live under CHAT_FILES_DIR/<id>/ (see chat/attachments.py)."""

    class Kind(models.TextChoices):
        IMAGE = "image", "Image"
        PDF = "pdf", "PDF"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="chat_attachments", null=True, blank=True)
    # Empty until the first message that uses it is sent (uploads come before the message).
    conversation = models.ForeignKey(Conversation, related_name="attachments", on_delete=models.CASCADE,
                                     null=True, blank=True)
    kind = models.CharField(max_length=8, choices=Kind.choices)
    filename = models.CharField(max_length=512)
    size = models.PositiveBigIntegerField(default=0)
    page_count = models.PositiveIntegerField(null=True, blank=True)     # PDFs
    width = models.PositiveIntegerField(null=True, blank=True)          # images, as stored
    height = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"{self.kind} {self.filename}"
