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
    created_at = models.DateTimeField(auto_now_add=True)

    # Default ordering: oldest → newest.
    class Meta:
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"{self.role}: {self.content[:50]}"
