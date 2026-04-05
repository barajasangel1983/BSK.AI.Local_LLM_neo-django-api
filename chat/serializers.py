# chat/serializers.py

"""
Serializers convert Django model instances <-> JSON.

We use three serializers:

1. MessageSerializer
- For individual chat messages.

2. ConversationSummarySerializer
- For listing conversations in the sidebar (id, title, last message, etc.).

3. ConversationDetailSerializer
- For returning a full conversation with all its messages.
"""

from rest_framework import serializers
from .models import Conversation, Message


class MessageSerializer(serializers.ModelSerializer):
    # Represents a single message in JSON.
    #
    # Example JSON:
    # {
    # "id": "uuid",
    # "role": "user",
    # "content": "Hello",
    # "created_at": "2025-03-28T21:00:00Z"
    # }

    class Meta:
        model = Message
        fields = ["id", "role", "content", "created_at"]


class ConversationSummarySerializer(serializers.ModelSerializer):
    # Used for the conversation list in the sidebar.
    # We add two extra read-only fields that are NOT on the model:
    # - last_message_at: timestamp of the most recent message
    # - last_message_preview: short snippet of the last message

    last_message_at = serializers.SerializerMethodField()
    last_message_preview = serializers.SerializerMethodField()

    class Meta:
        model = Conversation
        fields = [
            "id",
            "title",
            "created_at",
            "updated_at",
            "last_message_at",
            "last_message_preview",
        ]

    def get_last_message_at(self, obj):
        """Return the created_at of the last message, or None if none.

        If the queryset was annotated with last_message_at, prefer that;
        otherwise, fall back to the most recent message or updated_at.
        """
        annotated = getattr(obj, "last_message_at", None)
        if annotated is not None:
            return annotated
        last_msg = obj.messages.order_by("-created_at").first()
        return last_msg.created_at if last_msg else obj.updated_at

    def get_last_message_preview(self, obj):
        """Return a short preview of the last message (~80 chars).

        If there are no messages, return an empty string.
        """
        last_msg = obj.messages.order_by("-created_at").first()
        if not last_msg:
            return ""
        text = last_msg.content or ""
        return text[:80] + ("…" if len(text) > 80 else "")


class ConversationDetailSerializer(serializers.ModelSerializer):
    # Used when we want the full conversation with all messages.
    # We nest MessageSerializer to include a messages[] array inside.
    # `many=True` because a conversation has many messages.
    # `read_only=True` because we only use this to *return* messages,
    # not to create them directly via this serializer.

    messages = MessageSerializer(many=True, read_only=True)

    class Meta:
        model = Conversation
        fields = ["id", "title", "created_at", "updated_at", "messages"]
