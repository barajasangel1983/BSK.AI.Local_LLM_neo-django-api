"""Title conversations still named "New Conversation" from their first exchange.

    python manage.py backfill_conversation_titles [--dry-run]

Uses the same generator as new chats (DGX model, falling back to the start of
the first user message). Conversations without a user message are skipped;
renamed conversations are never touched.
"""

from __future__ import annotations

from django.core.management.base import BaseCommand

from chat.models import Conversation
from chat.titles import DEFAULT_TITLE, fallback_title, generate_title


class Command(BaseCommand):
    help = 'Generate titles for conversations still titled "New Conversation".'

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Show titles without saving them.")

    def handle(self, *args, dry_run=False, **options):
        pending = Conversation.objects.filter(title__in=["", DEFAULT_TITLE]).order_by("created_at")
        titled = skipped = 0
        for conv in pending:
            messages = list(conv.messages.order_by("created_at").values_list("role", "content"))
            user_msg = next((content for role, content in messages if role == "user"), None)
            if not user_msg:
                skipped += 1
                self.stdout.write(f"  skip {conv.id}: no user message")
                continue
            reply = next((content for role, content in messages if role == "assistant"), "")
            title, source = generate_title(user_msg, reply) if reply else (fallback_title(user_msg), "fallback")
            self.stdout.write(f"  {conv.id}: {title!r} ({source})")
            if not dry_run:
                conv.title = title
                conv.save(update_fields=["title"])
            titled += 1

        verb = "would title" if dry_run else "titled"
        self.stdout.write(f"{verb} {titled} conversation(s), skipped {skipped}.")
