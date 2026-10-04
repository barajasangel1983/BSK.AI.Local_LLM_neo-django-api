from django.apps import AppConfig


class ChatConfig(AppConfig):
    name = 'chat'

    def ready(self):
        from . import attachments  # noqa: F401  (registers the file clean-up on delete)
