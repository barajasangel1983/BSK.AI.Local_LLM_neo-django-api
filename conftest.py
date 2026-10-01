"""pytest bootstrap: configure Django so ingestion/tests can import settings-dependent modules."""

import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "neo_llm_api.settings")
django.setup()
