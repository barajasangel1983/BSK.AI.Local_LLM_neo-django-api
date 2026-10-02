"""Process document-library jobs (parse, Generate embeddings).

    python manage.py library_worker            # run forever
    python manage.py library_worker --once     # drain the queue, then exit

Runs outside runserver so code reloads can't kill long jobs. One job at a
time (Docling and the DGX are shared services). Running jobs whose heartbeat
stopped (worker killed) are re-queued after LIBRARY_JOB_STALE_SECONDS.
"""

import logging
import signal
import time

from django.core.management.base import BaseCommand
from django.db import close_old_connections

from ingestion import library

logger = logging.getLogger("chat")


class Command(BaseCommand):
    help = "Run the document-library worker."

    def add_arguments(self, parser):
        parser.add_argument("--once", action="store_true", help="Exit when the queue is empty.")
        parser.add_argument("--poll", type=float, default=2.0, help="Seconds between queue checks.")

    def handle(self, *args, once=False, poll=2.0, **options):
        self._stop = False
        signal.signal(signal.SIGTERM, self._request_stop)
        signal.signal(signal.SIGINT, self._request_stop)
        self.stdout.write("library worker started")
        while not self._stop:
            close_old_connections()
            requeued = library.requeue_stale()
            if requeued:
                self.stdout.write(f"re-queued {requeued} stale job(s)")
            job = library.claim_next()
            if job is None:
                if once:
                    break
                time.sleep(poll)
                continue
            self.stdout.write(f"running {job.kind} for {job.document.doc_key} ({job.id})")
            library.run_job(job)
            job.refresh_from_db()
            self.stdout.write(f"  -> {job.status}{': ' + job.error.splitlines()[0][:200] if job.error else ''}")
        self.stdout.write("library worker stopped")

    def _request_stop(self, *_):
        # Finish the current job, then exit.
        self._stop = True
