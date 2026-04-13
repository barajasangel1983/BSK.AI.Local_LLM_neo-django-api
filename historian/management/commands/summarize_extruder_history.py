from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

from django.core.management.base import BaseCommand
from django.utils import timezone

from historian.summarizer import ShiftStats, compute_shift_stats, format_shift_summary

# RAG ingestion from GraphRAG repo
from rag.ingestion import ingest_chunks  # type: ignore


UTC = timezone.UTC


class Command(BaseCommand):
    help = "Summarize extruder historian data by shift and ingest summaries into Chroma RAG."

    def add_arguments(self, parser):
        parser.add_argument(
            "--month",
            type=str,
            help="Target month in YYYY-MM format (e.g. 2026-03). If omitted, you must pass --start and --end.",
        )
        parser.add_argument(
            "--start",
            type=str,
            help="Start timestamp (ISO 8601, e.g. 2026-03-01T00:00:00Z).",
        )
        parser.add_argument(
            "--end",
            type=str,
            help="End timestamp (ISO 8601, e.g. 2026-03-31T23:59:00Z).",
        )
        parser.add_argument(
            "--extruder-id",
            type=str,
            default="EXTR01",
            help="Extruder ID to summarize (default: EXTR01).",
        )
        parser.add_argument(
            "--owner-user-id",
            type=str,
            default="admin",
            help="Owner user id to store in RAG metadata (default: admin).",
        )
        parser.add_argument(
            "--visibility",
            type=str,
            default="public",
            help="Visibility metadata for RAG (default: public).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Compute and show summaries but do not ingest into Chroma.",
        )

    def _parse_time_range(self, month: str | None, start: str | None, end: str | None):
        if month:
            year, mon = map(int, month.split("-"))
            start_dt = datetime(year, mon, 1, 0, 0, tzinfo=UTC)
            if mon == 12:
                end_dt = datetime(year + 1, 1, 1, 0, 0, tzinfo=UTC) - timedelta(minutes=1)
            else:
                end_dt = datetime(year, mon + 1, 1, 0, 0, tzinfo=UTC) - timedelta(minutes=1)
            return start_dt, end_dt

        if not start or not end:
            raise SystemExit("Either --month or both --start and --end must be provided.")

        def parse_ts(s: str) -> datetime:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt

        return parse_ts(start), parse_ts(end)

    def handle(self, *args, **options):
        month = options["month"]
        start_str = options["start"]
        end_str = options["end"]
        extruder_id = options["extruder_id"]
        owner_user_id = options["owner_user_id"]
        visibility = options["visibility"]
        dry_run: bool = options["dry_run"]

        start_dt, end_dt = self._parse_time_range(month, start_str, end_str)
        self.stdout.write(
            f"Summarizing extruder history for {extruder_id} from {start_dt} to {end_dt}"
        )

        stats_list: List[ShiftStats] = compute_shift_stats(start_dt, end_dt, extruder_id)
        if not stats_list:
            self.stdout.write("No samples found in this range.")
            return

        self.stdout.write(f"Computed {len(stats_list)} shift-level aggregates.")

        documents: list[str] = []
        metadatas: list[dict] = []

        for stats in stats_list:
            text = format_shift_summary(stats)
            documents.append(text)
            metadatas.append(
                {
                    "source": "plc_historian",
                    "extruder_id": stats.extruder_id,
                    "recipe_id": stats.recipe_id,
                    "date": stats.date.isoformat(),
                    "shift_code": stats.shift_code,
                    "time_window": f"{stats.start_ts.isoformat()}/{stats.end_ts.isoformat()}",
                    "owner_user_id": owner_user_id,
                    "visibility": visibility,
                }
            )

        if dry_run:
            self.stdout.write("Dry run: showing first 2 summaries")
            for text in documents[:2]:
                self.stdout.write("---")
                self.stdout.write(text)
            return

        # Ingest directly into Chroma using the shared bsk_rag collection
        self.stdout.write(
            f"Ingesting {len(documents)} shift summaries into Chroma collection 'bsk_rag'..."
        )

        # We re-use ingest_chunks' logic by temporarily patching its Chunk handling via a simple add call.
        # For v0 we call the Chroma client directly from here to avoid changing ingest_chunks.
        from chromadb import PersistentClient
        from chromadb.config import Settings
        from rag.config import CHROMA_DIR

        client = PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
        collection = client.get_or_create_collection(name="bsk_rag")

        ids = [
            f"plc_historian::{s.extruder_id}::{s.date.isoformat()}::{s.shift_code}::{s.recipe_id}"
            for s in stats_list
        ]
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

        self.stdout.write(self.style.SUCCESS("Ingestion complete."))
