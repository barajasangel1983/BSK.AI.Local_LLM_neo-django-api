"""Back up the Context Studio's data, or test a backup's restore (P25a): `manage.py backup [--keep 7]`,
`manage.py backup --restore-test <folder>`. See ingestion/backup.py for what a backup holds."""

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from ingestion import backup


class Command(BaseCommand):
    help = "Back up SQLite, data/, Chroma and Neo4j to ~/backups/<UTC time>/ (or test a backup's restore)."

    def add_arguments(self, parser):
        parser.add_argument("--dest", help="the backups folder (default ~/backups)")
        parser.add_argument("--keep", type=int, default=7, help="newest runs kept (hand copies are never removed)")
        parser.add_argument("--restore-test", metavar="FOLDER",
                            help="load this backup into throwaway containers and compare its counts with the manifest")

    def handle(self, *args, **opts):
        log = lambda m: self.stdout.write(m)        # noqa: E731
        try:
            if opts["restore_test"]:
                problems = backup.restore_test(Path(opts["restore_test"]).expanduser(), log=log)
                if problems:
                    raise CommandError("restore test FAILED:\n" + "\n".join(problems))
                self.stdout.write(self.style.SUCCESS("restore test passed"))
                return
            folder = backup.run(Path(opts["dest"]).expanduser() if opts["dest"] else None, keep=opts["keep"], log=log)
        except backup.BackupError as exc:
            raise CommandError(str(exc))
        self.stdout.write(self.style.SUCCESS(f"backup written: {folder}"))
