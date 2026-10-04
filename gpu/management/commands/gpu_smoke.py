"""Live check of the BSK GPU path, through the Hub's own client (lock → activate → call → touch).

    python manage.py gpu_smoke                 # parse check, then VLM check
    python manage.py gpu_smoke --parse         # Docling only
    python manage.py gpu_smoke --vlm           # VLM only
    python manage.py gpu_smoke --cold          # switch the GPU to idle first (real cold start)
    python manage.py gpu_smoke --file a.pdf    # parse this PDF instead of the built-in page

Refuses to run while library jobs are queued or running (a switch would interrupt them).
Runbook: frontend repo, claude/P9_vlm_integration_plan.md, section 3.
"""

import io
import time

import requests
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from gpu import orchestrator as gpu
from gpu import images, vlm

SMOKE_TEXT = "BSK SMOKE 4721"


def _test_image():
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1000, 400), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=96)
    except TypeError:       # older Pillow
        font = ImageFont.load_default()
    draw.rectangle((20, 20, 980, 380), outline="black", width=6)
    draw.text((70, 140), SMOKE_TEXT, fill="black", font=font)
    return img


class Command(BaseCommand):
    help = "Check Docling and the VLM on BSK through the GPU orchestrator."

    def add_arguments(self, parser):
        parser.add_argument("--parse", action="store_true", help="Run the Docling parse check.")
        parser.add_argument("--vlm", action="store_true", help="Run the VLM check.")
        parser.add_argument("--cold", action="store_true", help="Switch the GPU to idle first.")
        parser.add_argument("--file", help="PDF to parse (default: a generated one-page PDF).")
        parser.add_argument("--force", action="store_true", help="Run even when library jobs are active.")

    def _status(self, label: str) -> dict:
        body = gpu.status()
        self.stdout.write(f"  status {label}: active={body.get('active')} health={body.get('health')} "
                          f"idle_reset_at={body.get('idle_reset_at')} last_activation={body.get('last_activation')}")
        return body

    def handle(self, *args, **opts):
        run_parse = opts["parse"] or not opts["vlm"]
        run_vlm = opts["vlm"] or not opts["parse"]
        if not gpu.enabled():
            raise CommandError("GPU_ORCHESTRATOR_ENABLED is off: set it in .env first (runbook step 3).")

        from ingestion.models import Job
        active = Job.objects.filter(status__in=[Job.Status.QUEUED, Job.Status.RUNNING]).count()
        if active and not opts["force"]:
            raise CommandError(f"{active} library job(s) are queued or running; a GPU switch would interrupt them.")

        results: list[tuple[str, bool, str]] = []
        self._status("before")
        if opts["cold"]:
            with gpu._lock(5):
                gpu.activate("idle")
            self._status("after idle")

        if run_parse:
            results.append(self._check("parse", lambda: self._parse(opts.get("file"))))
        if run_vlm:
            results.append(self._check("vlm", self._vlm))
        self._status("after")

        self.stdout.write("")
        for name, ok, detail in results:
            self.stdout.write((self.style.SUCCESS if ok else self.style.ERROR)(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}"))
        if not all(ok for _, ok, _ in results):
            raise CommandError("gpu_smoke failed")

    def _check(self, name, fn) -> tuple[str, bool, str]:
        self.stdout.write(f"{name}:")
        start = time.monotonic()
        try:
            detail = fn()
            return name, True, f"{detail} ({time.monotonic() - start:.1f} s)"
        except Exception as exc:        # report every failure in the summary
            return name, False, f"{type(exc).__name__}: {exc}"

    def _parse(self, path: str | None) -> str:
        from ingestion.docling_client import convert_file

        if path:
            data, name = open(path, "rb").read(), path.rsplit("/", 1)[-1]
        else:
            buf = io.BytesIO()
            _test_image().save(buf, format="PDF")
            data, name = buf.getvalue(), "gpu-smoke.pdf"
        before = self._status("before parse")
        result = convert_file(data, name, wait=30)
        after = self._status("after parse")
        text = (result.get("document") or {}).get("md_content") or ""
        if result.get("status") != "success":
            raise RuntimeError(f"Docling status {result.get('status')}: {result.get('errors')}")
        if not after.get("idle_reset_at") or after.get("idle_reset_at") == before.get("idle_reset_at"):
            raise RuntimeError("the final touch did not move idle_reset_at")
        if after.get("active") != "docling":
            raise RuntimeError(f"active is {after.get('active')} after the parse")
        return f"{name}: status success, {len(text)} characters of Markdown"

    def _vlm(self) -> str:
        buf = io.BytesIO()
        _test_image().save(buf, format="PNG")
        jpeg, width, height = images.to_jpeg(buf.getvalue())
        start = time.monotonic()
        with gpu.use("vl", wait=30):
            self.stdout.write(f"  activate vl: {time.monotonic() - start:.1f} s")
            models = requests.get(settings.VLM_URL.rstrip("/") + "/models", timeout=10)
            models.raise_for_status()
        before = self._status("vl active")
        start = time.monotonic()
        reply = vlm.complete(jpeg, "Read the text in this image. Answer with the text only.", max_tokens=50,
                             purpose="gpu-smoke", wait=30)
        seconds = time.monotonic() - start
        after = self._status("after vlm")
        self.stdout.write(f"  reply: {reply.strip()[:120]!r}")
        if SMOKE_TEXT.replace(" ", "") not in reply.upper().replace(" ", ""):
            raise RuntimeError(f"the reply does not contain {SMOKE_TEXT!r}: {reply.strip()[:120]!r}")
        if not after.get("idle_reset_at") or after.get("idle_reset_at") == before.get("idle_reset_at"):
            raise RuntimeError("the final touch did not move idle_reset_at")
        return f"read {SMOKE_TEXT!r} from a {width}x{height} image in {seconds:.1f} s"
