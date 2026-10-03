"""One Evidence row per existing triple, from the evidence fields it already carries (P8a).

Before P8 a triple kept only its first occurrence (document window or data-file row);
that occurrence becomes its Evidence. The old fields stay on the triple for one release.
"""

from django.db import migrations


def forwards(apps, schema_editor):
    Triple = apps.get_model("context_graph", "CandidateTriple")
    Evidence = apps.get_model("context_graph", "Evidence")
    Link = Triple.evidence.through
    links = []
    for t in Triple.objects.filter(evidence__isnull=True).iterator():
        structured = t.source == "structured"
        e = Evidence.objects.create(
            source_kind="structured" if structured else "text",
            document_id=t.document_id, data_file_id=t.data_file_id,
            page_start=t.page_start, page_end=t.page_end, section_path=t.section_path or [],
            row_number=t.row_number, chunk_index=None if structured else t.chunk_index,
            extractor="column-mapping" if structured else "llm-text",
            model=t.model or "", prompt=f"{t.preset_name} v{t.preset_version}" if t.preset_name else "",
            schema_version=t.schema_version, excerpt=t.evidence_text or "", confidence=t.confidence,
        )
        links.append(Link(candidatetriple_id=t.pk, evidence_id=e.pk))
    Link.objects.bulk_create(links)


def backwards(apps, schema_editor):
    apps.get_model("context_graph", "Evidence").objects.all().delete()


class Migration(migrations.Migration):
    dependencies = [("context_graph", "0004_evidence")]
    operations = [migrations.RunPython(forwards, backwards)]
