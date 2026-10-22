from django.db import migrations, models


def backfill_anchor(apps, schema_editor):
    """Triples of a part-anchored run get that part (it was only in the job's params)."""
    CandidateTriple = apps.get_model("context_graph", "CandidateTriple")
    Job = apps.get_model("ingestion", "Job")
    for job_id, params in Job.objects.filter(triples__isnull=False).distinct().values_list("id", "params"):
        anchor = (params or {}).get("anchor_id") or ""
        if anchor:
            CandidateTriple.objects.filter(job_id=job_id).update(anchor_id=anchor)


class Migration(migrations.Migration):

    dependencies = [
        ("context_graph", "0010_prompt_versions"),
        ("ingestion", "0006_document_languages"),
    ]

    operations = [
        migrations.AddField(model_name="candidatetriple", name="anchor_id",
                            field=models.CharField(blank=True, db_index=True, default="", max_length=512)),
        migrations.AddField(model_name="candidatetriple", name="moved_from",
                            field=models.CharField(blank=True, default="", max_length=512)),
        migrations.AddField(model_name="candidatetriple", name="rel_props",
                            field=models.JSONField(blank=True, default=dict)),
        migrations.RunPython(backfill_anchor, migrations.RunPython.noop),
    ]
