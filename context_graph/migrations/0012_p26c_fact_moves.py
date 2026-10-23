import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("context_graph", "0011_p26b_anchor_move_rel_props"),
    ]

    operations = [
        migrations.CreateModel(
            name="FactMoveBatch",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("target_id", models.CharField(db_index=True, max_length=512)),
                ("status", models.CharField(choices=[("applied", "Applied"), ("undone", "Undone")], default="applied", max_length=10)),
                ("choices", models.JSONField(blank=True, default=dict)),
                ("summary", models.JSONField(blank=True, default=dict)),
                ("before", models.JSONField(blank=True, default=dict)),
                ("created_links", models.JSONField(blank=True, default=list)),
                ("author", models.CharField(blank=True, default="", max_length=150)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("undone_at", models.DateTimeField(blank=True, null=True)),
            ],
            options={"ordering": ["-created_at"]},
        ),
        migrations.CreateModel(
            name="FactMove",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("kind", models.CharField(choices=[("move", "Move"), ("merge", "Merge"), ("model", "Model"), ("stay", "Stay")], max_length=10)),
                ("label", models.CharField(blank=True, default="", max_length=40)),
                ("old_id", models.CharField(db_index=True, max_length=512)),
                ("new_id", models.CharField(blank=True, db_index=True, default="", max_length=512)),
                ("owner_before", models.CharField(blank=True, default="", max_length=512)),
                ("owner_after", models.CharField(blank=True, default="", max_length=512)),
                ("batch", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="moves", to="context_graph.factmovebatch")),
            ],
        ),
    ]
