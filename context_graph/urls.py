"""Context Graph routes, mounted under /api/graph/.

Canonical IDs contain ':' and '/', so ID segments use the `path` converter.
"""

from django.urls import path

from . import extraction_views, import_views, views

urlpatterns = [
    path("health/", views.graph_health, name="graph-health"),
    path("schema/", views.graph_schema, name="graph-schema"),
    path("schema/validate/", views.schema_validate, name="graph-schema-validate"),
    path("schema/versions/", views.schema_versions, name="graph-schema-versions"),
    path("schema/versions/<int:version>/activate/", views.schema_activate, name="graph-schema-activate"),
    path("schema/export/", views.schema_export, name="graph-schema-export"),
    path("assets/", views.asset_list, name="graph-assets"),
    path("assets/<path:asset_id>/context/", views.asset_context, name="graph-asset-context"),
    path("nodes/", views.node_list, name="graph-nodes"),
    path("nodes/<path:node_id>/", views.node_detail, name="graph-node"),
    path("alarms/<path:alarm_id>/procedures/", views.alarm_procedures, name="graph-alarm-procedures"),
    path("data/", views.graph_data, name="graph-data"),
    path("lab/", views.lab_summary, name="graph-lab"),
    # Triple extraction (GraphLab "Generate triples") and review
    path("presets/", extraction_views.presets, name="graph-presets"),
    path("presets/<int:preset_id>/", extraction_views.preset_detail, name="graph-preset"),
    path("extract/", extraction_views.extract, name="graph-extract"),
    path("extract/config/", extraction_views.extract_config, name="graph-extract-config"),
    path("triples/", extraction_views.triple_list, name="graph-triples"),
    path("triples/approve/", extraction_views.triples_approve, name="graph-triples-approve"),
    path("triples/reject/", extraction_views.triples_reject, name="graph-triples-reject"),
    path("triples/delete/", extraction_views.triples_delete, name="graph-triples-delete"),
    path("triples/<int:triple_id>/", extraction_views.triple_detail, name="graph-triple"),
    path("triples/<int:triple_id>/promote/", extraction_views.triple_promote, name="graph-triple-promote"),
    # Structured data import (GraphLab Import): CSV / Excel -> column mapping -> staged triples
    path("datafiles/", import_views.datafiles, name="graph-datafiles"),
    path("datafiles/<uuid:file_id>/", import_views.datafile_detail, name="graph-datafile"),
    path("datafiles/<uuid:file_id>/preview/", import_views.datafile_preview, name="graph-datafile-preview"),
    path("datafiles/<uuid:file_id>/stage/", import_views.datafile_stage, name="graph-datafile-stage"),
    path("mappings/", import_views.mappings, name="graph-mappings"),
    path("mappings/<int:mapping_id>/", import_views.mapping_detail, name="graph-mapping"),
]
