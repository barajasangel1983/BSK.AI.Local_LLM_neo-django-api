"""Context Graph routes, mounted under /api/graph/.

Canonical IDs contain ':' and '/', so ID segments use the `path` converter.
"""

from django.urls import path

from . import views

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
]
