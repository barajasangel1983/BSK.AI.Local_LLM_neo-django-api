from django.urls import path

from . import views

urlpatterns = [
    path("signals/", views.signals, name="spc-signals"),
    path("tools/", views.tools, name="spc-tools"),
    path("tools/order/", views.order, name="spc-order"),            # before tools/<id>/
    path("tools/<str:tool_id>/", views.tool, name="spc-tool"),
    path("tools/<str:tool_id>/baseline/", views.baseline, name="spc-baseline"),
    path("baseline/preview/", views.baseline_preview, name="spc-baseline-preview"),
    path("tools/<str:tool_id>/limits/", views.limits, name="spc-limits"),
    path("tools/<str:tool_id>/detail/", views.detail, name="spc-detail"),
    path("tools/<str:tool_id>/llm/", views.llm_packet, name="spc-llm-packet"),
    path("tools/<str:tool_id>/llm/analysis/", views.llm_analysis, name="spc-llm-analysis"),
    path("screening/", views.screening, name="spc-screening"),
    path("summary/", views.summary, name="spc-summary"),
    path("events/", views.events, name="spc-events"),
    path("events/<str:event_id>/acknowledge/", views.acknowledge, name="spc-acknowledge"),
]
