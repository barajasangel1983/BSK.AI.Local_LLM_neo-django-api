from django.urls import path

from . import views

urlpatterns = [
    path("signals/", views.signals, name="spc-signals"),
    path("tools/", views.tools, name="spc-tools"),
    path("tools/<str:tool_id>/", views.tool, name="spc-tool"),
    path("tools/<str:tool_id>/baseline/", views.baseline, name="spc-baseline"),
    path("tools/<str:tool_id>/limits/", views.limits, name="spc-limits"),
    path("tools/<str:tool_id>/detail/", views.detail, name="spc-detail"),
    path("screening/", views.screening, name="spc-screening"),
    path("summary/", views.summary, name="spc-summary"),
]
