"""Context Graph routes, mounted under /api/graph/."""

from django.urls import path

from . import views

urlpatterns = [
    path("health/", views.graph_health, name="graph-health"),
]
