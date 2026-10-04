from django.urls import path

from . import views

urlpatterns = [
    path("assets/", views.assets, name="context-assets"),
    path("resolve/", views.resolve, name="context-resolve"),
    path("entities/<path:entity_id>/sources/", views.entity_sources, name="context-entity-sources"),
    path("entities/<path:entity_id>/", views.entity, name="context-entity"),
    path("state/<path:asset_id>/", views.state, name="context-state"),
    path("search/", views.search, name="context-search"),
    path("assemble/", views.assemble, name="context-assemble"),
    path("ask/", views.ask, name="context-ask"),
]
