from django.urls import path

from . import views

urlpatterns = [
    path("assets/", views.assets, name="monitor-assets"),
    path("overview/", views.overview, name="monitor-overview"),
    path("posts/", views.posts, name="monitor-posts"),
    path("settings/", views.settings_view, name="monitor-settings"),
]
