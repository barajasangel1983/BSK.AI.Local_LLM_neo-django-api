from django.urls import path

from . import views

urlpatterns = [
    path("", views.manifest, name="twin-manifest"),
    path("generate/", views.generate, name="twin-generate"),
    path("live/", views.live, name="twin-live"),
]
