from django.urls import path

from . import views

urlpatterns = [
    path("summary/", views.summary, name="historian-summary"),
    path("samples/", views.samples, name="historian-samples"),
    path("pipeline/", views.pipeline, name="historian-pipeline"),
]
