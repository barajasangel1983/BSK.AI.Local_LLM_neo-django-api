from django.urls import path

from . import views

urlpatterns = [
    path("", views.summary, name="alarms-summary"),
    path("codes/", views.codes, name="alarms-codes"),
    path("codes/<str:code>/", views.code, name="alarms-code"),
]
