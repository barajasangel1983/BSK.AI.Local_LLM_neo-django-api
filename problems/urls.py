from django.urls import path

from . import views

urlpatterns = [
    path("tickets/", views.tickets, name="problems-tickets"),
    path("tickets/<int:ticket_id>/", views.ticket, name="problems-ticket"),
    path("tickets/<int:ticket_id>/pictures/<int:picture_id>/", views.picture, name="problems-picture"),
    path("tickets/<int:ticket_id>/report/", views.report, name="problems-report"),
    path("tickets/<int:ticket_id>/check/", views.run_check, name="problems-check"),
    path("tickets/<int:ticket_id>/recommend/", views.recommend, name="problems-recommend"),
    path("tickets/<int:ticket_id>/context/", views.refresh, name="problems-context"),
    path("settings/", views.settings_view, name="problems-settings"),
]
