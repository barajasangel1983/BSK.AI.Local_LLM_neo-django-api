from django.urls import path

from . import views

urlpatterns = [
    path("status/", views.plc_status, name="plc-status"),
    path("command/", views.plc_command, name="plc-command"),
    path("commands/", views.plc_commands, name="plc-commands"),
    path("opcua/", views.plc_opcua, name="plc-opcua"),
]
