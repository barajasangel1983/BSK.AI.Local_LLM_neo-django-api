"""Run the Context MCP server (see context_service/mcp_server.py)."""

import asyncio

from django.core.management.base import BaseCommand

from context_service import mcp_server


class Command(BaseCommand):
    help = "Run the Context MCP server (Streamable HTTP, token sign-in, follows the Studio's exposure setting)."

    def handle(self, *args, **options):
        self.stdout.write("context mcp server started")
        asyncio.run(mcp_server.serve(out=lambda line: self.stdout.write(line)))
        self.stdout.write("context mcp server stopped")
