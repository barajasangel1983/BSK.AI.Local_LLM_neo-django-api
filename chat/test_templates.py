"""Shared prompt templates (P15b)."""

from django.test import TestCase
from rest_framework.test import APIClient

from chat.models import PromptTemplate


class TemplateTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_starters_create_delete_and_import(self):
        names = [t["name"] for t in self.client.get("/api/chat/templates/").json()["templates"]]
        self.assertEqual(sorted(names), ["Code Review", "Debug Error", "Explain Like I'm 5", "Summarize Document"])
        r = self.client.post("/api/chat/templates/", {"name": "Die plug", "content": "What to do on DIE_PLUG at [time]?",
                                                     "category": "Plant"}, format="json")
        self.assertEqual((r.status_code, r.json()["category"]), (201, "Plant"))
        self.assertEqual(self.client.post("/api/chat/templates/", {"name": "", "content": "x"}, format="json").status_code, 400)

        browser = [{"name": "Die plug", "content": "What to do on DIE_PLUG at [time]?", "category": "Plant"},   # already there
                   {"name": "Shift handover", "content": "Summarize the shift for the next team.", "category": "Plant"},
                   {"name": "", "content": "no name"}, "not a template"]
        r = self.client.post("/api/chat/templates/import/", {"templates": browser}, format="json").json()
        self.assertEqual((r["added"], len(r["templates"])), (1, 6))
        self.assertEqual(self.client.post("/api/chat/templates/import/", {"templates": browser}, format="json").json()["added"], 0)

        pk = PromptTemplate.objects.get(name="Shift handover").pk
        self.assertEqual(self.client.delete(f"/api/chat/templates/{pk}/").status_code, 204)
        self.assertEqual(self.client.delete(f"/api/chat/templates/{pk}/").status_code, 404)
