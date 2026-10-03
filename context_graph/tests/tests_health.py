"""Context Graph P0: driver, health and health endpoints.

Run with the Django test runner (`manage.py test context_graph`), not pytest.

Unit tests mock the Neo4j driver. The integration test runs only with
GRAPH_INTEGRATION=1 against the Neo4j configured in .env.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, override_settings
from rest_framework.test import APIClient

from context_graph import driver

CONFIGURED = dict(GRAPH_ENABLED=True, NEO4J_URI="bolt://graph:7687", NEO4J_USER="neo4j",
                  NEO4J_PASSWORD="secret", NEO4J_DATABASE="neo4j")


def _fake_driver():
    """Driver whose session answers the two health queries."""
    session = MagicMock()
    components = MagicMock()
    components.single.return_value = {"name": "Neo4j Kernel", "version": "5.26.0", "edition": "community"}
    counts = MagicMock()
    counts.single.return_value = {"n": 42}
    session.run.side_effect = [components, counts]
    fake = MagicMock()
    fake.session.return_value.__enter__.return_value = session
    return fake


class DriverTestCase(SimpleTestCase):
    def setUp(self):
        driver.close_driver()
        self.addCleanup(driver.close_driver)


class HealthTests(DriverTestCase):
    @override_settings(GRAPH_ENABLED=False)
    def test_disabled(self):
        self.assertEqual(driver.health()["status"], "disabled")
        with self.assertRaises(driver.GraphUnavailable):
            driver.get_driver()

    @override_settings(**{**CONFIGURED, "NEO4J_PASSWORD": ""})
    def test_not_configured(self):
        self.assertEqual(driver.health()["status"], "not_configured")

    @override_settings(**CONFIGURED)
    @patch("context_graph.driver.GraphDatabase.driver")
    def test_online(self, mock_factory):
        fake = _fake_driver()
        mock_factory.return_value = fake

        result = driver.health()

        self.assertEqual(result["status"], "online")
        self.assertEqual(result["server"], "Neo4j Kernel 5.26.0 (community)")
        self.assertEqual(result["node_count"], 42)
        mock_factory.assert_called_once_with("bolt://graph:7687", auth=("neo4j", "secret"), connection_timeout=5.0,
                                             notifications_disabled_categories=["UNRECOGNIZED"])
        fake.session.assert_called_with(database="neo4j")

    @override_settings(**CONFIGURED)
    @patch("context_graph.driver.GraphDatabase.driver")
    def test_driver_is_reused(self, mock_factory):
        self.assertIs(driver.get_driver(), driver.get_driver())
        mock_factory.assert_called_once()

    @override_settings(**CONFIGURED)
    @patch("context_graph.driver.GraphDatabase.driver")
    def test_offline_reports_error(self, mock_factory):
        mock_factory.return_value.session.side_effect = OSError("connection refused")
        result = driver.health()
        self.assertEqual(result["status"], "offline")
        self.assertIn("connection refused", result["error"])


class HealthEndpointTests(SimpleTestCase):
    def setUp(self):
        self.client = APIClient()

    @patch("context_graph.views.driver.health", return_value={"status": "online", "node_count": 0})
    def test_graph_health_online(self, mock_health):
        r = self.client.get("/api/graph/health/")
        self.assertEqual((r.status_code, r.data["status"]), (200, "online"))

    @patch("context_graph.views.driver.health", return_value={"status": "disabled"})
    def test_graph_health_unavailable_is_503(self, mock_health):
        self.assertEqual(self.client.get("/api/graph/health/").status_code, 503)

    @patch("context_graph.driver.health", return_value={"status": "online", "latency_ms": 12})
    def test_model_health_page_entry(self, mock_health):
        r = self.client.post("/api/health/check/context-graph/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual((r.data["status"], r.data["latency"], r.data["model"]), ("online", 12, "graph"))

    @patch("context_graph.driver.health", return_value={"status": "not_configured"})
    def test_model_health_entry_offline_when_not_configured(self, mock_health):
        self.assertEqual(self.client.post("/api/health/check/context-graph/").data["status"], "offline")


@unittest.skipUnless(os.getenv("GRAPH_INTEGRATION") == "1", "set GRAPH_INTEGRATION=1 to test against the real Neo4j")
class Neo4jIntegrationTests(DriverTestCase):
    def test_real_instance_answers(self):
        from django.conf import settings
        with override_settings(GRAPH_ENABLED=True):
            result = driver.health()
            self.assertEqual(result["status"], "online", result)
            with driver.session() as s:
                self.assertEqual(s.run("RETURN 1 AS one").single()["one"], 1)
        self.assertTrue(settings.NEO4J_PASSWORD)
