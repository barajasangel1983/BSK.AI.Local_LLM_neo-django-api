"""Test runner that keeps tests off the live Chroma server.

With CHROMA_HOST set in .env, code would talk to the real Chroma server; tests
force embedded mode so they only touch the temp dirs they configure.
"""

from django.test.runner import DiscoverRunner
from django.test.utils import override_settings


class IsolatedTestRunner(DiscoverRunner):
    def setup_test_environment(self, **kwargs):
        super().setup_test_environment(**kwargs)
        self._chroma_override = override_settings(CHROMA_HOST="")
        self._chroma_override.enable()

    def teardown_test_environment(self, **kwargs):
        self._chroma_override.disable()
        super().teardown_test_environment(**kwargs)
