"""Test runner that keeps tests independent of this machine's .env.

With CHROMA_HOST set in .env, code would talk to the real Chroma server; tests
force embedded mode so they only touch the temp dirs they configure.

HISTORIAN_SOURCE in .env says which historian the running Studio serves. Tests start
from the default ("march") and set "live" themselves where they mean it.
"""

from django.test.runner import DiscoverRunner
from django.test.utils import override_settings


class IsolatedTestRunner(DiscoverRunner):
    def setup_test_environment(self, **kwargs):
        super().setup_test_environment(**kwargs)
        self._chroma_override = override_settings(CHROMA_HOST="", HISTORIAN_SOURCE="march")
        self._chroma_override.enable()

    def teardown_test_environment(self, **kwargs):
        self._chroma_override.disable()
        super().teardown_test_environment(**kwargs)
