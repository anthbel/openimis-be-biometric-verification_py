"""
template_key comes from the BIOMETRIC_TEMPLATE_KEY environment variable when
neither the module configuration nor settings.BIOMETRIC gives one; a blank
value is no key. The key encrypts templates and never appears in a query
result, a log line or the stored module configuration.
"""

import json
import os
from unittest.mock import patch

from cryptography.fernet import Fernet
from django.test import SimpleTestCase, TestCase, override_settings

from biometric import apps as biometric_apps
from biometric import crypto
from biometric.apps import BiometricConfig
from biometric.providers.fake import FakeEmbeddingProvider
from biometric.services import enrol
from biometric.tests.test_admin_schema import CRITERIA_QUERY, RETENTION_QUERY, _execute, _User
from biometric.tests.test_config_refresh import FAKE_FACE, ConfigSnapshotMixin

ENV = "BIOMETRIC_TEMPLATE_KEY"


def _env(value):
    """Patches os.environ with BIOMETRIC_TEMPLATE_KEY set to value, or removed for None."""
    environ = {k: v for k, v in os.environ.items() if k != ENV}
    if value is not None:
        environ[ENV] = value
    return patch.dict(os.environ, environ, clear=True)


@override_settings(BIOMETRIC={})
class TestTemplateKeyResolution(SimpleTestCase):

    def setUp(self):
        super().setUp()
        self.env_key = Fernet.generate_key().decode()
        self.module_key = Fernet.generate_key().decode()
        self.settings_key = Fernet.generate_key().decode()

    def _resolved(self, module_cfg=None):
        return biometric_apps.resolve_config(module_cfg or {})["template_key"]

    def test_the_key_is_read_from_the_environment(self):
        with _env(self.env_key):
            self.assertEqual(self._resolved(), self.env_key)

    def test_the_module_configuration_takes_precedence(self):
        with _env(self.env_key):
            self.assertEqual(self._resolved({"template_key": self.module_key}), self.module_key)

    def test_settings_biometric_takes_precedence(self):
        for name in ("TEMPLATE_KEY", "template_key"):
            with self.subTest(name), _env(self.env_key), override_settings(BIOMETRIC={name: self.settings_key}):
                self.assertEqual(self._resolved({"template_key": self.module_key}), self.settings_key)

    def test_a_blank_environment_value_is_no_key(self):
        for value in ("", "   ", None):
            with self.subTest(repr(value)), _env(value):
                self.assertIsNone(self._resolved())

    def test_a_blank_configured_value_falls_back_to_the_environment(self):
        with _env(self.env_key):
            self.assertEqual(self._resolved({"template_key": "  "}), self.env_key)
            with override_settings(BIOMETRIC={"TEMPLATE_KEY": ""}):
                self.assertEqual(self._resolved({"template_key": None}), self.env_key)


@override_settings(BIOMETRIC={})
class TestTemplateKeyFromTheEnvironment(ConfigSnapshotMixin, TestCase):

    def setUp(self):
        super().setUp()
        self.key = Fernet.generate_key().decode()
        self._store({"modalities": FAKE_FACE})

    def _refresh_with_env_key(self):
        with _env(self.key):
            BiometricConfig.refresh_config()

    def test_templates_are_encrypted_with_it(self):
        self._refresh_with_env_key()

        row = enrol("individual.Individual", "s1", "face", b"photo", actor="agent")

        row.refresh_from_db()
        self.assertEqual(BiometricConfig.template_key, self.key)
        self.assertTrue(row.encrypted)
        self.assertEqual(crypto.decrypt_vector(row.vector, self.key), FakeEmbeddingProvider().extract(b"photo").vector)

    def test_a_blank_environment_value_stores_templates_unencrypted(self):
        with _env(" "):
            BiometricConfig.refresh_config()

        row = enrol("individual.Individual", "s1", "face", b"photo", actor="agent")

        self.assertIsNone(BiometricConfig.template_key)
        self.assertFalse(row.encrypted)

    def test_the_key_is_in_no_configuration_query_result(self):
        self._refresh_with_env_key()
        user = _User(perms=["174007"])

        for query in (CRITERIA_QUERY, RETENTION_QUERY):
            with self.subTest(query.split()[2]):
                result = _execute(query, user)
                self.assertIsNone(result.errors)
                self.assertNotIn(self.key, json.dumps(result.data))

    def test_the_key_is_in_no_log_line(self):
        with self.assertLogs("biometric", level="DEBUG") as logs:
            self._refresh_with_env_key()
            enrol("individual.Individual", "s1", "face", b"photo", actor="agent")
            # A queryset update bypasses the validator, as a direct database write would.
            self._stored_rows().update(config=json.dumps({
                "modalities": FAKE_FACE, "dedup_threshold": {"face": "high"}, "template_key": self.key,
            }))
            self._refresh_with_env_key()

        self.assertTrue(logs.output)
        self.assertNotIn(self.key, "\n".join(logs.output))

    def test_the_key_is_never_written_to_the_module_configuration(self):
        before = list(self._stored_rows().values_list("config", flat=True))

        self._refresh_with_env_key()
        enrol("individual.Individual", "s1", "face", b"photo", actor="agent")

        after = list(self._stored_rows().values_list("config", flat=True))
        self.assertEqual(after, before)
        self.assertNotIn(self.key, "".join(after))
