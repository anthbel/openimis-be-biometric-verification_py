"""
BiometricConfig.refresh_config() applies a changed `biometric` module
configuration on request_started and at the start of every candidate scan,
with settings.BIOMETRIC still winning; core's module configuration validator
refuses an unusable dedup_threshold / dedup_modalities.
"""

import json
import math

from django.core.exceptions import ValidationError
from django.core.signals import request_started
from django.db import close_old_connections
from django.test import TestCase, override_settings

from biometric import apps as biometric_apps
from biometric.apps import BiometricConfig
from biometric.dedup_source import BiometricCandidateSource
from biometric.models import BiometricTemplate
from biometric.providers.fake import FakeEmbeddingProvider
from biometric.registry import ProviderRegistry

SUBJECT_MODEL = "individual.Individual"
FAKE_FACE = {"face": {"provider": "fake_embedding", "threshold": 0.68}}
# Cosine similarity of these two vectors is 0.95.
VECTOR_A = [1.0, 0.0]
VECTOR_B = [0.95, math.sqrt(1 - 0.95 ** 2)]


class ConfigSnapshotMixin:
    """
    Restores every BiometricConfig setting, the applied/rejected snapshots
    refresh_config() compares against, and the provider registry, so a
    reload in one test never reaches the next.
    """

    def setUp(self):
        super().setUp()
        names = set(biometric_apps.DEFAULT_CFG) | {"_applied_config", "_rejected_config"}
        snapshot = {name: getattr(BiometricConfig, name, None) for name in names}

        def restore():
            for name, value in snapshot.items():
                setattr(BiometricConfig, name, value)

        self.addCleanup(restore)
        registry = dict(ProviderRegistry._modality_registry)
        instances = dict(ProviderRegistry._modality_instances)

        def restore_registry():
            ProviderRegistry._modality_registry.clear()
            ProviderRegistry._modality_registry.update(registry)
            ProviderRegistry._modality_instances.clear()
            ProviderRegistry._modality_instances.update(instances)

        self.addCleanup(restore_registry)
        ProviderRegistry._modality_instances.clear()
        ProviderRegistry.register_modality("face", "fake_embedding", FakeEmbeddingProvider)

    @staticmethod
    def _store(cfg):
        from core.models import ModuleConfiguration

        return ModuleConfiguration.objects.create(module="biometric", layer="be", version="1", config=json.dumps(cfg))

    @staticmethod
    def _stored_rows():
        from core.models import ModuleConfiguration

        return ModuleConfiguration.objects.filter(module="biometric")


@override_settings(BIOMETRIC={})
class TestRefreshAppliesTheModuleConfiguration(ConfigSnapshotMixin, TestCase):

    def setUp(self):
        super().setUp()
        BiometricConfig.modalities = dict(FAKE_FACE)
        BiometricConfig.dedup_threshold = {"face": 0.5}
        for subject_id, vector in (("a-subject", VECTOR_A), ("b-subject", VECTOR_B)):
            BiometricTemplate.objects.create(
                subject_model=SUBJECT_MODEL, subject_id=subject_id, modality="face", kind="embedding",
                vector=vector, provider="fake_embedding", model_name="",
            )

    @staticmethod
    def _scan():
        return list(BiometricCandidateSource(modality="face").scan(None))

    def test_a_threshold_changed_in_the_module_configuration_applies_to_the_next_scan(self):
        self.assertEqual(len(self._scan()), 1)

        row = self._store({"modalities": FAKE_FACE, "dedup_threshold": {"face": 0.99}})
        self.assertEqual(self._scan(), [])

        row.config = json.dumps({"modalities": FAKE_FACE, "dedup_threshold": {"face": 0.9}})
        row.save()
        self.assertEqual(len(self._scan()), 1)
        self.assertEqual(BiometricConfig.dedup_threshold, {"face": 0.9})

    def test_settings_biometric_still_wins(self):
        self._store({"modalities": FAKE_FACE, "dedup_threshold": {"face": 0.5}})

        with override_settings(BIOMETRIC={"DEDUP_THRESHOLD": {"face": 0.99}}):
            self.assertEqual(self._scan(), [])
            self.assertEqual(BiometricConfig.dedup_threshold, {"face": 0.99})

    def test_request_started_applies_a_stored_change(self):
        self._store({"modalities": FAKE_FACE, "dedup_threshold": {"face": 0.75}})
        # close_old_connections would close the test transaction's connection.
        request_started.disconnect(close_old_connections)
        self.addCleanup(request_started.connect, close_old_connections)

        request_started.send(sender=self.__class__)

        self.assertEqual(BiometricConfig.dedup_threshold, {"face": 0.75})

    def test_an_unchanged_configuration_costs_one_query_and_reloads_nothing(self):
        BiometricConfig.refresh_config()
        BiometricConfig.dedup_threshold = {"face": 0.1}

        with self.assertNumQueries(1):
            BiometricConfig.refresh_config()

        self.assertEqual(BiometricConfig.dedup_threshold, {"face": 0.1})

    def test_an_invalid_stored_configuration_is_logged_by_key_once_and_the_last_valid_one_stays(self):
        row = self._store({"modalities": FAKE_FACE, "dedup_threshold": {"face": 0.9}})
        BiometricConfig.refresh_config()
        secret = "SENTINEL-TEMPLATE-KEY"
        # A queryset update bypasses the validator, as a direct database write would.
        self._stored_rows().filter(id=row.id).update(config=json.dumps({
            "modalities": FAKE_FACE, "dedup_threshold": {"face": "high"}, "template_key": secret,
        }))

        with self.assertLogs("biometric", level="ERROR") as logs:
            BiometricConfig.refresh_config()

        self.assertEqual(BiometricConfig.dedup_threshold, {"face": 0.9})
        output = "\n".join(logs.output)
        self.assertIn("dedup_threshold['face'] is not a number", output)
        self.assertNotIn(secret, output)
        self.assertNotIn("high", output)
        with self.assertNoLogs("biometric", level="ERROR"):
            BiometricConfig.refresh_config()


@override_settings(BIOMETRIC={})
class TestModuleConfigurationValidator(ConfigSnapshotMixin, TestCase):

    def _assert_refused(self, cfg, message):
        with self.assertRaises(ValidationError) as raised:
            self._store(cfg)

        self.assertIn(message, " ".join(raised.exception.messages))
        self.assertFalse(self._stored_rows().exists())

    def test_a_non_numeric_threshold_is_refused(self):
        self._assert_refused({"dedup_threshold": {"face": "high"}}, "dedup_threshold['face'] is not a number")

    def test_a_boolean_threshold_is_refused(self):
        self._assert_refused({"dedup_threshold": {"face": True}}, "dedup_threshold['face'] is not a number")

    def test_a_dedup_modality_without_a_threshold_is_refused(self):
        self._assert_refused(
            {"dedup_threshold": {"face": 0.62}, "dedup_modalities": ["face", "fingerprint"]},
            "dedup_modalities entry 'fingerprint' has no numeric dedup_threshold entry",
        )

    def test_a_dedup_modality_outside_modalities_is_refused(self):
        self._assert_refused(
            {"dedup_threshold": {"face": 0.62, "iris": 0.5}, "dedup_modalities": ["iris"]},
            "dedup_modalities entry 'iris' is not a key of modalities",
        )

    def test_the_message_never_quotes_the_template_key(self):
        secret = "SENTINEL-TEMPLATE-KEY"

        with self.assertRaises(ValidationError) as raised:
            self._store({"dedup_threshold": {"face": "high"}, "template_key": secret})

        self.assertNotIn(secret, str(raised.exception))

    def test_a_valid_configuration_is_stored(self):
        self._store({"dedup_threshold": {"face": 0.7}, "dedup_modalities": ["face"]})

        self.assertEqual(self._stored_rows().count(), 1)

    def test_dedup_modalities_defaults_to_face(self):
        self.assertEqual(biometric_apps.DEFAULT_CFG["dedup_modalities"], ["face"])
        self.assertEqual(biometric_apps._SETTINGS_KEY_MAP["DEDUP_MODALITIES"], "dedup_modalities")
