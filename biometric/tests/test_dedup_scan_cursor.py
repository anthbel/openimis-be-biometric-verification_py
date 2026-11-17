"""Two connections: a face template committed after a duplicate scan's cursor was taken, but stamped before it, is
compared by the next scan and does not cover its subject until then."""
import threading
import unittest
import uuid
from unittest import mock

from django.apps import apps
from django.db import connection, connections, transaction
from django.test import TransactionTestCase

from biometric.apps import BiometricConfig
from biometric.dedup_source import BiometricCandidateSource, order_pair
from biometric.models import BiometricTemplate
from biometric.providers.fake import FakeEmbeddingProvider
from biometric.registry import ProviderRegistry

SUBJECT_MODEL = "individual.Individual"
WAIT_SECONDS = 10


@unittest.skipUnless(apps.is_installed("deduplication"), "the deduplication app is not installed")
class LateFaceCommitTest(TransactionTestCase):
    def setUp(self):
        if connection.vendor != "postgresql":
            self.skipTest("This test can only be executed for PSQL database")
        from deduplication.models import ScanState

        dedup_threshold, modalities = BiometricConfig.dedup_threshold, BiometricConfig.modalities
        registry = dict(ProviderRegistry._modality_registry)
        instances = dict(ProviderRegistry._modality_instances)

        def restore():
            BiometricConfig.dedup_threshold, BiometricConfig.modalities = dedup_threshold, modalities
            ProviderRegistry._modality_registry.clear()
            ProviderRegistry._modality_registry.update(registry)
            ProviderRegistry._modality_instances.clear()
            ProviderRegistry._modality_instances.update(instances)

        self.addCleanup(restore)
        BiometricConfig.dedup_threshold = {"face": 0.9}
        BiometricConfig.modalities = {"face": {"provider": "fake_embedding", "threshold": 0.68}}
        ProviderRegistry._modality_instances.clear()
        ProviderRegistry.register_modality("face", "fake_embedding", FakeEmbeddingProvider)

        self.source = BiometricCandidateSource(modality="face")
        sources = mock.patch("deduplication.services.registered_sources", return_value=[self.source])
        sources.start()
        self.addCleanup(sources.stop)
        ScanState.objects.filter(kind=self.source.kind).delete()
        self.registered, self.late, self.other = (str(uuid.uuid4()) for _ in range(3))
        self.template(self.registered, [1.0, 0.0])
        self.other_template = self.template(self.other, [0.0, 1.0])

    def _fixture_teardown(self):
        from deduplication.models import DuplicateCandidate, ScanState

        subjects = [self.registered, self.late, self.other]
        ScanState.objects.filter(kind="biometric").delete()
        DuplicateCandidate.objects.filter(kind="biometric", subject_a__in=subjects).delete()
        BiometricTemplate.objects.filter(subject_model=SUBJECT_MODEL, subject_id__in=subjects).delete()

    @staticmethod
    def template(subject_id, vector):
        return BiometricTemplate.objects.create(
            subject_model=SUBJECT_MODEL, subject_id=subject_id, modality="face",
            kind="embedding", vector=vector, provider="fake_embedding", model_name="",
        )

    def scan(self):
        from deduplication.services import run_scan

        self.assertEqual(set(run_scan(kinds=[self.source.kind], actor="test")), {self.source.kind})

    def state(self):
        from deduplication.models import ScanState

        return ScanState.objects.get(kind=self.source.kind)

    def enrolled_after_a_scan(self):
        """Enrols the late face in another connection, saves a later template meanwhile, scans, then commits the
        late face: its stamp is before the scan's cursor, its commit after."""
        saved, commit, results = threading.Event(), threading.Event(), {}

        def body():
            try:
                with transaction.atomic():
                    results["late"] = self.template(self.late, [1.0, 0.0])
                    saved.set()
                    if not commit.wait(WAIT_SECONDS):
                        raise TimeoutError("commit never signalled")
            except Exception as exc:
                results["error"] = exc
                saved.set()
            finally:
                connections.close_all()

        writer = threading.Thread(target=body, name="late-enrolment")
        writer.start()
        try:
            self.assertTrue(saved.wait(WAIT_SECONDS))
            self.assertNotIn("error", results)
            self.other_template.save()
            self.scan()
        finally:
            commit.set()
            writer.join(2 * WAIT_SECONDS)
        self.assertFalse(writer.is_alive())
        self.assertNotIn("error", results)
        self.assertLess(results["late"].date_updated, self.state().updated_at)

    def recorded(self):
        from deduplication.models import DuplicateCandidate

        a, b = order_pair(self.registered, self.late)
        return DuplicateCandidate.objects.filter(kind=self.source.kind, subject_a=a, subject_b=b).exists()

    def covered(self):
        return self.source.covered(SUBJECT_MODEL, [self.late], self.state())

    def test_a_face_committed_after_the_cursor_does_not_cover_its_subject(self):
        self.enrolled_after_a_scan()
        self.assertEqual(self.covered(), set())

    def test_the_next_scan_compares_a_face_committed_after_the_cursor_and_covers_its_subject(self):
        self.enrolled_after_a_scan()
        self.scan()
        self.assertTrue(self.recorded())
        self.assertEqual(self.covered(), {self.late})
        self.scan()
        self.assertTrue(self.recorded())
        self.assertEqual(self.covered(), {self.late})
