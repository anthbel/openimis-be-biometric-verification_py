"""
Unit tests for dedup_source.BiometricCandidateSource (§3.5, §3.7): threshold,
pair ordering, and watermark advance. Exercises whichever Candidate/Watermark
shape is active in this environment (real deduplication.sources here, since
it's installed side-by-side — see dedup_source.py's try/except).
"""

import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

from django.test import TestCase

from biometric.apps import BiometricConfig
from biometric.dedup_source import BiometricCandidateSource, Watermark, order_pair
from biometric.models import BiometricTemplate
from biometric.providers.fake import FakeEmbeddingProvider
from biometric.registry import ProviderRegistry

SUBJECT_MODEL = "individual.Individual"


class _DedupSourceTestCase(TestCase):
    """
    identify() (called by scan()) filters the gallery on the modality's
    configured provider/model_name, so the fake provider registered here must
    match the provider/model_name stamped on the manually-created rows below.
    """

    def setUp(self):
        super().setUp()
        self._dedup_threshold = BiometricConfig.dedup_threshold
        self._modalities = BiometricConfig.modalities
        BiometricConfig.dedup_threshold = {"face": 0.9}
        BiometricConfig.modalities = {"face": {"provider": "fake_embedding", "threshold": 0.68}}
        self._registry_snapshot = dict(ProviderRegistry._modality_registry)
        self._instances_snapshot = dict(ProviderRegistry._modality_instances)
        ProviderRegistry._modality_instances.clear()
        ProviderRegistry.register_modality("face", "fake_embedding", FakeEmbeddingProvider)

    def tearDown(self):
        super().tearDown()
        BiometricConfig.dedup_threshold = self._dedup_threshold
        BiometricConfig.modalities = self._modalities
        ProviderRegistry._modality_registry.clear()
        ProviderRegistry._modality_registry.update(self._registry_snapshot)
        ProviderRegistry._modality_instances.clear()
        ProviderRegistry._modality_instances.update(self._instances_snapshot)

    def _make_template(self, subject_id, vector):
        return BiometricTemplate.objects.create(
            subject_model=SUBJECT_MODEL, subject_id=subject_id, modality="face",
            kind="embedding", vector=vector, provider="fake_embedding", model_name="",
        )


class TestScan(_DedupSourceTestCase):

    def test_yields_candidate_at_or_above_threshold(self):
        self._make_template("b-subject", [1.0, 0.0])
        self._make_template("a-subject", [1.0, 0.0])  # identical -> similarity 1.0

        source = BiometricCandidateSource(modality="face")
        candidates = list(source.scan(None))

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.kind, "biometric")
        self.assertGreaterEqual(candidate.score, 0.9)

    def test_below_threshold_yields_nothing(self):
        self._make_template("a-subject", [1.0, 0.0])
        self._make_template("b-subject", [0.0, 1.0])  # orthogonal -> similarity 0.0

        source = BiometricCandidateSource(modality="face")
        candidates = list(source.scan(None))

        self.assertEqual(candidates, [])

    def test_pair_is_ordered_with_order_pair(self):
        self._make_template("zzz", [1.0, 0.0])
        self._make_template("aaa", [1.0, 0.0])

        source = BiometricCandidateSource(modality="face")
        candidates = list(source.scan(None))

        self.assertEqual(len(candidates), 1)
        expected = order_pair("zzz", "aaa")
        self.assertEqual((candidates[0].subject_a, candidates[0].subject_b), expected)
        self.assertLess(candidates[0].subject_a, candidates[0].subject_b)

    def test_evidence_names_the_two_templates(self):
        t1 = self._make_template("a-subject", [1.0, 0.0])
        t2 = self._make_template("b-subject", [1.0, 0.0])

        source = BiometricCandidateSource(modality="face")
        candidates = list(source.scan(None))

        evidence = candidates[0].evidence
        self.assertEqual(evidence["modality"], "face")
        self.assertEqual(evidence["provider"], "fake_embedding")
        self.assertEqual({evidence["template_a"], evidence["template_b"]}, {str(t1.id), str(t2.id)})

    def test_scan_ignores_other_modalities(self):
        BiometricTemplate.objects.create(
            subject_model=SUBJECT_MODEL, subject_id="a", modality="fingerprint",
            kind="template", template=b"x", provider="device_reported", model_name="",
        )
        source = BiometricCandidateSource(modality="face")
        self.assertEqual(list(source.scan(None)), [])


class TestWatermark(_DedupSourceTestCase):

    def test_none_when_no_active_templates(self):
        source = BiometricCandidateSource(modality="face")
        watermark = source.watermark()
        self.assertIsNone(watermark.updated_at)
        self.assertIsNone(watermark.last_id)

    def test_advances_to_newest_active_template(self):
        self._make_template("a", [1.0, 0.0])
        newest = self._make_template("b", [0.0, 1.0])

        source = BiometricCandidateSource(modality="face")
        watermark = source.watermark()

        self.assertEqual(watermark.last_id, str(newest.id))
        self.assertIsNotNone(watermark.updated_at)

    def test_scan_since_watermark_skips_already_seen_rows(self):
        self._make_template("a-subject", [1.0, 0.0])
        first_watermark = BiometricCandidateSource(modality="face").watermark()

        self._make_template("b-subject", [1.0, 0.0])  # arrives "since" the first watermark

        source = BiometricCandidateSource(modality="face")
        candidates = list(source.scan(first_watermark))

        # Only the newly-scanned template drives a scan iteration, but it still
        # matches against the full gallery (including the earlier template).
        self.assertEqual(len(candidates), 1)


class TestCovered(_DedupSourceTestCase):
    """covered(): a subject is covered when every active face template of it is at or before the cursor, or
    when it has none, once a scan has completed."""

    def state(self, watermark=None, scanned=True):
        """A stand-in for the source's ScanState row after a scan that took ``watermark`` as its cursor."""
        watermark = watermark or Watermark()
        return SimpleNamespace(
            kind="biometric", updated_at=watermark.updated_at, last_id=watermark.last_id or "",
            last_scan_at=datetime.now() if scanned else None,
        )

    def test_an_empty_gallery_covers_every_subject_once_scanned(self):
        source = BiometricCandidateSource(modality="face")
        state = self.state(source.watermark())
        self.assertIsNone(state.updated_at)
        self.assertEqual(source.covered(SUBJECT_MODEL, ["a", "b"], state), {"a", "b"})

    def test_no_completed_scan_covers_nobody(self):
        source = BiometricCandidateSource(modality="face")
        self.assertEqual(source.covered(SUBJECT_MODEL, ["a"], self.state(scanned=False)), set())
        self.assertEqual(source.covered(SUBJECT_MODEL, ["a"], None), set())

    def test_a_template_at_or_before_the_cursor_is_covered(self):
        self._make_template("a", [1.0, 0.0])
        source = BiometricCandidateSource(modality="face")
        state = self.state(source.watermark())
        self.assertEqual(source.covered(SUBJECT_MODEL, ["a", "b"], state), {"a", "b"})

    def test_a_template_enrolled_after_the_cursor_holds_its_subject(self):
        self._make_template("a", [1.0, 0.0])
        source = BiometricCandidateSource(modality="face")
        state = self.state(source.watermark())
        self._make_template("b", [0.0, 1.0])
        self.assertEqual(source.covered(SUBJECT_MODEL, ["a", "b", "c"], state), {"a", "c"})

    def test_a_template_enrolled_after_an_empty_gallery_scan_holds_its_subject(self):
        source = BiometricCandidateSource(modality="face")
        state = self.state(source.watermark())
        self._make_template("a", [1.0, 0.0])
        self.assertEqual(source.covered(SUBJECT_MODEL, ["a", "b"], state), {"b"})

    def test_one_newer_template_of_several_holds_the_subject(self):
        self._make_template("a", [1.0, 0.0])
        source = BiometricCandidateSource(modality="face")
        state = self.state(source.watermark())
        BiometricTemplate.objects.create(
            subject_model=SUBJECT_MODEL, subject_id="a", modality="face", kind="embedding", vector=[0.0, 1.0],
            provider="fake_embedding", model_name="other",
        )
        self.assertEqual(source.covered(SUBJECT_MODEL, ["a"], state), set())

    def test_other_modalities_subject_models_and_superseded_templates_are_ignored(self):
        source = BiometricCandidateSource(modality="face")
        state = self.state(source.watermark())
        BiometricTemplate.objects.create(
            subject_model=SUBJECT_MODEL, subject_id="a", modality="fingerprint",
            kind="template", template=b"x", provider="device_reported", model_name="",
        )
        BiometricTemplate.objects.create(
            subject_model="insuree.Insuree", subject_id="a", modality="face", kind="embedding", vector=[1.0, 0.0],
            provider="fake_embedding", model_name="",
        )
        superseded = self._make_template("a", [1.0, 0.0])
        superseded.validity_to = datetime.now()
        superseded.save()
        self.assertEqual(source.covered(SUBJECT_MODEL, ["a"], state), {"a"})

    def test_ids_come_back_as_passed(self):
        person = uuid.uuid4()
        self._make_template(str(person), [1.0, 0.0])
        source = BiometricCandidateSource(modality="face")
        state = self.state(source.watermark())
        other = uuid.uuid4()
        self.assertEqual(source.covered(SUBJECT_MODEL, [person, other], state), {person, other})
        later = self.state(Watermark(updated_at=state.updated_at - timedelta(seconds=1), last_id=""))
        self.assertEqual(source.covered(SUBJECT_MODEL, [person, other], later), {other})
