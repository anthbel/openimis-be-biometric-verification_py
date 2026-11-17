"""
Registered consent checks (services.register_consent_check) run for every
caller, superusers included, before enrol() or a granting recordBiometricConsent
writes anything; a refusal (granted false) is recorded unchecked; enrol() lists
the checks that passed in metadata["consent_checks"].
"""

import base64
from types import SimpleNamespace

import graphene
from django.core.cache import cache

from biometric import services
from biometric.apps import BiometricConfig
from biometric.models import BiometricAuditEvent, BiometricConsent, BiometricTemplate
from biometric.providers.base import Extracted
from biometric.schema import Mutation
from biometric.services import ConsentRequiredError, enrol
from biometric.tests.test_services import SUBJECT_MODEL, _MultimodalServiceTestCase
from biometric.tests.test_subject_scope import _individual, _role, _Root

SAMPLE = base64.b64encode(b"photo").decode()


class _ConsentCheckTestCase(_MultimodalServiceTestCase):

    def setUp(self):
        super().setUp()
        checks = getattr(services, "_consent_checks", None)
        if checks is not None:
            snapshot = dict(checks)

            def restore():
                checks.clear()
                checks.update(snapshot)

            self.addCleanup(restore)
            checks.clear()
        self.calls = []

    def _register(self, key, refuse=False):
        def check(subject_model, subject_id, modality):
            self.calls.append((key, subject_model, subject_id, modality))
            if refuse:
                raise ConsentRequiredError(f"{key}: no consent")

        services.register_consent_check(key, check)


class TestEnrolRunsTheChecks(_ConsentCheckTestCase):

    def test_a_refusing_check_stops_enrol_before_anything_is_written(self):
        previous = enrol(SUBJECT_MODEL, "s1", "face", b"earlier", actor="agent")
        BiometricConfig.audit = {"enabled": True, "rules": {}}
        events = BiometricAuditEvent.objects.count()
        self._register("first")
        self._register("registry", refuse=True)

        with self.assertRaises(ConsentRequiredError):
            enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="agent")

        self.assertEqual(BiometricTemplate.objects.count(), 1)
        previous.refresh_from_db()
        self.assertIsNone(previous.validity_to)
        self.assertEqual(BiometricAuditEvent.objects.count(), events)
        self.assertEqual([call[0] for call in self.calls], ["first", "registry"])

    def test_the_passing_checks_are_listed_in_registration_order(self):
        self._register("first")
        self._register("registry")

        row = enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="agent")

        row.refresh_from_db()
        self.assertEqual(row.metadata["consent_checks"], ["first", "registry"])
        self.assertEqual(self.calls, [
            ("first", SUBJECT_MODEL, "s1", "face"), ("registry", SUBJECT_MODEL, "s1", "face"),
        ])

    def test_registering_a_key_again_replaces_its_check(self):
        self._register("registry", refuse=True)
        self._register("registry")

        row = enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="agent")

        self.assertEqual(row.metadata["consent_checks"], ["registry"])

    def test_no_registered_check_records_an_empty_list(self):
        row = enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="agent")

        row.refresh_from_db()
        self.assertEqual(row.metadata["consent_checks"], [])

    def test_a_caller_supplied_value_is_overwritten(self):
        forged = {"consent_checks": ["social_registry"], "k": "v"}

        unchecked = enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="agent", metadata=forged)
        unchecked.refresh_from_db()
        self.assertEqual(unchecked.metadata, {"consent_checks": [], "k": "v"})

        self._register("other")
        checked = enrol(SUBJECT_MODEL, "s2", "face", b"photo", actor="agent", metadata=forged)
        checked.refresh_from_db()
        self.assertEqual(checked.metadata, {"consent_checks": ["other"], "k": "v"})

    def test_the_device_template_path_records_the_checks(self):
        self._register("registry")
        device = Extracted(vector=[0.1] * 8, metadata={"consent_checks": ["forged"], "device": "tab-1"})

        row = enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="agent", device_template=device)

        row.refresh_from_db()
        self.assertEqual(row.metadata, {"consent_checks": ["registry"], "device": "tab-1"})

    def test_the_device_template_path_is_refused_by_a_check(self):
        self._register("registry", refuse=True)

        with self.assertRaises(ConsentRequiredError):
            enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="agent", device_template=Extracted(vector=[0.1] * 8))

        self.assertFalse(BiometricTemplate.objects.exists())


class TestRecordConsent(_ConsentCheckTestCase):

    def test_a_refused_grant_writes_nothing(self):
        self._register("registry", refuse=True)

        with self.assertRaises(ConsentRequiredError):
            services.record_consent(SUBJECT_MODEL, "s1", "face", True, actor="agent")

        self.assertFalse(BiometricConsent.objects.exists())

    def test_a_refusal_is_recorded_without_any_check(self):
        self._register("registry", refuse=True)

        row = services.record_consent(SUBJECT_MODEL, "s1", "face", False, actor="agent", note="withdrawn")

        self.assertEqual(self.calls, [])
        self.assertEqual(
            (row.subject_model, row.subject_id, row.modality, row.granted, row.recorded_by, row.note),
            (SUBJECT_MODEL, "s1", "face", False, "agent", "withdrawn"),
        )

    def test_a_passing_grant_is_recorded(self):
        self._register("registry")

        row = services.record_consent(None, "s1", "face", True, actor="agent")

        self.assertEqual(self.calls, [("registry", BiometricConfig.subject_model, "s1", "face")])
        self.assertTrue(BiometricConsent.objects.filter(id=row.id, granted=True, note="").exists())


class TestGraphqlCallersAreChecked(_ConsentCheckTestCase):
    """A superuser and a user holding only 174001 meet the same checks."""

    def setUp(self):
        super().setUp()
        from core.test_helpers import create_admin_role, create_test_interactive_user
        from location.test_helpers import assign_user_districts, create_test_village

        self.addCleanup(
            setattr, BiometricConfig, "gql_biometric_enrol_perms", BiometricConfig.gql_biometric_enrol_perms,
        )
        BiometricConfig.gql_biometric_enrol_perms = ["174001"]
        cache.clear()
        self.addCleanup(cache.clear)
        village = create_test_village({"name": "Consent Village", "code": "BioCoV"})
        self.superuser = create_test_interactive_user(username="bioConsentAdmin", roles=[create_admin_role().id])
        self.enroller = create_test_interactive_user(
            username="bioConsentEnroller", roles=[_role("Biometric enroller", (174001,)).id],
        )
        assign_user_districts(self.enroller, [village.parent.parent.code])
        cache.clear()
        self.subject_id = _individual(self.superuser, village)

    def _execute(self, query, user):
        return graphene.Schema(query=_Root, mutation=Mutation).execute(
            query, context_value=SimpleNamespace(user=user, headers={}),
        )

    def _enrol(self, user):
        return self._execute(
            'mutation { enrolBiometric(subjectId: "%s", modality: "face", sample: "%s") { id } }'
            % (self.subject_id, SAMPLE), user,
        )

    def _consent(self, user, granted):
        return self._execute(
            'mutation { recordBiometricConsent(subjectId: "%s", modality: "face", granted: %s) { ok id } }'
            % (self.subject_id, "true" if granted else "false"), user,
        )

    def _users(self):
        self.assertTrue(self.superuser.is_superuser)
        self.assertFalse(self.enroller.is_superuser)
        return {"superuser": self.superuser, "174001 holder": self.enroller}

    def test_a_refusing_check_refuses_enrol_and_a_grant_for_every_caller(self):
        self._register("registry", refuse=True)

        for label, user in self._users().items():
            with self.subTest(label):
                for result, field in ((self._enrol(user), "enrolBiometric"),
                                      (self._consent(user, True), "recordBiometricConsent")):
                    self.assertIsNone((result.data or {}).get(field))
                    self.assertIsInstance(getattr(result.errors[0], "original_error", None), ConsentRequiredError)
        self.assertFalse(BiometricTemplate.objects.exists())
        self.assertFalse(BiometricConsent.objects.exists())

    def test_a_refusal_is_recorded_unchecked_for_every_caller(self):
        self._register("registry", refuse=True)

        for label, user in self._users().items():
            with self.subTest(label):
                result = self._consent(user, False)
                self.assertIsNone(result.errors)
                self.assertTrue(result.data["recordBiometricConsent"]["ok"])
        self.assertEqual(self.calls, [])
        self.assertEqual(BiometricConsent.objects.filter(subject_id=self.subject_id, granted=False).count(), 2)

    def test_a_passing_check_lets_both_mutations_through(self):
        self._register("registry")

        for label, user in self._users().items():
            with self.subTest(label):
                self.assertIsNone(self._consent(user, True).errors)
                self.assertIsNone(self._enrol(user).errors)
        row = BiometricTemplate.objects.get(subject_id=self.subject_id, validity_to__isnull=True)
        self.assertEqual(row.metadata["consent_checks"], ["registry"])
        self.assertEqual(BiometricConsent.objects.filter(subject_id=self.subject_id, granted=True).count(), 2)
