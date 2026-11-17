"""
Sample and template arguments through the assembly's GraphQL view: an argument of the wrong shape is refused
with a fixed message, and the sample or template sent reaches neither the response nor any log record.
"""

import base64
import json
import logging

from graphene_django.settings import graphene_settings

from core.models.openimis_graphql_test_case import BaseTestContext, openIMISGraphQLTestCase
from core.test_helpers import create_test_interactive_user

from biometric.models import BiometricMultimodalDecision, BiometricTemplate, BiometricVerification
from biometric.schema import LEGS_REFUSED, SAMPLE_REFUSED, VECTOR_REFUSED
from biometric.tests.synthetic_subjects import SyntheticSubjectsMixin
from biometric.tests.test_services import _MultimodalServiceTestCase
from biometric.tests.test_subject_scope import _role

# 87 bytes: base64 without padding, so one character more is not base64.
SAMPLE = base64.b64encode(b"raw-face-sample-of-the-person" * 3).decode()
TEMPLATE = base64.b64encode(b"device-template-of-the-person" * 3).decode()
VECTOR_DIGITS = "0.123456789"
REFUSED_CODE = "BIOMETRIC_SAMPLE_INPUT_REFUSED"


class EveryLogger(logging.Handler):
    """Every record of every logger at DEBUG, formatted with its traceback and its attributes."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.texts = []
        self.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))

    def emit(self, record):
        self.texts.append(self.format(record))
        self.texts.append(repr(vars(record)))

    def __enter__(self):
        self._disabled = logging.root.manager.disable
        logging.disable(logging.NOTSET)
        loggers = [logging.getLogger()] + [
            logger for logger in logging.root.manager.loggerDict.values() if isinstance(logger, logging.Logger)
        ]
        self._saved = [(logger, logger.level, logger.disabled) for logger in loggers]
        for logger in loggers:
            logger.setLevel(logging.DEBUG)
            logger.disabled = False
            if logger is logging.root or not logger.propagate:
                logger.addHandler(self)
        return self

    def __exit__(self, *exc):
        for logger, level, disabled in self._saved:
            logger.removeHandler(self)
            logger.setLevel(level)
            logger.disabled = disabled
        logging.disable(self._disabled)

    def text(self):
        return "\n".join(self.texts)


def _argument_type(operation, field, argument):
    """The type the served schema gives an argument, as a client declares its variable."""
    schema = graphene_settings.SCHEMA
    root = schema.get_mutation_type() if operation == "mutation" else schema.get_query_type()
    return str(root.fields[field].args[argument].type)


ENROL = "mutation ($sample: %s) { enrolBiometric(subjectId: \"s1\", modality: \"face\", sample: $sample) { id } }"
VERIFY = (
    "mutation ($modality: String!, $deviceScore: Float, $sample: %s, $deviceVector: %s, $deviceTemplate: %s) { "
    "verifyBiometric(subjectId: \"s1\", modality: $modality, deviceScore: $deviceScore, sample: $sample, "
    "deviceVector: $deviceVector, deviceTemplate: $deviceTemplate) { verified origin } }"
)
IDENTIFY = "query ($sample: %s) { identifyBiometric(modality: \"face\", sample: $sample) { subjectId } }"
MULTIMODAL = (
    "mutation ($legs: %s) { verifyBiometricMultimodal(subjectId: \"s1\", legs: $legs) { outcome legs { origin } } }"
)


class TestSampleInputsOverHttp(SyntheticSubjectsMixin, _MultimodalServiceTestCase, openIMISGraphQLTestCase):

    def setUp(self):
        super().setUp()
        self.user = create_test_interactive_user(
            username="bioSampleInputs", roles=[_role("Biometric sample inputs", (174001, 174002, 174003)).id],
        )
        self.token = BaseTestContext(user=self.user).get_jwt()

    def gql(self, query, variables=None):
        response = self.query(query, headers={"HTTP_AUTHORIZATION": f"Bearer {self.token}"}, variables=variables)
        return json.loads(response.content)

    def enrol(self, sample):
        return self.gql(ENROL % _argument_type("mutation", "enrolBiometric", "sample"), {"sample": sample})

    def verify(self, modality="face", **variables):
        types = tuple(_argument_type("mutation", "verifyBiometric", name)
                      for name in ("sample", "deviceVector", "deviceTemplate"))
        return self.gql(VERIFY % types, {"modality": modality, **variables})

    def multimodal(self, legs):
        legs_type = _argument_type("mutation", "verifyBiometricMultimodal", "legs")
        return self.gql(MULTIMODAL % legs_type, {"legs": legs})

    def assertRefused(self, answers, message):
        for answer in answers:
            with self.subTest(answer=answer):
                self.assertEqual([(error["message"], error.get("extensions", {}).get("code"))
                                  for error in answer["errors"]], [(message, REFUSED_CODE)])
                self.assertNotIn(SAMPLE[:24], json.dumps(answer))
                self.assertNotIn(TEMPLATE[:24], json.dumps(answer))
                self.assertNotIn(VECTOR_DIGITS, json.dumps(answer))

    def assertQuotedNowhere(self, capture):
        text = capture.text()
        self.assertIn("GraphQL client error", text)
        for sent in (SAMPLE[:24], TEMPLATE[:24], VECTOR_DIGITS):
            self.assertNotIn(sent, text)

    def test_enrol_refuses_a_sample_that_is_not_base64_text(self):
        with EveryLogger() as capture:
            answers = [self.enrol(sample) for sample in ({"data": SAMPLE}, [SAMPLE], 7, True, "a")]
            answers.append(self.gql(
                'mutation { enrolBiometric(subjectId: "s1", modality: "face", sample: {data: "%s"}) { id } }' % SAMPLE))
        self.assertRefused(answers, SAMPLE_REFUSED % "sample")
        self.assertQuotedNowhere(capture)
        self.assertFalse(BiometricTemplate.objects.exists())

    def test_verify_refuses_a_sample_vector_or_template_of_the_wrong_shape(self):
        with EveryLogger() as capture:
            samples = [self.verify(sample={"data": SAMPLE}), self.verify(sample=[SAMPLE])]
            templates = [self.verify(deviceTemplate={"data": TEMPLATE}), self.verify(deviceTemplate=12)]
            vectors = [self.verify(deviceVector=[float(VECTOR_DIGITS), SAMPLE]),
                       self.verify(deviceVector={"values": [float(VECTOR_DIGITS)]}),
                       self.verify(deviceVector=[float(VECTOR_DIGITS), None]),
                       self.verify(deviceVector=[float(VECTOR_DIGITS), True])]
            vectors.append(self.gql(
                'mutation { verifyBiometric(subjectId: "s1", modality: "face", deviceScore: 0.9, '
                'deviceVector: [%s, "%s"]) { verified } }' % (VECTOR_DIGITS, SAMPLE)))
        self.assertRefused(samples, SAMPLE_REFUSED % "sample")
        self.assertRefused(templates, SAMPLE_REFUSED % "deviceTemplate")
        self.assertRefused(vectors, VECTOR_REFUSED)
        self.assertQuotedNowhere(capture)
        self.assertFalse(BiometricVerification.objects.exists())

    def test_multimodal_refuses_legs_with_an_unknown_missing_or_mistyped_field(self):
        face = {"modality": "face", "sample": SAMPLE}
        with EveryLogger() as capture:
            answers = [self.multimodal(legs) for legs in (
                [{**face, "extra": "x"}],
                [{"sample": SAMPLE}],
                [{**face, "modality": None}],
                [{**face, "sample": {"data": SAMPLE}}],
                [{**face, "deviceScore": "high"}],
                [{**face, "deviceScore": False}],
                [{**face, "deviceVector": [float(VECTOR_DIGITS), SAMPLE]}],
                [{**face, "deviceTemplate": [TEMPLATE]}],
                [{**face, "position": 3}],
                [face, "face"],
                face,
            )]
            answers.append(self.gql(
                'mutation { verifyBiometricMultimodal(subjectId: "s1", legs: [{modality: "face", sample: "%s", '
                'extra: "x"}]) { outcome } }' % SAMPLE))
            answers.append(self.gql(
                'mutation { verifyBiometricMultimodal(subjectId: "s1", legs: [{sample: "%s"}]) { outcome } }'
                % SAMPLE))
        self.assertRefused(answers, LEGS_REFUSED)
        self.assertQuotedNowhere(capture)
        self.assertFalse(BiometricVerification.objects.exists())
        self.assertFalse(BiometricMultimodalDecision.objects.exists())

    def test_a_sample_leg_that_is_not_base64_is_refused_without_quoting_it(self):
        with EveryLogger() as capture:
            answer = self.multimodal([{"modality": "face", "sample": SAMPLE + "a"}])
        self.assertRefused([answer], SAMPLE_REFUSED % "legs.sample")
        self.assertQuotedNowhere(capture)

    def test_well_formed_arguments_still_enrol_and_verify(self):
        with EveryLogger() as capture:
            enrolled = self.enrol("data:image/jpeg;base64," + SAMPLE)
            verified = self.verify(sample=SAMPLE)
            device = self.verify(modality="voice_device", deviceScore=60, deviceVector=[1, float(VECTOR_DIGITS)],
                                 deviceTemplate=TEMPLATE)
            fused = self.multimodal([{"modality": "face", "sample": SAMPLE, "deviceScore": None},
                                     {"modality": "voice_device", "deviceScore": 60, "position": ""}])
        for answer in (enrolled, verified, device, fused):
            self.assertNotIn("errors", answer)
        self.assertTrue(enrolled["data"]["enrolBiometric"]["id"])
        result = verified["data"]["verifyBiometric"]
        self.assertEqual((result["origin"], result["verified"]), ("server", True))
        self.assertEqual(device["data"]["verifyBiometric"]["origin"], "device")
        self.assertEqual([leg["origin"] for leg in fused["data"]["verifyBiometricMultimodal"]["legs"]],
                         ["server", "device"])
        for sent in (SAMPLE[:24], TEMPLATE[:24], VECTOR_DIGITS):
            self.assertNotIn(sent, capture.text())

    def test_a_client_declaring_the_former_types_gets_an_error_quoting_nothing(self):
        with EveryLogger() as capture:
            answers = [
                self.gql(ENROL % "String!", {"sample": SAMPLE}),
                self.gql(MULTIMODAL % "[BiometricVerifyLegInput!]!",
                         {"legs": [{"modality": "face", "sample": SAMPLE}]}),
                self.gql(VERIFY % ("String", "[Float!]", "String"), {
                    "modality": "face", "sample": SAMPLE, "deviceVector": [float(VECTOR_DIGITS)],
                    "deviceTemplate": TEMPLATE}),
            ]
        for answer in answers:
            self.assertTrue(answer["errors"])
            self.assertNotIn(SAMPLE[:24], json.dumps(answer))
            self.assertNotIn(TEMPLATE[:24], json.dumps(answer))
            self.assertNotIn(VECTOR_DIGITS, json.dumps(answer))
        for sent in (SAMPLE[:24], TEMPLATE[:24], VECTOR_DIGITS):
            self.assertNotIn(sent, capture.text())

    def test_identify_refuses_a_probe_sample_that_is_not_base64_text(self):
        sample_type = _argument_type("query", "identifyBiometric", "sample")
        with EveryLogger() as capture:
            answers = [self.gql(IDENTIFY % sample_type, {"sample": sample}) for sample in ({"data": SAMPLE}, [SAMPLE])]
            answers.append(self.gql('query { identifyBiometric(modality: "face", sample: {data: "%s"}) { subjectId } }'
                                    % SAMPLE))
        self.assertRefused(answers, SAMPLE_REFUSED % "sample")
        self.assertQuotedNowhere(capture)

    def test_identify_ranks_a_well_formed_probe_sample(self):
        self.enrol(SAMPLE)
        answer = self.gql(IDENTIFY % "GenericScalar!", {"sample": "data:image/jpeg;base64," + SAMPLE})
        self.assertNotIn("errors", answer)
        self.assertEqual([match["subjectId"] for match in answer["data"]["identifyBiometric"]], ["s1"])
