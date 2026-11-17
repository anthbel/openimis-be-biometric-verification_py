import copy
import logging
import os
from datetime import datetime
from numbers import Number

from django.apps import AppConfig
from django.core.signals import request_started

logger = logging.getLogger(__name__)

MODULE_NAME = "biometric"

# Environment variable holding the Fernet key when no configuration gives one.
TEMPLATE_KEY_ENV = "BIOMETRIC_TEMPLATE_KEY"

DEFAULT_CFG = {
    "subject_model": "individual.Individual",
    "modalities": {
        "face": {"provider": "deepface", "threshold": 0.32},
        "fingerprint": {"provider": "device_reported", "threshold": 48},
    },
    "vector_index": "numpy",  # "numpy" | "pgvector" (pgvector requires the biometric_pgvector app)
    # Fernet key; templates/vectors encrypted at rest when set. Unset or blank:
    # read from BIOMETRIC_TEMPLATE_KEY (resolve_config).
    "template_key": None,
    "require_consent": False,
    "max_top_k": 50,          # identifyBiometric clamps topK to 1..max_top_k
    "dedup_threshold": {"face": 0.62},  # similarity at/above which a candidate is emitted
    # Modalities that get a candidate source; each needs a dedup_threshold entry.
    # The face source is the only one registered.
    "dedup_modalities": ["face"],
    "fusion": {
        "weights": {"face": 1.0},
        "thresholds": {"accept": 0.7, "review": 0.6},
        "floors": {},
        "floor_decision": "review",
    },
    # Enrolment quality gate (biometric/quality.py): "advisory" stores the
    # verdict only, "enforce" refuses a REFUSED sample. Nested keys are lowercase.
    "quality": {
        "mode": "advisory",
        "modalities": {
            "face": {
                "min_sharpness": 100.0,
                "max_yaw": 20.0,
                "max_pitch": None,
                "max_roll": 20.0,
                "max_yaw_ratio": None,
                "min_lower_face_uniformity": None,
                "min_quality": None,
            },
        },
    },
    # Named tighten-only overrides of FUSION / MODALITIES thresholds
    # (biometric/risk_profiles.py).
    "risk_profiles": {},
    # 1:N impersonation probe inside verify() (biometric/impersonation.py).
    # Nested keys are lowercase; missing keys take PROBE_DEFAULTS at read time.
    "impersonation_probe": {
        "enabled": False, "modalities": ["face"], "top_k": 5, "thresholds": {}, "margin": None, "device_path": False,
    },
    # Hash-chained audit events and alert rules (biometric/audit_chain.py,
    # biometric/audit_rules.py). Nested keys are lowercase; "rules" overrides
    # audit_rules.DEFAULT_RULES per kind, key by key.
    "audit": {"enabled": False, "rules": {}},
    "gql_biometric_enrol_perms": ["174001"],
    "gql_biometric_verify_perms": ["174002"],
    "gql_biometric_identify_perms": ["174003"],
    "gql_biometric_read_perms": ["174004"],
    "gql_biometric_audit_perms": ["174005"],   # read audit events and alerts
    "gql_biometric_alert_perms": ["174006"],   # acknowledge and resolve alerts
    "gql_biometric_config_perms": ["174007"],  # read decision criteria and the retention policy
    "gql_biometric_audit_verify_perms": ["174008"],  # run the audit chain verification (with 174005)
}

# Maps uppercase Django settings keys -> lowercase ModuleConfiguration keys.
_SETTINGS_KEY_MAP = {
    "SUBJECT_MODEL": "subject_model",
    "MODALITIES": "modalities",
    "VECTOR_INDEX": "vector_index",
    "TEMPLATE_KEY": "template_key",
    "REQUIRE_CONSENT": "require_consent",
    "MAX_TOP_K": "max_top_k",
    "DEDUP_THRESHOLD": "dedup_threshold",
    "DEDUP_MODALITIES": "dedup_modalities",
    "FUSION": "fusion",
    "QUALITY": "quality",
    "RISK_PROFILES": "risk_profiles",
    "IMPERSONATION_PROBE": "impersonation_probe",
    "AUDIT": "audit",
    "GQL_BIOMETRIC_ENROL_PERMS": "gql_biometric_enrol_perms",
    "GQL_BIOMETRIC_VERIFY_PERMS": "gql_biometric_verify_perms",
    "GQL_BIOMETRIC_IDENTIFY_PERMS": "gql_biometric_identify_perms",
    "GQL_BIOMETRIC_READ_PERMS": "gql_biometric_read_perms",
    "GQL_BIOMETRIC_AUDIT_PERMS": "gql_biometric_audit_perms",
    "GQL_BIOMETRIC_ALERT_PERMS": "gql_biometric_alert_perms",
    "GQL_BIOMETRIC_CONFIG_PERMS": "gql_biometric_config_perms",
    "GQL_BIOMETRIC_AUDIT_VERIFY_PERMS": "gql_biometric_audit_verify_perms",
}


def _given_key(value):
    """The key as given, or None for a missing, non-text or blank value."""
    if isinstance(value, (str, bytes)):
        value = value.strip()
        return value or None
    return None


def resolve_config(module_cfg=None) -> dict:
    """
    The effective configuration: DEFAULT_CFG merged with module_cfg (the
    stored `biometric` module configuration when None), then settings.BIOMETRIC
    (upper or lower case keys), which wins. template_key is the first non-blank
    value of settings.BIOMETRIC, the module configuration and the
    BIOMETRIC_TEMPLATE_KEY environment variable; None when none gives one.
    """
    from django.conf import settings

    if module_cfg is None:
        from core.models import ModuleConfiguration

        cfg = ModuleConfiguration.get_or_default(MODULE_NAME, DEFAULT_CFG)
    else:
        cfg = {**copy.deepcopy(DEFAULT_CFG), **copy.deepcopy(dict(module_cfg))}

    key = _given_key(cfg.get("template_key"))
    for name, value in (getattr(settings, "BIOMETRIC", None) or {}).items():
        normalised = _SETTINGS_KEY_MAP.get(name.upper(), name.lower())
        if normalised == "template_key":
            key = _given_key(value) or key
        else:
            cfg[normalised] = copy.deepcopy(value)
    cfg["template_key"] = key or _given_key(os.environ.get(TEMPLATE_KEY_ENV))
    return cfg


def _is_number(value):
    return isinstance(value, Number) and not isinstance(value, bool)


def config_errors(cfg) -> list:
    """
    The deduplication settings of cfg that cannot be used, one message per
    problem. Messages name the key and the modality, never a configured value.
    """
    errors = []
    thresholds = cfg.get("dedup_threshold")
    if not isinstance(thresholds, dict):
        errors.append("dedup_threshold must be a mapping of modality to a number")
        thresholds = {}
    for modality, value in thresholds.items():
        if not _is_number(value):
            errors.append(f"dedup_threshold[{str(modality)!r}] is not a number")
    modalities = cfg.get("modalities") if isinstance(cfg.get("modalities"), dict) else {}
    dedup_modalities = cfg.get("dedup_modalities")
    if not isinstance(dedup_modalities, list):
        errors.append("dedup_modalities must be a list of modality names")
        dedup_modalities = []
    for modality in dedup_modalities:
        if not isinstance(modality, str):
            errors.append("dedup_modalities holds an entry that is not a modality name")
            continue
        if modality not in modalities:
            errors.append(f"dedup_modalities entry {modality!r} is not a key of modalities")
        if not _is_number(thresholds.get(modality)):
            errors.append(f"dedup_modalities entry {modality!r} has no numeric dedup_threshold entry")
    return errors


def validate_module_config(instance):
    """core.module_config_registry validator: refuses a stored configuration whose effective one has config_errors."""
    from django.core.exceptions import ValidationError

    if not isinstance(instance._cfg, dict):
        raise ValidationError({"config": [f"{MODULE_NAME}: the configuration must be a JSON object"]})
    errors = config_errors(resolve_config(instance._cfg))
    if errors:
        raise ValidationError({"config": [f"{MODULE_NAME}: {message}" for message in errors]})


class BiometricConfig(AppConfig):
    name = MODULE_NAME

    subject_model = "individual.Individual"
    modalities = {
        "face": {"provider": "deepface", "threshold": 0.32},
        "fingerprint": {"provider": "device_reported", "threshold": 48},
    }
    vector_index = "numpy"
    template_key = None
    require_consent = False
    max_top_k = 50
    dedup_threshold = {"face": 0.62}
    dedup_modalities = ["face"]
    fusion = {
        "weights": {"face": 1.0},
        "thresholds": {"accept": 0.7, "review": 0.6},
        "floors": {},
        "floor_decision": "review",
    }
    quality = {
        "mode": "advisory",
        "modalities": {
            "face": {
                "min_sharpness": 100.0,
                "max_yaw": 20.0,
                "max_pitch": None,
                "max_roll": 20.0,
                "max_yaw_ratio": None,
                "min_lower_face_uniformity": None,
                "min_quality": None,
            },
        },
    }
    # name -> partial override of thresholds / floors / floor_decision /
    # required / modality_thresholds; each value may only equal or tighten
    # the base in fusion and modalities.
    risk_profiles = {}
    impersonation_probe = {
        "enabled": False, "modalities": ["face"], "top_k": 5, "thresholds": {}, "margin": None, "device_path": False,
    }
    audit = {"enabled": False, "rules": {}}
    gql_biometric_enrol_perms = ["174001"]
    gql_biometric_verify_perms = ["174002"]
    gql_biometric_identify_perms = ["174003"]
    gql_biometric_read_perms = ["174004"]
    gql_biometric_audit_perms = ["174005"]
    gql_biometric_alert_perms = ["174006"]
    gql_biometric_config_perms = ["174007"]
    gql_biometric_audit_verify_perms = ["174008"]

    # The resolved configuration last applied, and the last one refused by
    # config_errors; refresh_config() compares against these, not against the
    # class attributes.
    _applied_config = None
    _rejected_config = None

    @classmethod
    def _apply(cls, cfg):
        for field, value in cfg.items():
            if hasattr(BiometricConfig, field):
                setattr(BiometricConfig, field, value)
        BiometricConfig._applied_config = cfg
        cls._check_quality_config()
        cls._check_risk_profiles()
        cls._check_audit_config()

    def ready(self):
        from core.module_config_registry import register_validator

        # 1. Module configuration (or defaults), then settings.BIOMETRIC, then
        #    the template key from the environment (resolve_config).
        cfg = resolve_config()
        for message in config_errors(cfg):
            logger.error("BIOMETRIC: %s", message)
        self._apply(cfg)

        # 2. Validate stored changes, and apply them per request.
        register_validator(MODULE_NAME, validate_module_config)
        request_started.connect(_on_request_started, weak=False, dispatch_uid=f"{MODULE_NAME}_config_refresh")

        # 3. Bind the deduplication.subject_merged -> consolidate() signal.
        from . import signals  # noqa: F401

        # 4. Register this module's candidate source with the deduplication
        #    module, if installed (docs/wb-biometric-dedup-seam.md §2.1).
        self._register_candidate_source()

    @classmethod
    def refresh_config(cls):
        """
        Applies the stored module configuration (resolved as resolve_config does)
        when it differs from the one last applied: one query, no reload when
        unchanged. A configuration refused by config_errors is logged by key,
        once, and the last applied one stays in use.
        """
        from django.db.models import Q

        from core.models import ModuleConfiguration

        try:
            row = ModuleConfiguration.objects.filter(
                Q(is_disabled_until=None) | Q(is_disabled_until__lt=datetime.now()),
                layer="be", module=MODULE_NAME,
            ).first()
            cfg = resolve_config(row._cfg if row else {})
        except Exception as exc:
            logger.error("%s: stored configuration not read, the one in use stays (%s)", MODULE_NAME,
                         type(exc).__name__)
            return
        if cfg == BiometricConfig._applied_config:
            BiometricConfig._rejected_config = None
            return
        if cfg == BiometricConfig._rejected_config:
            return
        errors = config_errors(cfg)
        if errors:
            BiometricConfig._rejected_config = cfg
            logger.error("%s: stored configuration not applied, the one in use stays: %s", MODULE_NAME,
                         "; ".join(errors))
            return
        cls._apply(cfg)
        BiometricConfig._rejected_config = None

    @staticmethod
    def _check_quality_config():
        """Logs an unusable quality gate config; enrol() raises on it through quality.mode()."""
        import logging

        from .quality import MODES, _warn_if_pillow_missing_in_enforce

        quality = BiometricConfig.quality if isinstance(BiometricConfig.quality, dict) else {}
        if quality.get("mode", "advisory") not in MODES:
            logging.getLogger(__name__).error(
                "BIOMETRIC['QUALITY']['mode'] is %r; expected one of %s. enrol() refuses to run until it is fixed.",
                quality.get("mode"), MODES,
            )
        _warn_if_pillow_missing_in_enforce()

    @staticmethod
    def _check_risk_profiles():
        """Logs every invalid risk profile; risk_profiles.resolve() raises when one is used."""
        from .risk_profiles import validate_profiles

        for message in validate_profiles():
            logger.error("BIOMETRIC['RISK_PROFILES']: %s", message)

    @staticmethod
    def _check_audit_config():
        """Logs every problem in BIOMETRIC['AUDIT']; audit_rules.rule_params() raises when a bad rule is read."""
        from .audit_rules import validate_audit_config

        for message in validate_audit_config(BiometricConfig.audit):
            logger.error("BIOMETRIC['AUDIT']: %s", message)

    @staticmethod
    def _register_candidate_source():
        try:
            from deduplication.sources import register
        except ImportError:
            return
        from .dedup_source import BiometricCandidateSource
        register(BiometricCandidateSource())


def _on_request_started(sender=None, **kwargs):
    BiometricConfig.refresh_config()
