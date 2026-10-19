"""Evidence-based claim verification."""

from modules.verification.models import Verdict, VerificationResult
from modules.verification.service import VerificationService
from modules.verification.verifier import RuleBasedVerifier

__all__ = ["RuleBasedVerifier", "Verdict", "VerificationResult", "VerificationService"]
