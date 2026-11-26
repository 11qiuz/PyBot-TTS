"""Pluggable PyBot acoustic rule modules (`exp_20_pybot_larynx.rules`)."""
from exp_20_pybot_larynx.arbiter import RuleModule
from exp_20_pybot_larynx.rules.coarticulation import RuleBoundaryCoarticulation
from exp_20_pybot_larynx.rules.formant_dynamics import RuleFormantDynamics
from exp_20_pybot_larynx.rules.glottal_coupling import RuleGlottalF0Coupling
from exp_20_pybot_larynx.rules.nasals import RuleNasalAntiresonance
from exp_20_pybot_larynx.rules.obstruents import RuleObstruentCrispness
from exp_20_pybot_larynx.rules.prosodic_energy import RuleProsodicEnergyTilt


def default_stage0_rules() -> list[RuleModule]:
    """Return the initial 4 orthogonal PyBot expert rule modules for Stage 0."""
    return [
        RuleGlottalF0Coupling(),
        RuleObstruentCrispness(),
        RuleNasalAntiresonance(),
        RuleBoundaryCoarticulation(),
    ]


def default_stage5_rules() -> list[RuleModule]:
    """Return all 6 orthogonal PyBot expert rule modules for Stage 5."""
    return [
        RuleGlottalF0Coupling(),
        RuleObstruentCrispness(),
        RuleNasalAntiresonance(),
        RuleBoundaryCoarticulation(),
        RuleProsodicEnergyTilt(),
        RuleFormantDynamics(),
    ]


__all__ = [
    "RuleBoundaryCoarticulation",
    "RuleFormantDynamics",
    "RuleGlottalF0Coupling",
    "RuleNasalAntiresonance",
    "RuleObstruentCrispness",
    "RuleProsodicEnergyTilt",
    "default_stage0_rules",
    "default_stage5_rules",
]
