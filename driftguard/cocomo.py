"""COCOMO II Post-Architecture effort and schedule estimation.

All constants below are the published COCOMO II.2000 calibration values
(Boehm et al., "Software Cost Estimation with COCOMO II", 2000). They are the
*nominal* published calibration -- this module performs no local calibration
and invents no project data. Any organisation using this for real planning
should re-calibrate constant A against its own completed-project history.

Core equations (Post-Architecture model)
----------------------------------------
    E    = B + 0.01 * sum(SF_j)                for the 5 scale factors
    PM   = A * (Size ** E) * prod(EM_i)        for the 17 effort multipliers
    F    = D + 0.2 * (E - B)
    TDEV = C * (PM_NS ** F) * (SCED% / 100)

where
    A = 2.94, B = 0.91, C = 3.67, D = 0.28
    Size is in KSLOC (thousands of source lines of code)
    PM   is effort in person-months (152 person-hours each, by convention)
    TDEV is calendar duration in months
    PM_NS is effort *excluding* the SCED multiplier ("nominal schedule")

Requirements evolution and volatility (REVL) inflates size before the effort
equation, which is exactly where measured scope drift belongs:

    Size_effective = Size_base * (1 + REVL / 100)

IMPORTANT distinction maintained throughout this module: PM is *effort*
(person-months of work) and TDEV is *duration* (elapsed calendar months).
They are not interchangeable, and PM / TDEV gives average staffing, not a
schedule you can compress at will.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

# --------------------------------------------------------------------------
# Published COCOMO II.2000 calibration constants
# --------------------------------------------------------------------------
A_CONSTANT = 2.94
B_CONSTANT = 0.91
C_CONSTANT = 3.67
D_CONSTANT = 0.28

#: Person-hours in one person-month, by COCOMO convention.
HOURS_PER_PERSON_MONTH = 152

# --------------------------------------------------------------------------
# Scale factors (SF) -- Table: COCOMO II.2000 scale factor values.
# Lower value = less effort. Summed, then folded into the exponent E.
# --------------------------------------------------------------------------
SCALE_FACTORS: dict[str, dict[str, float]] = {
    # Precedentedness: how similar this project is to ones already done
    "PREC": {"VL": 6.20, "L": 4.96, "N": 3.72, "H": 2.48, "VH": 1.24, "XH": 0.00},
    # Development flexibility: freedom from pre-set requirements/process
    "FLEX": {"VL": 5.07, "L": 4.05, "N": 3.04, "H": 2.03, "VH": 1.01, "XH": 0.00},
    # Architecture / risk resolution
    "RESL": {"VL": 7.07, "L": 5.65, "N": 4.24, "H": 2.83, "VH": 1.41, "XH": 0.00},
    # Team cohesion
    "TEAM": {"VL": 5.48, "L": 4.38, "N": 3.29, "H": 2.19, "VH": 1.10, "XH": 0.00},
    # Process maturity (derived from CMM/CMMI level)
    "PMAT": {"VL": 7.80, "L": 6.24, "N": 4.68, "H": 3.12, "VH": 1.56, "XH": 0.00},
}

# --------------------------------------------------------------------------
# Effort multipliers (EM) -- Table: COCOMO II.2000 Post-Architecture EM values.
# A rating of "N" (nominal) is 1.00 for every multiplier by construction.
# Blank cells in the published table (rating not defined for that driver) are
# simply absent from these dicts and will raise on use.
# --------------------------------------------------------------------------
EFFORT_MULTIPLIERS: dict[str, dict[str, float]] = {
    # --- Product factors ---
    "RELY": {"VL": 0.82, "L": 0.92, "N": 1.00, "H": 1.10, "VH": 1.26},
    "DATA": {"L": 0.90, "N": 1.00, "H": 1.14, "VH": 1.28},
    "CPLX": {"VL": 0.73, "L": 0.87, "N": 1.00, "H": 1.17, "VH": 1.34, "XH": 1.74},
    "RUSE": {"L": 0.95, "N": 1.00, "H": 1.07, "VH": 1.15, "XH": 1.24},
    "DOCU": {"VL": 0.81, "L": 0.91, "N": 1.00, "H": 1.11, "VH": 1.23},
    # --- Platform factors ---
    "TIME": {"N": 1.00, "H": 1.11, "VH": 1.29, "XH": 1.63},
    "STOR": {"N": 1.00, "H": 1.05, "VH": 1.17, "XH": 1.46},
    "PVOL": {"L": 0.87, "N": 1.00, "H": 1.15, "VH": 1.30},
    # --- Personnel factors ---
    "ACAP": {"VL": 1.42, "L": 1.19, "N": 1.00, "H": 0.85, "VH": 0.71},
    "PCAP": {"VL": 1.34, "L": 1.15, "N": 1.00, "H": 0.88, "VH": 0.76},
    "PCON": {"VL": 1.29, "L": 1.12, "N": 1.00, "H": 0.90, "VH": 0.81},
    "APEX": {"VL": 1.22, "L": 1.10, "N": 1.00, "H": 0.88, "VH": 0.81},
    "PLEX": {"VL": 1.19, "L": 1.09, "N": 1.00, "H": 0.91, "VH": 0.85},
    "LTEX": {"VL": 1.20, "L": 1.09, "N": 1.00, "H": 0.91, "VH": 0.84},
    # --- Project factors ---
    "TOOL": {"VL": 1.17, "L": 1.09, "N": 1.00, "H": 0.90, "VH": 0.78},
    "SITE": {"VL": 1.22, "L": 1.09, "N": 1.00, "H": 0.93, "VH": 0.86, "XH": 0.80},
    # SCED is applied to PM but EXCLUDED from PM_NS used for TDEV.
    # Note COCOMO II gives no effort penalty for *stretching* a schedule.
    "SCED": {"VL": 1.43, "L": 1.14, "N": 1.00, "H": 1.00, "VH": 1.00},
}

#: SCED has a second, distinct table: the percentage of the nominal schedule
#: the project is being asked to hit. This is what scales TDEV, and it is NOT
#: the same as the SCED effort multiplier above -- compressing a schedule
#: raises effort (EM 1.43) while shortening the calendar (75% of nominal).
SCED_SCHEDULE_PERCENT: dict[str, float] = {
    "VL": 75.0,    # 75% of nominal -- heavily compressed
    "L": 85.0,
    "N": 100.0,
    "H": 130.0,
    "VH": 160.0,   # stretched out
}

#: Human-readable names, used by the dashboard so a reviewer can see what each
#: driver actually means rather than a four-letter code.
DRIVER_NAMES: dict[str, str] = {
    "PREC": "Precedentedness",
    "FLEX": "Development flexibility",
    "RESL": "Architecture / risk resolution",
    "TEAM": "Team cohesion",
    "PMAT": "Process maturity",
    "RELY": "Required software reliability",
    "DATA": "Database size",
    "CPLX": "Product complexity",
    "RUSE": "Required reusability",
    "DOCU": "Documentation match to life-cycle needs",
    "TIME": "Execution time constraint",
    "STOR": "Main storage constraint",
    "PVOL": "Platform volatility",
    "ACAP": "Analyst capability",
    "PCAP": "Programmer capability",
    "PCON": "Personnel continuity",
    "APEX": "Applications experience",
    "PLEX": "Platform experience",
    "LTEX": "Language and tool experience",
    "TOOL": "Use of software tools",
    "SITE": "Multisite development",
    "SCED": "Required development schedule",
}

RATING_LABELS: dict[str, str] = {
    "VL": "Very Low",
    "L": "Low",
    "N": "Nominal",
    "H": "High",
    "VH": "Very High",
    "XH": "Extra High",
}


def nominal_ratings() -> dict[str, str]:
    """Every scale factor and effort multiplier set to Nominal.

    This is the explicit default used by the dashboard. Nominal is an
    *assumption*, not a measurement: with all drivers nominal the model reduces
    to PM = 2.94 * Size^1.0997.
    """
    return {name: "N" for name in list(SCALE_FACTORS) + list(EFFORT_MULTIPLIERS)}


# --------------------------------------------------------------------------
# Size conversion
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SizeConversion:
    """Explicit, auditable rules for turning ticket sizes into KSLOC.

    DriftGuard never guesses these. The defaults below are *placeholders* that
    the user is expected to replace with figures from their own organisation's
    history; the dashboard labels them as unvalidated assumptions.

    Attributes
    ----------
    sloc_per_story_point:
        Source lines of code produced per story point. Highly team-specific.
    sloc_per_function_point:
        Language gearing ratio, used only when a ticket is sized in function
        points. The default (53) is the commonly cited figure for third
        generation languages; for Python a value nearer 20-30 is more usual.
    """

    sloc_per_story_point: float = 60.0
    sloc_per_function_point: float = 53.0

    def to_ksloc(self, size: float, unit: str = "story_points") -> float:
        """Convert a size in `unit` to KSLOC."""
        if size < 0:
            raise ValueError("Size cannot be negative; use the removal class instead.")
        unit = unit.lower().strip()
        if unit in {"story_points", "story_point", "sp", "points"}:
            sloc = size * self.sloc_per_story_point
        elif unit in {"function_points", "function_point", "fp"}:
            sloc = size * self.sloc_per_function_point
        elif unit in {"sloc", "loc"}:
            sloc = size
        elif unit in {"ksloc"}:
            return float(size)
        else:
            raise ValueError(
                f"Unknown size unit {unit!r}. Expected one of: story_points, "
                "function_points, sloc, ksloc."
            )
        return sloc / 1000.0


# --------------------------------------------------------------------------
# Driver settings
# --------------------------------------------------------------------------
@dataclass
class CocomoDrivers:
    """A full set of COCOMO II driver ratings plus REVL.

    Parameters
    ----------
    ratings:
        Mapping of driver code -> rating code ("VL".."XH"). Missing drivers
        default to Nominal.
    revl_percent:
        Requirements evolution and volatility, as a percentage of base size.
        DriftGuard can derive this from measured scope drift.
    """

    ratings: dict[str, str] = field(default_factory=nominal_ratings)
    revl_percent: float = 0.0

    def __post_init__(self) -> None:
        merged = nominal_ratings()
        merged.update({k: v for k, v in self.ratings.items()})
        self.ratings = merged
        self.validate()

    def validate(self) -> None:
        for code, rating in self.ratings.items():
            table = SCALE_FACTORS.get(code) or EFFORT_MULTIPLIERS.get(code)
            if table is None:
                raise ValueError(f"Unknown COCOMO driver {code!r}.")
            if rating not in table:
                allowed = ", ".join(table)
                raise ValueError(
                    f"Rating {rating!r} is not defined for driver {code!r}. "
                    f"Allowed ratings: {allowed}."
                )
        if self.revl_percent < 0:
            raise ValueError("REVL percent cannot be negative.")

    # -- derived quantities -------------------------------------------------
    def scale_factor_sum(self) -> float:
        return sum(SCALE_FACTORS[c][self.ratings[c]] for c in SCALE_FACTORS)

    def exponent_e(self) -> float:
        """E = B + 0.01 * sum(SF)."""
        return B_CONSTANT + 0.01 * self.scale_factor_sum()

    def exponent_f(self) -> float:
        """F = D + 0.2 * (E - B)."""
        return D_CONSTANT + 0.2 * (self.exponent_e() - B_CONSTANT)

    def effort_multiplier_product(self, *, include_sced: bool = True) -> float:
        product = 1.0
        for code in EFFORT_MULTIPLIERS:
            if code == "SCED" and not include_sced:
                continue
            product *= EFFORT_MULTIPLIERS[code][self.ratings[code]]
        return product


# --------------------------------------------------------------------------
# Estimate
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CocomoEstimate:
    """Result of one COCOMO II Post-Architecture run."""

    size_ksloc: float            # size before REVL inflation
    effective_ksloc: float       # size after REVL inflation
    exponent_e: float
    exponent_f: float
    em_product: float            # includes SCED
    em_product_ns: float         # excludes SCED (nominal schedule)
    person_months: float         # PM -- effort, not duration
    person_months_nominal_schedule: float  # PM_NS
    duration_months: float       # TDEV -- calendar duration, not effort

    @property
    def average_staff(self) -> float:
        """PM / TDEV -- average full-time-equivalent staff implied."""
        if self.duration_months == 0:
            return 0.0
        return self.person_months / self.duration_months

    @property
    def person_hours(self) -> float:
        return self.person_months * HOURS_PER_PERSON_MONTH

    def labour_cost(self, monthly_rate: float) -> float:
        """Effort cost at a supplied monthly labour rate.

        This is a *derived estimate*, not an actual or committed cost, and it
        covers labour only -- no tooling, licences, overhead, or contingency.
        Callers must supply the rate; DriftGuard never assumes one.
        """
        if monthly_rate < 0:
            raise ValueError("Monthly labour rate cannot be negative.")
        return self.person_months * monthly_rate

    def as_dict(self) -> dict[str, float]:
        return {
            "size_ksloc": self.size_ksloc,
            "effective_ksloc": self.effective_ksloc,
            "exponent_e": self.exponent_e,
            "exponent_f": self.exponent_f,
            "em_product": self.em_product,
            "person_months": self.person_months,
            "duration_months": self.duration_months,
            "average_staff": self.average_staff,
        }


def estimate(size_ksloc: float, drivers: CocomoDrivers | None = None) -> CocomoEstimate:
    """Run the COCOMO II Post-Architecture model.

    Parameters
    ----------
    size_ksloc:
        Base size in thousands of source lines of code, before REVL.
    drivers:
        Driver ratings and REVL. Defaults to all-Nominal with REVL = 0.

    Returns
    -------
    CocomoEstimate with effort (person-months) and duration (calendar months)
    reported separately.
    """
    if size_ksloc < 0:
        raise ValueError("Size in KSLOC cannot be negative.")
    drivers = drivers or CocomoDrivers()

    effective = size_ksloc * (1.0 + drivers.revl_percent / 100.0)
    e = drivers.exponent_e()
    f = drivers.exponent_f()
    em = drivers.effort_multiplier_product(include_sced=True)
    em_ns = drivers.effort_multiplier_product(include_sced=False)

    base = A_CONSTANT * (effective ** e) if effective > 0 else 0.0
    pm = base * em
    pm_ns = base * em_ns

    sced_pct = SCED_SCHEDULE_PERCENT[drivers.ratings["SCED"]] / 100.0
    tdev = C_CONSTANT * (pm_ns ** f) * sced_pct if pm_ns > 0 else 0.0

    return CocomoEstimate(
        size_ksloc=size_ksloc,
        effective_ksloc=effective,
        exponent_e=e,
        exponent_f=f,
        em_product=em,
        em_product_ns=em_ns,
        person_months=pm,
        person_months_nominal_schedule=pm_ns,
        duration_months=tdev,
    )


@dataclass(frozen=True)
class EstimateComparison:
    """Baseline vs revised estimate, with deltas."""

    baseline: CocomoEstimate
    revised: CocomoEstimate

    @property
    def delta_ksloc(self) -> float:
        return self.revised.size_ksloc - self.baseline.size_ksloc

    @property
    def delta_person_months(self) -> float:
        return self.revised.person_months - self.baseline.person_months

    @property
    def delta_duration_months(self) -> float:
        return self.revised.duration_months - self.baseline.duration_months

    def _pct(self, new: float, old: float) -> float:
        if old == 0:
            return 0.0
        return (new - old) / old * 100.0

    @property
    def effort_change_percent(self) -> float:
        return self._pct(self.revised.person_months, self.baseline.person_months)

    @property
    def duration_change_percent(self) -> float:
        return self._pct(self.revised.duration_months, self.baseline.duration_months)

    @property
    def size_change_percent(self) -> float:
        return self._pct(self.revised.size_ksloc, self.baseline.size_ksloc)


def compare(
    baseline_ksloc: float,
    revised_ksloc: float,
    drivers: CocomoDrivers | None = None,
    revised_drivers: CocomoDrivers | None = None,
) -> EstimateComparison:
    """Estimate baseline and revised sizes and return the comparison.

    `revised_drivers` lets the revised run carry a different REVL (the usual
    case: the baseline plan assumed little volatility, the revised one reflects
    the drift actually observed).
    """
    drivers = drivers or CocomoDrivers()
    return EstimateComparison(
        baseline=estimate(baseline_ksloc, drivers),
        revised=estimate(revised_ksloc, revised_drivers or drivers),
    )


def describe_drivers(drivers: CocomoDrivers) -> list[dict[str, object]]:
    """Flatten driver settings into rows for display in the dashboard."""
    rows: list[dict[str, object]] = []
    for code, table in SCALE_FACTORS.items():
        rating = drivers.ratings[code]
        rows.append(
            {
                "Code": code,
                "Driver": DRIVER_NAMES[code],
                "Kind": "Scale factor",
                "Rating": RATING_LABELS[rating],
                "Value": table[rating],
            }
        )
    for code, table in EFFORT_MULTIPLIERS.items():
        rating = drivers.ratings[code]
        rows.append(
            {
                "Code": code,
                "Driver": DRIVER_NAMES[code],
                "Kind": "Effort multiplier",
                "Rating": RATING_LABELS[rating],
                "Value": table[rating],
            }
        )
    return rows


def ratings_from_mapping(mapping: Mapping[str, str]) -> CocomoDrivers:
    """Build drivers from a plain mapping, validating as we go."""
    return CocomoDrivers(ratings=dict(mapping))
