"""Shared constants, class labels and tunable assumptions for DriftGuard.

Everything in this module that is an *assumption* rather than a definition is
marked as such, because the honesty of the whole tool rests on the user being
able to see and change these numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# Classification taxonomy
# --------------------------------------------------------------------------
IN_SCOPE = "In-scope"
REFINEMENT = "Refinement"
NEW_SCOPE = "New scope"
REMOVAL = "Removal"

CLASSES: tuple[str, ...] = (IN_SCOPE, REFINEMENT, NEW_SCOPE, REMOVAL)

CLASS_DEFINITIONS: dict[str, str] = {
    IN_SCOPE: (
        "Implements a baseline requirement as already agreed. Contributes no "
        "scope change."
    ),
    REFINEMENT: (
        "Modifies, clarifies or elaborates an existing baseline requirement. "
        "Contributes partial added scope, because refinement usually carries "
        "incremental work without being a wholly new requirement."
    ),
    NEW_SCOPE: (
        "Introduces work with no corresponding baseline requirement. "
        "Contributes its full size as added scope."
    ),
    REMOVAL: (
        "Withdraws, de-scopes or cancels agreed baseline work. Contributes its "
        "full size as removed scope."
    ),
}

#: Categorical palette for the four classes.
#:
#: Validated, not chosen by eye. These four hues clear every hard gate of the
#: data-visualisation colour checks on the all-pairs list against the light
#: chart surface (#fcfcfb): lightness band, chroma floor, colour-vision-
#: deficiency separation (worst pair dE 9.1 protan / 18.0 tritan, target >= 8)
#: and the normal-vision floor (worst pair dE 22.9, floor 15).
#:
#: Two of them -- In-scope and Refinement -- sit below 3:1 contrast against the
#: surface. That is permitted only under the "relief" rule, which obliges every
#: chart using them to carry visible direct labels or an accompanying table
#: view. The dashboard honours this; do not drop those labels.
#:
#: Blue is deliberately absent: it is reserved for the accent colour and the
#: sequential ramp (for example the confusion matrix), so that no magnitude
#: encoding can be mistaken for a class identity.
CLASS_COLOURS: dict[str, str] = {
    IN_SCOPE: "#1baf7a",      # aqua-green
    REFINEMENT: "#eda100",    # yellow
    NEW_SCOPE: "#C62828",     # red
    REMOVAL: "#6250d6",       # violet
}

# --------------------------------------------------------------------------
# Matching / classification thresholds
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchingThresholds:
    """Cosine-similarity cut-offs for the rule-based classifier.

    ASSUMPTION. These values were tuned on the bundled sample project and are
    not derived from any published study. Because they were fitted on the same
    data the evaluation page scores, the headline accuracy there is optimistic.
    Re-tune them on your own labelled tickets before drawing conclusions.

    ``weak_match`` separates "related to the baseline at all" from "unrelated".
    ``strong_match`` separates a confident match from one a reviewer should
    look at; it does not by itself decide the class.
    """

    strong_match: float = 0.45
    weak_match: float = 0.18

    def __post_init__(self) -> None:
        if not 0.0 <= self.weak_match <= self.strong_match <= 1.0:
            raise ValueError(
                "Thresholds must satisfy 0 <= weak_match <= strong_match <= 1."
            )


# --------------------------------------------------------------------------
# Drift accounting assumptions
# --------------------------------------------------------------------------

#: ASSUMPTION. Fraction of a Refinement ticket's size that counts as *added*
#: scope. 0.0 would treat refinement as free; 1.0 would treat it as entirely
#: new work. 0.5 is a deliberate midpoint and is exposed in the dashboard.
DEFAULT_REFINEMENT_WEIGHT = 0.5

#: Keyword cues used by the explainable rule-based classifier. Deliberately
#: small and visible rather than a hidden model: the reviewer can read them.
#:
#: Removal cues are split in two because the obvious verbs collide badly with
#: ordinary domain vocabulary. "Administrators can withdraw a course section"
#: and "notify the student when an enrolment is withdrawn" are both ordinary
#: in-scope work, not de-scoping. So the generic verbs only count as removal
#: evidence when a scope noun appears alongside them.
STRONG_REMOVAL_CUES: tuple[str, ...] = (
    "de-scope",
    "descope",
    "out of scope",
    "no longer required",
    "no longer needed",
    "will not implement",
    "will not be implemented",
    "wontfix",
    "won't fix",
)

#: Generic withdrawal verbs. Count only with a scope noun in the same ticket.
CONTEXTUAL_REMOVAL_CUES: tuple[str, ...] = (
    "remove",
    "removed",
    "removal",
    "drop",
    "dropped",
    "delete",
    "deleted",
    "cancel",
    "cancelled",
    "canceled",
    "withdraw",
    "withdrawn",
    "deprecate",
    "deprecated",
)

#: Nouns that make a generic withdrawal verb a statement about *scope*.
SCOPE_CONTEXT_NOUNS: tuple[str, ...] = (
    "scope",
    "requirement",
    "requirements",
    "baseline",
    "backlog",
    "from the project",
    "from this project",
)

#: Modification language. Deliberately excludes bare "change", "add" and
#: "in addition": all three fire on ordinary in-scope and new-scope tickets in
#: this domain ("notify on enrolment change", "add keyboard navigation"), and
#: a cue that fires on three classes carries no information.
REFINEMENT_CUES: tuple[str, ...] = (
    "clarify",
    "clarification",
    "refine",
    "adjust",
    "amend",
    "revise",
    "update",
    "modify",
    "tweak",
    "extend",
    "expand",
    "broaden",
    "tighten",
    "instead of",
    "rather than",
    "replacing",
    "in place of",
)

NEW_SCOPE_CUES: tuple[str, ...] = (
    "new feature",
    "net new",
    "brand new",
    "introduce",
    "add support for",
    "stakeholder request",
    "additional module",
    "not in the original",
    "not part of the agreed",
    "was never specified",
    "out of the agreed",
    "newly requested",
)

#: English stop-word handling is delegated to scikit-learn. DriftGuard is
#: scoped to English-language tickets only; see README limitations.
TEXT_LANGUAGE = "english"
