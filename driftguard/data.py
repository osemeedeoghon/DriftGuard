"""Loading and validation of baseline requirements and project tickets.

DriftGuard is deliberately strict about input: a silently mis-parsed size or a
dropped requirement would corrupt every number downstream, so every failure
mode here raises a ``DataValidationError`` carrying a message a coursework
marker or a project manager can act on without reading the source.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import pandas as pd

from .config import CLASSES

#: Columns DriftGuard needs from a baseline requirements CSV.
BASELINE_REQUIRED = ("requirement_id", "description")
BASELINE_OPTIONAL = ("title", "size", "category", "priority")

#: Columns DriftGuard needs from a tickets CSV.
TICKET_REQUIRED = ("ticket_id", "description")
TICKET_OPTIONAL = ("title", "size", "created_date", "status", "reporter", "true_class")

#: Accepted spellings for each canonical column, lower-cased and stripped of
#: separators. Real exports rarely agree on naming, and rejecting a file for
#: calling a column "Ticket ID" instead of "ticket_id" is needless friction.
COLUMN_ALIASES: dict[str, str] = {
    "requirementid": "requirement_id",
    "reqid": "requirement_id",
    "id": "requirement_id",
    "requirement": "description",
    "requirementtext": "description",
    "text": "description",
    "body": "description",
    "summary": "title",
    "name": "title",
    "ticketid": "ticket_id",
    "issueid": "ticket_id",
    "issuekey": "ticket_id",
    "key": "ticket_id",
    "storypoints": "size",
    "points": "size",
    "estimate": "size",
    "effort": "size",
    "created": "created_date",
    "createddate": "created_date",
    "createdat": "created_date",
    "date": "created_date",
    "label": "true_class",
    "trueclass": "true_class",
    "goldlabel": "true_class",
    "actualclass": "true_class",
}


class DataValidationError(ValueError):
    """Raised when an uploaded file cannot be used as-is."""


@dataclass(frozen=True)
class LoadReport:
    """What happened during a load, so the UI can be honest about it."""

    rows: int
    warnings: list[str]

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return self.rows > 0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _canonicalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        key = re.sub(r"[^a-z0-9]", "", str(column).lower())
        canonical = COLUMN_ALIASES.get(key, re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_"))
        renamed[column] = canonical
    return frame.rename(columns=renamed)


def _read_csv(source: str | Path | IO[bytes] | IO[str]) -> pd.DataFrame:
    try:
        frame = pd.read_csv(source)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(
            "The file is empty. Expected a CSV with a header row."
        ) from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(
            f"The file could not be parsed as CSV: {exc}. Check for unescaped "
            "commas or quotes inside a description field."
        ) from exc
    except UnicodeDecodeError as exc:
        raise DataValidationError(
            "The file is not valid UTF-8 text. Re-export it as UTF-8 CSV."
        ) from exc
    if frame.empty:
        raise DataValidationError("The file has a header row but no data rows.")
    return _canonicalise_columns(frame)


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], kind: str) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise DataValidationError(
            f"The {kind} file is missing required column(s): "
            f"{', '.join(missing)}. Columns found: "
            f"{', '.join(map(str, frame.columns))}."
        )


def _coerce_size(
    frame: pd.DataFrame, default: float, kind: str, warnings: list[str]
) -> pd.DataFrame:
    if "size" not in frame.columns:
        warnings.append(
            f"No size column in the {kind} file; every row was assigned the "
            f"default size of {default:g}. All drift figures therefore assume "
            "uniform sizing, which is an assumption, not a measurement."
        )
        frame["size"] = float(default)
        return frame

    sizes = pd.to_numeric(frame["size"], errors="coerce")
    blank = sizes.isna()
    if blank.any():
        ids = frame.loc[blank].iloc[:, 0].astype(str).tolist()
        shown = ", ".join(ids[:5]) + ("..." if len(ids) > 5 else "")
        warnings.append(
            f"{int(blank.sum())} {kind} row(s) had a missing or non-numeric "
            f"size and were set to the default {default:g}: {shown}"
        )
        sizes = sizes.fillna(float(default))
    negative = sizes < 0
    if negative.any():
        raise DataValidationError(
            f"{int(negative.sum())} {kind} row(s) have a negative size. Sizes "
            "must be zero or positive; record withdrawn work as a Removal "
            "ticket instead of a negative size."
        )
    frame["size"] = sizes.astype(float)
    return frame


def _dedupe_ids(frame: pd.DataFrame, id_column: str, kind: str, warnings: list[str]) -> pd.DataFrame:
    frame[id_column] = frame[id_column].astype(str).str.strip()
    duplicated = frame[id_column].duplicated(keep="first")
    if duplicated.any():
        ids = sorted(set(frame.loc[duplicated, id_column]))
        warnings.append(
            f"Dropped {int(duplicated.sum())} duplicate {kind} row(s) sharing "
            f"an id with an earlier row: {', '.join(ids[:5])}"
            + ("..." if len(ids) > 5 else "")
        )
        frame = frame.loc[~duplicated].copy()
    return frame


def _flag_synthetic(frame: pd.DataFrame, kind: str, warnings: list[str]) -> None:
    """Warn when a dataset announces itself as synthetic.

    The generator marks its output with a ``synthetic`` column precisely so
    that a file re-uploaded later cannot be mistaken for real project data.
    """
    if "synthetic" not in frame.columns:
        return
    flagged = frame["synthetic"].astype(str).str.lower().isin({"true", "1", "yes"})
    if flagged.any():
        warnings.append(
            f"This {kind} file is marked SYNTHETIC ({int(flagged.sum())} of "
            f"{len(frame)} rows). It was produced by DriftGuard's test-data "
            "generator. Results from it demonstrate behaviour only and are not "
            "evidence about any real project."
        )


def _build_text(frame: pd.DataFrame) -> pd.Series:
    """Combine title and description into the text used for similarity."""
    description = frame["description"].fillna("").astype(str)
    if "title" in frame.columns:
        title = frame["title"].fillna("").astype(str)
        combined = (title + ". " + description).str.strip(". ").str.strip()
    else:
        combined = description.str.strip()
    return combined


# --------------------------------------------------------------------------
# Baseline requirements
# --------------------------------------------------------------------------
def load_baseline_csv(
    source: str | Path | IO[bytes] | IO[str], *, default_size: float = 5.0
) -> tuple[pd.DataFrame, LoadReport]:
    """Load agreed baseline requirements from CSV.

    Required columns: ``requirement_id`` and ``description`` (common aliases
    such as "Req ID" or "Requirement Text" are accepted). ``size`` is optional
    but strongly recommended -- without it, every requirement is assumed to be
    the same size.
    """
    frame = _read_csv(source)
    _require_columns(frame, BASELINE_REQUIRED, "baseline requirements")

    warnings: list[str] = []
    frame = _dedupe_ids(frame, "requirement_id", "requirement", warnings)

    blank_text = frame["description"].fillna("").astype(str).str.strip() == ""
    if blank_text.all():
        raise DataValidationError(
            "Every requirement has an empty description. Similarity matching "
            "has nothing to work with."
        )
    if blank_text.any():
        warnings.append(
            f"{int(blank_text.sum())} requirement(s) have an empty description "
            "and can never be matched by a ticket."
        )

    frame = _coerce_size(frame, default_size, "baseline requirements", warnings)
    _flag_synthetic(frame, "baseline requirements", warnings)
    frame["text"] = _build_text(frame)

    keep = [c for c in (*BASELINE_REQUIRED, *BASELINE_OPTIONAL, "text") if c in frame.columns]
    frame = frame[keep].reset_index(drop=True)
    return frame, LoadReport(rows=len(frame), warnings=warnings)


def load_baseline_txt(
    source: str | Path | IO[bytes] | IO[str], *, default_size: float = 5.0
) -> tuple[pd.DataFrame, LoadReport]:
    """Load baseline requirements from plain text, one requirement per line.

    Blank lines are ignored. A leading list marker ("1.", "-", "*", "R-04:")
    is stripped and, where it looks like an identifier, reused as the
    requirement id. Because a text file carries no sizes, every requirement
    receives ``default_size`` -- this is recorded as a warning, not hidden.
    """
    raw = _read_text(source)
    lines = [line.strip() for line in raw.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise DataValidationError(
            "The text file contains no non-blank lines. Expected one "
            "requirement per line."
        )

    ids: list[str] = []
    texts: list[str] = []
    for index, line in enumerate(lines, start=1):
        explicit = re.match(r"^\s*([A-Za-z]{1,6}[-_ ]?\d{1,4})\s*[:.)-]\s+(.*)$", line)
        bullet = re.match(r"^\s*(?:[-*•]|\d{1,3}[.)])\s+(.*)$", line)
        if explicit:
            ids.append(explicit.group(1).replace(" ", "-").upper())
            texts.append(explicit.group(2).strip())
        elif bullet:
            ids.append(f"R-{index:03d}")
            texts.append(bullet.group(1).strip())
        else:
            ids.append(f"R-{index:03d}")
            texts.append(line)

    frame = pd.DataFrame({"requirement_id": ids, "description": texts})
    warnings = [
        f"Text upload carries no sizes; all {len(frame)} requirements were "
        f"assigned the default size of {default_size:g}. The Scope Drift Index "
        "will therefore treat every baseline requirement as equally large."
    ]
    frame = _dedupe_ids(frame, "requirement_id", "requirement", warnings)
    frame["size"] = float(default_size)
    frame["text"] = frame["description"]
    return frame.reset_index(drop=True), LoadReport(rows=len(frame), warnings=warnings)


def load_baseline(
    source: str | Path | IO[bytes] | IO[str],
    *,
    filename: str | None = None,
    default_size: float = 5.0,
) -> tuple[pd.DataFrame, LoadReport]:
    """Dispatch to the CSV or TXT loader based on the file extension."""
    name = filename or (str(source) if isinstance(source, (str, Path)) else "")
    if name.lower().endswith((".txt", ".md")):
        return load_baseline_txt(source, default_size=default_size)
    return load_baseline_csv(source, default_size=default_size)


def _read_text(source: str | Path | IO[bytes] | IO[str]) -> str:
    if isinstance(source, (str, Path)) and Path(source).exists():
        return Path(source).read_text(encoding="utf-8")
    if hasattr(source, "read"):
        data = source.read()
        if isinstance(data, bytes):
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise DataValidationError(
                    "The text file is not valid UTF-8. Re-save it as UTF-8."
                ) from exc
        return str(data)
    if isinstance(source, str):
        return source
    raise DataValidationError("Could not read the uploaded text file.")


# --------------------------------------------------------------------------
# Tickets
# --------------------------------------------------------------------------
def load_tickets(
    source: str | Path | IO[bytes] | IO[str], *, default_size: float = 3.0
) -> tuple[pd.DataFrame, LoadReport]:
    """Load project tickets from CSV.

    Required columns: ``ticket_id`` and ``description``. ``size`` and
    ``created_date`` are optional but the drift timeline needs dates, and the
    Scope Drift Index is only meaningful with real sizes.

    An optional ``true_class`` column carries manual labels; when present the
    file doubles as the evaluation set.
    """
    frame = _read_csv(source)
    _require_columns(frame, TICKET_REQUIRED, "tickets")

    warnings: list[str] = []
    frame = _dedupe_ids(frame, "ticket_id", "ticket", warnings)

    blank_text = frame["description"].fillna("").astype(str).str.strip() == ""
    if blank_text.all():
        raise DataValidationError(
            "Every ticket has an empty description. There is no text to match "
            "against the baseline."
        )
    if blank_text.any():
        warnings.append(
            f"{int(blank_text.sum())} ticket(s) have an empty description and "
            "will fall back to the lowest-confidence classification."
        )

    frame = _coerce_size(frame, default_size, "tickets", warnings)

    if "created_date" in frame.columns:
        parsed = pd.to_datetime(frame["created_date"], errors="coerce")
        unparsed = parsed.isna() & frame["created_date"].notna()
        if unparsed.any():
            warnings.append(
                f"{int(unparsed.sum())} ticket(s) have an unrecognised date "
                "and are excluded from the drift timeline (they still count "
                "towards the overall Scope Drift Index)."
            )
        frame["created_date"] = parsed
    else:
        warnings.append(
            "No created_date column; the Scope Drift Index over time cannot "
            "be plotted."
        )

    if "true_class" in frame.columns:
        labels = frame["true_class"].dropna().astype(str).str.strip()
        unknown = sorted(set(labels) - set(CLASSES) - {""})
        if unknown:
            raise DataValidationError(
                "The true_class column contains label(s) DriftGuard does not "
                f"recognise: {', '.join(unknown)}. Allowed labels are: "
                f"{', '.join(CLASSES)}."
            )

    _flag_synthetic(frame, "tickets", warnings)

    frame["text"] = _build_text(frame)
    keep = [c for c in (*TICKET_REQUIRED, *TICKET_OPTIONAL, "text") if c in frame.columns]
    frame = frame[keep].reset_index(drop=True)
    return frame, LoadReport(rows=len(frame), warnings=warnings)


# --------------------------------------------------------------------------
# Bundled sample data
# --------------------------------------------------------------------------
SAMPLE_DIR = Path(__file__).parent / "sample_data"


def sample_baseline() -> pd.DataFrame:
    frame, _ = load_baseline_csv(SAMPLE_DIR / "baseline_requirements.csv")
    return frame


def sample_tickets() -> pd.DataFrame:
    frame, _ = load_tickets(SAMPLE_DIR / "tickets.csv")
    return frame


def sample_labelled_tickets() -> pd.DataFrame:
    """Tickets carrying manual ``true_class`` labels, for evaluation."""
    frame, _ = load_tickets(SAMPLE_DIR / "labelled_test_set.csv")
    return frame


def read_sample_bytes(name: str) -> bytes:
    """Raw bytes of a bundled sample file, for download buttons in the UI."""
    path = SAMPLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"No bundled sample file named {name!r}.")
    return path.read_bytes()


def frame_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")
