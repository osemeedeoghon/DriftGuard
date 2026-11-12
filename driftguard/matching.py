"""Local text similarity matching between tickets and baseline requirements.

Uses TF-IDF vectors and cosine similarity from scikit-learn. Everything runs
on the local CPU: no embedding service, no API key, no network access.

What this does and does not do
------------------------------
TF-IDF measures *lexical* overlap weighted by how distinctive each term is
across the corpus. It has no understanding of meaning. A ticket that paraphrases
a requirement without reusing its vocabulary will score low, and two unrelated
tickets that share domain jargon will score high. The evidence returned by
``match_tickets`` exists so a human reviewer can see exactly which words drove
a score and overrule it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import TEXT_LANGUAGE


@dataclass(frozen=True)
class MatchEvidence:
    """Why one ticket was matched to one requirement."""

    requirement_id: str
    similarity: float
    shared_terms: tuple[str, ...]

    def describe(self) -> str:
        if not self.shared_terms:
            return (
                f"No distinctive vocabulary shared with {self.requirement_id} "
                f"(cosine similarity {self.similarity:.2f})."
            )
        terms = ", ".join(self.shared_terms)
        return (
            f"Cosine similarity {self.similarity:.2f} with {self.requirement_id}, "
            f"driven by the shared terms: {terms}."
        )


class RequirementMatcher:
    """Fits a TF-IDF space on the baseline and scores tickets against it.

    The vectoriser is fitted on the baseline requirements *and* the tickets
    together, so that inverse document frequency reflects the whole project
    vocabulary rather than only the baseline's.
    """

    def __init__(
        self,
        *,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int = 1,
        sublinear_tf: bool = True,
    ) -> None:
        """Unigrams by default.

        Adding bigrams was measured on the bundled sample project and made
        separation markedly worse: a ticket phrases a requirement in its own
        words, so bigram overlap is sparse, and the extra zero-weight features
        dilute the unigram signal. On that data unigrams lifted the median
        in-scope similarity from 0.39 to 0.58 while leaving new-scope tickets
        near 0.11.
        """
        self.vectorizer = TfidfVectorizer(
            stop_words=TEXT_LANGUAGE,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            strip_accents="unicode",
            lowercase=True,
        )
        self._requirement_ids: list[str] = []
        self._requirement_matrix = None
        self._fitted = False

    def fit(self, baseline: pd.DataFrame, tickets: pd.DataFrame | None = None) -> "RequirementMatcher":
        if baseline.empty:
            raise ValueError("Cannot match against an empty baseline.")
        if "text" not in baseline.columns:
            raise ValueError("Baseline frame must carry a 'text' column.")

        corpus = list(baseline["text"].fillna("").astype(str))
        if tickets is not None and "text" in tickets.columns:
            corpus = corpus + list(tickets["text"].fillna("").astype(str))

        if not any(t.strip() for t in corpus):
            raise ValueError(
                "All requirement and ticket text is empty; nothing to match."
            )

        self.vectorizer.fit(corpus)
        self._requirement_matrix = self.vectorizer.transform(
            baseline["text"].fillna("").astype(str)
        )
        self._requirement_ids = list(baseline["requirement_id"].astype(str))
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("RequirementMatcher.fit() must be called first.")

    def similarity_matrix(self, tickets: pd.DataFrame) -> np.ndarray:
        """Cosine similarity of every ticket against every requirement."""
        self._check_fitted()
        ticket_matrix = self.vectorizer.transform(
            tickets["text"].fillna("").astype(str)
        )
        return cosine_similarity(ticket_matrix, self._requirement_matrix)

    def _shared_terms(self, ticket_text: str, requirement_index: int, limit: int = 6) -> tuple[str, ...]:
        """Terms present in both texts, ranked by combined TF-IDF weight."""
        ticket_vector = self.vectorizer.transform([ticket_text])
        requirement_vector = self._requirement_matrix[requirement_index]

        ticket_terms = dict(zip(ticket_vector.indices, ticket_vector.data))
        requirement_terms = dict(zip(requirement_vector.indices, requirement_vector.data))
        overlap = set(ticket_terms) & set(requirement_terms)
        if not overlap:
            return ()

        names = self.vectorizer.get_feature_names_out()
        ranked = sorted(
            overlap,
            key=lambda i: ticket_terms[i] * requirement_terms[i],
            reverse=True,
        )
        return tuple(str(names[i]) for i in ranked[:limit])

    def shared_term_weights(
        self, ticket_text: str, requirement_id: str, *, limit: int = 12
    ) -> pd.DataFrame:
        """Per-term TF-IDF contribution behind one ticket-requirement match.

        The TF-IDF vocabulary *is* this model's feature space, so this is the
        feature-level view of a single prediction: the terms present in both
        texts, ranked by the product of their weights -- which is exactly what
        the cosine similarity sums over.

        Returns a frame with columns ``term``, ``ticket_weight``,
        ``requirement_weight`` and ``weight`` (their product).
        """
        self._check_fitted()
        if requirement_id not in self._requirement_ids:
            raise ValueError(
                f"Unknown requirement id {requirement_id!r}. Known ids: "
                + ", ".join(self._requirement_ids[:5])
                + ("..." if len(self._requirement_ids) > 5 else "")
            )

        index = self._requirement_ids.index(requirement_id)
        ticket_vector = self.vectorizer.transform([ticket_text or ""])
        requirement_vector = self._requirement_matrix[index]

        ticket_terms = dict(zip(ticket_vector.indices, ticket_vector.data))
        requirement_terms = dict(zip(requirement_vector.indices, requirement_vector.data))
        overlap = set(ticket_terms) & set(requirement_terms)
        if not overlap:
            return pd.DataFrame(
                columns=["term", "ticket_weight", "requirement_weight", "weight"]
            )

        names = self.vectorizer.get_feature_names_out()
        rows = [
            {
                "term": str(names[i]),
                "ticket_weight": float(ticket_terms[i]),
                "requirement_weight": float(requirement_terms[i]),
                "weight": float(ticket_terms[i] * requirement_terms[i]),
            }
            for i in overlap
        ]
        frame = pd.DataFrame(rows).sort_values("weight", ascending=False)
        return frame.head(limit).reset_index(drop=True)

    def match_tickets(self, tickets: pd.DataFrame, *, top_k: int = 3) -> pd.DataFrame:
        """Attach the best-matching requirement and supporting evidence.

        Returns a copy of ``tickets`` with these added columns:

        ``matched_requirement_id``  best-scoring baseline requirement
        ``similarity``              cosine similarity of that match, 0..1
        ``runner_up_id``            second-best requirement, or None
        ``runner_up_similarity``    its score (the margin shows how decisive
                                    the top match was)
        ``shared_terms``            vocabulary the match rests on
        ``match_evidence``          a human-readable sentence
        ``alternative_matches``     list of (requirement_id, similarity) tuples
        """
        self._check_fitted()
        if tickets.empty:
            return tickets.copy()
        if "text" not in tickets.columns:
            raise ValueError("Ticket frame must carry a 'text' column.")

        scores = self.similarity_matrix(tickets)
        texts = list(tickets["text"].fillna("").astype(str))

        best_ids: list[str | None] = []
        best_scores: list[float] = []
        runner_ids: list[str | None] = []
        runner_scores: list[float] = []
        shared: list[tuple[str, ...]] = []
        evidence: list[str] = []
        alternatives: list[list[tuple[str, float]]] = []

        k = min(top_k, len(self._requirement_ids))
        for row, text in zip(scores, texts):
            order = np.argsort(row)[::-1]
            top = order[:k]
            alternatives.append(
                [(self._requirement_ids[i], float(row[i])) for i in top]
            )

            best_index = int(order[0])
            best_score = float(row[best_index])
            best_ids.append(self._requirement_ids[best_index])
            best_scores.append(best_score)

            if len(order) > 1:
                second = int(order[1])
                runner_ids.append(self._requirement_ids[second])
                runner_scores.append(float(row[second]))
            else:
                runner_ids.append(None)
                runner_scores.append(0.0)

            terms = self._shared_terms(text, best_index) if best_score > 0 else ()
            shared.append(terms)
            evidence.append(
                MatchEvidence(
                    requirement_id=self._requirement_ids[best_index],
                    similarity=best_score,
                    shared_terms=terms,
                ).describe()
            )

        result = tickets.copy()
        result["matched_requirement_id"] = best_ids
        result["similarity"] = best_scores
        result["runner_up_id"] = runner_ids
        result["runner_up_similarity"] = runner_scores
        result["match_margin"] = result["similarity"] - result["runner_up_similarity"]
        result["shared_terms"] = shared
        result["match_evidence"] = evidence
        result["alternative_matches"] = alternatives
        return result


def attach_requirement_details(
    matched: pd.DataFrame, baseline: pd.DataFrame
) -> pd.DataFrame:
    """Join the matched requirement's title and text onto the ticket rows."""
    columns = ["requirement_id"]
    for optional in ("title", "description", "size"):
        if optional in baseline.columns:
            columns.append(optional)
    lookup = baseline[columns].rename(
        columns={
            "requirement_id": "matched_requirement_id",
            "title": "matched_requirement_title",
            "description": "matched_requirement_text",
            "size": "matched_requirement_size",
        }
    )
    return matched.merge(lookup, on="matched_requirement_id", how="left")
