"""Políticas para dimensionar a recuperação antes do reranking."""

from __future__ import annotations


def candidate_pool_size(
    *,
    top_k: int,
    per_collection: int,
    multiplier: int = 4,
    minimum: int = 20,
    maximum: int = 100,
) -> int:
    """Calcula quantos candidatos recuperar por coleção.

    O reranker precisa receber mais documentos do que o número final solicitado.
    A política preserva solicitações explícitas maiores e aplica limites para
    evitar consultas excessivamente pequenas ou grandes.
    """
    safe_top_k = max(1, int(top_k))
    safe_per_collection = max(1, int(per_collection))
    safe_multiplier = max(1, int(multiplier))
    safe_minimum = max(1, int(minimum))
    safe_maximum = max(safe_minimum, int(maximum))

    desired = max(
        safe_per_collection,
        safe_top_k * safe_multiplier,
        safe_minimum,
    )
    return min(desired, safe_maximum)
