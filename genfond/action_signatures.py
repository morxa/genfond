"""Action signature classes and the pairwise distinguishing sets derived from them.

The datalog separation constraint observes an (instance, state, action) triple only through the
action name, the concept membership of each argument position, the role membership of each
ordered position pair, and the source state's boolean feature vector.  `ActionSignature` records
exactly that, so the constraint may range over signature classes instead of over triples.

Even quotiented, writing the constraint as a set of ASP rules over pairs of classes costs
|K|^2 * (|F| + |C|*arity + |R|*arity^2) ground rules, because the ``sig_*_dist`` rules pair every
class pair with every feature/concept/role.  Which elements distinguish a pair is a property of
the instance, not of the model, so it is computed here instead and emitted as

    sig_pair(K1, K2, D).     % one fact per unordered pair of classes with the same action name
    dist(D, f("...")).       % one fact per element of each *distinct* distinguishing set

Identical sets share an id, so the second family collapses hard in practice.  The ASP side then
only needs the hitting-set constraint, which is linear in the number of emitted facts.
"""

import logging
from typing import Iterator, NamedTuple, Sequence

import numpy as np

log = logging.getLogger("genfond.signatures")

WORD_BITS = 64


class ActionSignature(NamedTuple):
    """Everything the datalog separation constraint can observe about an action in a state."""

    name: str
    arity: int
    # (concept, argument position) memberships.
    concepts: frozenset[tuple[str, int]]
    # (role, argument position, argument position) memberships.
    roles: frozenset[tuple[str, int, int]]
    # The source state's boolean feature vector, thresholded at 0, in a fixed feature order.
    bools: tuple[tuple[str, int], ...]


def _num_words(n: int) -> int:
    return (n + WORD_BITS - 1) // WORD_BITS


def _set_bit(array: np.ndarray, row: tuple[int, ...], index: int) -> None:
    array[row + (index // WORD_BITS,)] |= np.uint64(1) << np.uint64(index % WORD_BITS)


def _decode(key: bytes, terms: Sequence[str]) -> list[str]:
    """Turn a packed distinguishing-set key back into the ASP terms it selects."""
    elements = []
    for word_index, word in enumerate(np.frombuffer(key, dtype=np.uint64)):
        value = int(word)
        while value:
            lowest = value & -value
            elements.append(terms[word_index * WORD_BITS + lowest.bit_length() - 1])
            value ^= lowest
    return elements


def iter_dist_set_facts(signatures: Sequence[ActionSignature]) -> Iterator[str]:
    """Yield the ``sig_pair/3`` and ``dist/2`` facts for a list of signature classes.

    The sizes involved (classes, pairs, distinct sets, emitted facts) are logged at INFO once
    the iterator is exhausted.
    """
    features = sorted({feature for signature in signatures for feature, _ in signature.bools})
    concepts = sorted({concept for signature in signatures for concept, _ in signature.concepts})
    roles = sorted({role for signature in signatures for role, _, _ in signature.roles})
    feature_index = {feature: index for index, feature in enumerate(features)}
    concept_index = {concept: index for index, concept in enumerate(concepts)}
    role_index = {role: index for index, role in enumerate(roles)}
    feature_words = _num_words(len(features))
    concept_words = _num_words(len(concepts))
    role_words = _num_words(len(roles))
    # The packed key is [features | concepts | roles] over the *global* universe, with the
    # argument positions already or-reduced away, so two pairs with the same distinguishing set
    # share a key even when their actions have different arities.
    terms = (
        [f'f("{feature}")' for feature in features]
        + [""] * (feature_words * WORD_BITS - len(features))
        + [f'c("{concept}")' for concept in concepts]
        + [""] * (concept_words * WORD_BITS - len(concepts))
        + [f'r("{role}")' for role in roles]
        + [""] * (role_words * WORD_BITS - len(roles))
    )
    row_bytes = (feature_words + concept_words + role_words) * 8

    groups: dict[tuple[str, int], list[int]] = dict()
    for identifier, signature in enumerate(signatures):
        groups.setdefault((signature.name, signature.arity), []).append(identifier)

    dist_set_ids: dict[bytes, int] = dict()
    num_pairs = 0
    num_dist_facts = 0
    for (_, arity), members in groups.items():
        size = len(members)
        if size < 2:
            continue
        feature_bits = np.zeros((size, feature_words), dtype=np.uint64)
        concept_bits = np.zeros((size, arity, concept_words), dtype=np.uint64)
        role_bits = np.zeros((size, arity * arity, role_words), dtype=np.uint64)
        for row, identifier in enumerate(members):
            signature = signatures[identifier]
            for feature, value in signature.bools:
                if value:
                    _set_bit(feature_bits, (row,), feature_index[feature])
            for concept, position in signature.concepts:
                _set_bit(concept_bits, (row, position), concept_index[concept])
            for role, position1, position2 in signature.roles:
                _set_bit(role_bits, (row, position1 * arity + position2), role_index[role])
        for row in range(size - 1):
            # One chunk per left-hand class: all pairs (members[row], members[row + 1 + k]).
            differing_features = feature_bits[row] ^ feature_bits[row + 1 :]
            differing_concepts = np.bitwise_or.reduce(concept_bits[row] ^ concept_bits[row + 1 :], axis=1)
            differing_roles = np.bitwise_or.reduce(role_bits[row] ^ role_bits[row + 1 :], axis=1)
            keys = np.concatenate([differing_features, differing_concepts, differing_roles], axis=1)
            buffer = keys.tobytes()
            for offset in range(size - row - 1):
                key = buffer[offset * row_bytes : (offset + 1) * row_bytes]
                dist_set_id = dist_set_ids.get(key)
                if dist_set_id is None:
                    dist_set_id = len(dist_set_ids)
                    dist_set_ids[key] = dist_set_id
                    elements = _decode(key, terms)
                    num_dist_facts += len(elements)
                    yield "".join(f"dist({dist_set_id},{element}).\n" for element in elements)
                num_pairs += 1
                yield f"sig_pair({members[row]},{members[row + 1 + offset]},{dist_set_id}).\n"
    log.info(
        f"Separation layer: {len(signatures)} signature class(es), {num_pairs} pair(s), "
        f"{len(dist_set_ids)} distinct distinguishing set(s), {num_dist_facts} dist/2 fact(s)"
    )
