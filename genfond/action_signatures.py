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

``Sum |D|`` over all pairs is nevertheless the dominant term of the instance, and most pairs
never constrain the optimum: any reasonable selection separates them for free.  `LazyPairs`
therefore serves the same relation on demand, driven by the models of a relaxation that omits it
(see `lazy_pairs.py`): given a selection and a good/bad labelling of the classes it reports the
*violated* pairs, and emits facts only for those.
"""

import logging
from typing import Iterable, Iterator, NamedTuple, Optional, Sequence

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


class _Group:
    """The classes of one (action name, arity), packed into bit vectors.

    `concept_bits` keeps one row per argument position and `role_bits` one per ordered position
    pair. Both are or-reduced away when a pair's distinguishing set is formed, but the masked
    equality test that recognises an already separated pair needs them unreduced: a selected
    concept separates the pair as soon as it differs at *some* position.
    """

    def __init__(self, arity: int) -> None:
        self.arity = arity
        self.members: list[int] = []
        self.feature_bits = np.zeros((0, 0), dtype=np.uint64)
        self.concept_bits = np.zeros((0, 0, 0), dtype=np.uint64)
        self.role_bits = np.zeros((0, 0, 0), dtype=np.uint64)
        self.flat = np.zeros((0, 0), dtype=np.uint64)

    def pack(self, index: "SignatureIndex", signatures: Sequence[ActionSignature]) -> None:
        size = len(self.members)
        arity = self.arity
        self.feature_bits = np.zeros((size, index.feature_words), dtype=np.uint64)
        self.concept_bits = np.zeros((size, arity, index.concept_words), dtype=np.uint64)
        self.role_bits = np.zeros((size, arity * arity, index.role_words), dtype=np.uint64)
        for row, identifier in enumerate(self.members):
            signature = signatures[identifier]
            for feature, value in signature.bools:
                if value:
                    _set_bit(self.feature_bits, (row,), index.feature_index[feature])
            for concept, position in signature.concepts:
                _set_bit(self.concept_bits, (row, position), index.concept_index[concept])
            for role, position1, position2 in signature.roles:
                _set_bit(self.role_bits, (row, position1 * arity + position2), index.role_index[role])
        self.flat = np.concatenate(
            [self.feature_bits, self.concept_bits.reshape(size, -1), self.role_bits.reshape(size, -1)],
            axis=1,
        )

    def dist_keys(self, row: int, others: np.ndarray) -> np.ndarray:
        """The packed distinguishing sets of `row` against each of `others`, one per output row."""
        return np.concatenate(
            [
                self.feature_bits[row] ^ self.feature_bits[others],
                np.bitwise_or.reduce(self.concept_bits[row] ^ self.concept_bits[others], axis=1),
                np.bitwise_or.reduce(self.role_bits[row] ^ self.role_bits[others], axis=1),
            ],
            axis=1,
        )


class SignatureIndex:
    """Packed bit vectors for a list of signature classes, grouped by (action name, arity).

    The packed key of a distinguishing set is `[features | concepts | roles]` over the *global*
    universe with the argument positions already or-reduced away, so two pairs with the same set
    share a key even when their actions have different arities.
    """

    def __init__(self, signatures: Sequence[ActionSignature]) -> None:
        self.signatures = signatures
        features = sorted({feature for signature in signatures for feature, _ in signature.bools})
        concepts = sorted({concept for signature in signatures for concept, _ in signature.concepts})
        roles = sorted({role for signature in signatures for role, _, _ in signature.roles})
        self.feature_index = {feature: index for index, feature in enumerate(features)}
        self.concept_index = {concept: index for index, concept in enumerate(concepts)}
        self.role_index = {role: index for index, role in enumerate(roles)}
        self.feature_words = _num_words(len(features))
        self.concept_words = _num_words(len(concepts))
        self.role_words = _num_words(len(roles))
        self.terms = (
            [f'f("{feature}")' for feature in features]
            + [""] * (self.feature_words * WORD_BITS - len(features))
            + [f'c("{concept}")' for concept in concepts]
            + [""] * (self.concept_words * WORD_BITS - len(concepts))
            + [f'r("{role}")' for role in roles]
            + [""] * (self.role_words * WORD_BITS - len(roles))
        )
        self.row_bytes = (self.feature_words + self.concept_words + self.role_words) * 8

        group_ids: dict[tuple[str, int], int] = dict()
        self.groups: list[_Group] = []
        # signature id -> (index into self.groups, row inside that group)
        self.row_of: dict[int, tuple[int, int]] = dict()
        for identifier, signature in enumerate(signatures):
            key = (signature.name, signature.arity)
            if key not in group_ids:
                group_ids[key] = len(self.groups)
                self.groups.append(_Group(signature.arity))
            group_id = group_ids[key]
            group = self.groups[group_id]
            self.row_of[identifier] = (group_id, len(group.members))
            group.members.append(identifier)
        for group in self.groups:
            if len(group.members) > 1:
                group.pack(self, signatures)

    def decode(self, key: bytes) -> list[str]:
        """Turn a packed distinguishing-set key back into the ASP terms it selects."""
        elements = []
        for word_index, word in enumerate(np.frombuffer(key, dtype=np.uint64)):
            value = int(word)
            while value:
                lowest = value & -value
                elements.append(self.terms[word_index * WORD_BITS + lowest.bit_length() - 1])
                value ^= lowest
        return elements

    def num_pairs(self) -> int:
        """How many pairs the eager encoding would emit."""
        return sum(len(group.members) * (len(group.members) - 1) // 2 for group in self.groups)


def iter_dist_set_facts(signatures: Sequence[ActionSignature]) -> Iterator[str]:
    """Yield the ``sig_pair/3`` and ``dist/2`` facts for a list of signature classes.

    The sizes involved (classes, pairs, distinct sets, emitted facts) are logged at INFO once
    the iterator is exhausted.
    """
    index = SignatureIndex(signatures)
    dist_set_ids: dict[bytes, int] = dict()
    num_pairs = 0
    num_dist_facts = 0
    for group in index.groups:
        size = len(group.members)
        if size < 2:
            continue
        for row in range(size - 1):
            # One chunk per left-hand class: all pairs (members[row], members[row + 1 + k]).
            buffer = group.dist_keys(row, np.arange(row + 1, size)).tobytes()
            for offset in range(size - row - 1):
                key = buffer[offset * index.row_bytes : (offset + 1) * index.row_bytes]
                dist_set_id = dist_set_ids.get(key)
                if dist_set_id is None:
                    dist_set_id = len(dist_set_ids)
                    dist_set_ids[key] = dist_set_id
                    elements = index.decode(key)
                    num_dist_facts += len(elements)
                    yield "".join(f"dist({dist_set_id},{element}).\n" for element in elements)
                num_pairs += 1
                yield f"sig_pair({group.members[row]},{group.members[row + 1 + offset]},{dist_set_id}).\n"
    log.info(
        f"Separation layer: {len(signatures)} signature class(es), {num_pairs} pair(s), "
        f"{len(dist_set_ids)} distinct distinguishing set(s), {num_dist_facts} dist/2 fact(s)"
    )


def _trim(
    sizes: list[np.ndarray], lefts: list[np.ndarray], rights: list[np.ndarray], limit: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Concatenate the accumulated candidates, keeping the `limit` smallest distinguishing sets."""
    size = np.concatenate(sizes)
    left = np.concatenate(lefts)
    right = np.concatenate(rights)
    if len(size) > limit:
        keep = np.argpartition(size, limit)[:limit]
        size, left, right = size[keep], left[keep], right[keep]
    return [size], [left], [right]


class LazyPairs:
    """The same relation as `iter_dist_set_facts`, served on demand.

    A pair {K1, K2} constrains a model only if K1 is good, K2 is bad (or the mirror) and no
    selected element distinguishes them. The second condition is a masked equality: the pair is
    separated iff the selection differs on some feature, on some concept at some argument
    position, or on some role at some ordered position pair. Grouping the classes of one action
    name by their *masked* bit vector therefore yields the violated pairs directly, as the
    good x bad cross product inside each bucket -- no scan over all pairs.
    """

    def __init__(self, signatures: Sequence[ActionSignature]) -> None:
        self.index = SignatureIndex(signatures)
        self.dist_set_ids: dict[bytes, int] = dict()
        self.num_pairs_emitted = 0
        self.num_dist_facts = 0

    def _masks(
        self, features: Iterable[str], concepts: Iterable[str], roles: Iterable[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feature_mask = np.zeros(self.index.feature_words, dtype=np.uint64)
        concept_mask = np.zeros(self.index.concept_words, dtype=np.uint64)
        role_mask = np.zeros(self.index.role_words, dtype=np.uint64)
        for feature in features:
            if feature in self.index.feature_index:
                _set_bit(feature_mask, (), self.index.feature_index[feature])
        for concept in concepts:
            if concept in self.index.concept_index:
                _set_bit(concept_mask, (), self.index.concept_index[concept])
        for role in roles:
            if role in self.index.role_index:
                _set_bit(role_mask, (), self.index.role_index[role])
        return feature_mask, concept_mask, role_mask

    def violated_pairs(
        self,
        selected_features: Iterable[str],
        selected_concepts: Iterable[str],
        selected_roles: Iterable[str],
        good: Iterable[int],
        bad: Iterable[int],
        limit: Optional[int] = None,
    ) -> tuple[int, list[tuple[int, int]]]:
        """Return (number of violated pairs, up to `limit` of them, smallest |D| first).

        Small distinguishing sets are the hardest to hit, so they constrain the selection most;
        pairs with a large |D| tend to be separated for free by whatever the solver picks next.
        """
        feature_mask, concept_mask, role_mask = self._masks(selected_features, selected_concepts, selected_roles)
        good = set(good)
        bad = set(bad)
        total = 0
        sizes: list[np.ndarray] = []
        lefts: list[np.ndarray] = []
        rights: list[np.ndarray] = []
        kept = 0
        for group in self.index.groups:
            if len(group.members) < 2:
                continue
            members = np.array(group.members)
            good_rows = [row for row, identifier in enumerate(group.members) if identifier in good]
            bad_rows = [row for row, identifier in enumerate(group.members) if identifier in bad]
            if not good_rows or not bad_rows:
                continue
            flat_mask = np.concatenate(
                [feature_mask, np.tile(concept_mask, group.arity), np.tile(role_mask, group.arity * group.arity)]
            )
            masked = (group.flat & flat_mask).tobytes()
            width = group.flat.shape[1] * 8
            buckets: dict[bytes, list[int]] = dict()
            for row in bad_rows:
                buckets.setdefault(masked[row * width : (row + 1) * width], []).append(row)
            for row in good_rows:
                bucket = buckets.get(masked[row * width : (row + 1) * width])
                if not bucket:
                    continue
                total += len(bucket)
                others = np.array(bucket)
                keys = group.dist_keys(row, others)
                sizes.append(np.bitwise_count(keys).sum(axis=1, dtype=np.int64))
                lefts.append(np.full(len(others), members[row]))
                rights.append(members[others])
                kept += len(others)
            if limit and kept > max(4 * limit, 100_000):
                sizes, lefts, rights = _trim(sizes, lefts, rights, limit)
                kept = len(sizes[0])
        if not sizes:
            return total, []
        sizes, lefts, rights = _trim(sizes, lefts, rights, limit if limit else kept)
        order = np.argsort(sizes[0], kind="stable")
        return total, [(int(lefts[0][index]), int(rights[0][index])) for index in order]

    def facts(self, batch: int, pairs: Sequence[tuple[int, int]]) -> str:
        """The ``dist/3`` and ``sig_pair/4`` facts for `pairs`, tagged with the batch index.

        The tag keeps the incrementally grounded rules of ``#program pairs(k)`` from re-grounding
        over the facts of earlier batches. A ``dist/3`` fact is emitted only in the batch that
        first needs its set; `sig_hit/1` is a plain atom, so the rule that derives it from an
        earlier batch stays valid for the later batches that reuse the id.
        """
        program = []
        for left, right in pairs:
            group_id, left_row = self.index.row_of[left]
            _, right_row = self.index.row_of[right]
            group = self.index.groups[group_id]
            key = group.dist_keys(left_row, np.array([right_row])).tobytes()
            dist_set_id = self.dist_set_ids.get(key)
            if dist_set_id is None:
                dist_set_id = len(self.dist_set_ids)
                self.dist_set_ids[key] = dist_set_id
                elements = self.index.decode(key)
                self.num_dist_facts += len(elements)
                program += [f"dist({batch},{dist_set_id},{element}).\n" for element in elements]
            self.num_pairs_emitted += 1
            program.append(f"sig_pair({batch},{left},{right},{dist_set_id}).\n")
        return "".join(program)
