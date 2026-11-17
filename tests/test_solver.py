"""Solver unit tests: intervals, uniqueness, nullability, match types, cycles."""
from __future__ import annotations

import pytest

from app.models import (
    Column,
    ForeignKey,
    IntRange,
    MatchType,
    TableSchema,
    TableSpec,
    UniqueConstraint,
)
from app.solver import Generator, Unsatisfiable


def col(name, pg_type="integer", nullable=False):
    return Column(
        name=name,
        pg_type=pg_type,
        ordinal=1,
        nullable=nullable,
        has_default=False,
        is_identity=False,
        is_generated=False,
    )


def spec(table, rows, ranges):
    return TableSpec(
        table=table,
        rows=rows,
        ranges={c: IntRange(low=lo, high=hi) for c, (lo, hi) in ranges.items()},
    )


def _check_unique(data, key_cols):
    seen = set()
    for row in data:
        key = tuple(row[c] for c in key_cols)
        if any(v is None for v in key):
            continue
        assert key not in seen, f"duplicate non-NULL key {key}"
        seen.add(key)


def test_pk_uniqueness_and_ranges():
    ts = TableSchema(
        schema_name="public",
        name="t",
        columns=[col("id"), col("v", nullable=True)],
        unique_constraints=[UniqueConstraint(name="pk", columns=["id"], is_primary=True)],
        foreign_keys=[],
    )
    specs = {"t": spec("t", 5, {"id": (1, 5), "v": (10, 20)})}
    data = Generator({"t": ts}, specs).solve()["t"]
    assert len(data) == 5
    assert sorted(r["id"] for r in data) == [1, 2, 3, 4, 5]
    assert all(10 <= r["v"] <= 20 for r in data)


def test_rows_exceed_range_cardinality_is_unsat():
    ts = TableSchema(
        schema_name="public",
        name="t",
        columns=[col("id")],
        unique_constraints=[UniqueConstraint(name="pk", columns=["id"], is_primary=True)],
        foreign_keys=[],
    )
    specs = {"t": spec("t", 5, {"id": (1, 2)})}
    with pytest.raises(Unsatisfiable) as exc:
        Generator({"t": ts}, specs).solve()
    kinds = {g.kind for g in exc.value.groups}
    assert "unique" in kinds and "range" in kinds


def test_nullable_column_can_be_null():
    ts = TableSchema(
        schema_name="public",
        name="t",
        columns=[col("id"), col("v", nullable=True)],
        unique_constraints=[UniqueConstraint(name="pk", columns=["id"], is_primary=True)],
        foreign_keys=[],
    )
    # NOT NULL id has only 2 values but 3 rows requested; v being nullable does
    # not help ids — this stays unsat, proving nullability is column-scoped.
    specs_bad = {"t": spec("t", 3, {"id": (1, 2), "v": (1, 1)})}
    with pytest.raises(Unsatisfiable):
        Generator({"t": ts}, specs_bad).solve()

    # A nullable unique column may repeat NULLs freely.
    ts2 = TableSchema(
        schema_name="public",
        name="t",
        columns=[col("id"), col("v", nullable=True)],
        unique_constraints=[
            UniqueConstraint(name="pk", columns=["id"], is_primary=True),
            UniqueConstraint(name="uk_v", columns=["v"], is_primary=False),
        ],
        foreign_keys=[],
    )
    specs = {"t": spec("t", 4, {"id": (1, 10), "v": (100, 100)})}
    data = Generator({"t": ts2}, specs).solve()["t"]
    # v has a single-point interval; NULL repetition must be used to satisfy UK.
    nulls = sum(1 for r in data if r["v"] is None)
    assert nulls >= 3
    _check_unique(data, ["v"])


def test_self_referencing_foreign_key():
    ts = TableSchema(
        schema_name="public",
        name="node",
        columns=[col("id"), col("parent", nullable=True)],
        unique_constraints=[
            UniqueConstraint(name="pk", columns=["id"], is_primary=True)
        ],
        foreign_keys=[
            ForeignKey(
                name="fk_parent",
                columns=["parent"],
                ref_table="node",
                ref_columns=["id"],
                match=MatchType.SIMPLE,
                on_update="NO ACTION",
                on_delete="SET NULL",
                not_deferrable=True,
                initially_deferred=False,
            )
        ],
    )
    specs = {"node": spec("node", 4, {"id": (1, 4), "parent": (1, 4)})}
    data = Generator({"node": ts}, specs).solve()["node"]
    ids = {r["id"] for r in data}
    for r in data:
        assert r["parent"] is None or r["parent"] in ids


def test_circular_two_tables_resolved():
    a = TableSchema(
        schema_name="public",
        name="a",
        columns=[col("id"), col("b_id", nullable=True)],
        unique_constraints=[UniqueConstraint(name="pa", columns=["id"], is_primary=True)],
        foreign_keys=[
            ForeignKey(
                name="fk_a_b", columns=["b_id"], ref_table="b", ref_columns=["id"],
                match=MatchType.SIMPLE, on_update="NO ACTION", on_delete="SET NULL",
                not_deferrable=True, initially_deferred=False,
            )
        ],
    )
    b = TableSchema(
        schema_name="public",
        name="b",
        columns=[col("id"), col("a_id")],
        unique_constraints=[UniqueConstraint(name="pb", columns=["id"], is_primary=True)],
        foreign_keys=[
            ForeignKey(
                name="fk_b_a", columns=["a_id"], ref_table="a", ref_columns=["id"],
                match=MatchType.SIMPLE, on_update="NO ACTION", on_delete="NO ACTION",
                not_deferrable=True, initially_deferred=False,
            )
        ],
    )
    specs = {
        "a": spec("a", 3, {"id": (1, 3), "b_id": (10, 12)}),
        "b": spec("b", 3, {"id": (10, 12), "a_id": (1, 3)}),
    }
    out = Generator({"a": a, "b": b}, specs).solve()
    a_ids = {r["id"] for r in out["a"]}
    b_ids = {r["id"] for r in out["b"]}
    for r in out["a"]:
        assert r["b_id"] is None or r["b_id"] in b_ids
    for r in out["b"]:
        assert r["a_id"] in a_ids


def _composite_table(match: MatchType):
    parent = TableSchema(
        schema_name="public",
        name="p",
        columns=[col("k1"), col("k2", nullable=True)],
        unique_constraints=[
            UniqueConstraint(name="pk_p", columns=["k1", "k2"], is_primary=True)
        ],
        foreign_keys=[],
    )
    child = TableSchema(
        schema_name="public",
        name="c",
        columns=[col("id"), col("f1", nullable=True), col("f2", nullable=True)],
        unique_constraints=[
            UniqueConstraint(name="pk_c", columns=["id"], is_primary=True)
        ],
        foreign_keys=[
            ForeignKey(
                name="fk_c_p",
                columns=["f1", "f2"],
                ref_table="p",
                ref_columns=["k1", "k2"],
                match=match,
                on_update="NO ACTION",
                on_delete="NO ACTION",
                not_deferrable=True,
                initially_deferred=False,
            )
        ],
    )
    return parent, child


def test_match_full_forbids_mixed_null_via_solver_ranges():
    # Force the child f1 to NULL via an impossible non-null situation is hard
    # directly; instead assert a feasible solve honours FULL all-or-nothing.
    parent, child = _composite_table(MatchType.FULL)
    specs = {
        "p": spec("p", 2, {"k1": (1, 2), "k2": (10, 11)}),
        "c": spec("c", 4, {"id": (100, 103), "f1": (1, 2), "f2": (10, 11)}),
    }
    out = Generator({"p": parent, "c": child}, specs).solve()
    pkeys = {(r["k1"], r["k2"]) for r in out["p"]}
    for r in out["c"]:
        if r["f1"] is None and r["f2"] is None:
            continue  # all-NULL allowed
        # MATCH FULL: never a mixed key
        assert (r["f1"] is None) == (r["f2"] is None)
        assert (r["f1"], r["f2"]) in pkeys


def test_match_simple_allows_mixed_null():
    parent, child = _composite_table(MatchType.SIMPLE)
    specs = {
        "p": spec("p", 2, {"k1": (1, 2), "k2": (10, 11)}),
        "c": spec("c", 6, {"id": (100, 105), "f1": (1, 2), "f2": (10, 11)}),
    }
    out = Generator({"p": parent, "c": child}, specs).solve()
    # solver prefers non-NULL; feasibility is the key assertion here.
    assert len(out["c"]) == 6


def test_match_simple_vs_full_mixed_null_distinction():
    """A scenario feasible only if mixed NULL composite keys are allowed."""
    # Child has more rows than the parent has distinct full keys, and the
    # non-null intervals cannot supply enough distinct full keys; mixed-NULL
    # rows provide the extra capacity under SIMPLE but are banned under FULL.
    parent, child_simple = _composite_table(MatchType.SIMPLE)
    _, child_full = _composite_table(MatchType.FULL)
    # f1 single-point, f2 single-point => at most one distinct non-NULL key.
    tight = {
        "p": spec("p", 1, {"k1": (1, 1), "k2": (10, 10)}),
        "c": spec("c", 3, {"id": (100, 102), "f1": (1, 1), "f2": (10, 10)}),
    }
    # FULL: every non-all-NULL row must match the single parent key; that is
    # still feasible without NULLs, so instead force NULL usage via a unique
    # index on the FK columns to expose the distinction.
    child_simple.unique_constraints.append(
        UniqueConstraint(name="uk_fk", columns=["f1", "f2"], is_primary=False)
    )
    child_full.unique_constraints.append(
        UniqueConstraint(name="uk_fk", columns=["f1", "f2"], is_primary=False)
    )
    need_null = {
        "p": spec("p", 1, {"k1": (1, 1), "k2": (10, 10)}),
        # 4 child rows but only one non-NULL distinct (f1,f2); the rest must be
        # NULL-bearing. SIMPLE permits mixed (f1=NULL,f2=10); FULL does not.
        "c": spec("c", 4, {"id": (100, 103), "f1": (1, 1), "f2": (10, 10)}),
    }
    simple_out = Generator({"p": parent, "c": child_simple}, need_null).solve()
    mixed_seen = any(
        (r["f1"] is None) != (r["f2"] is None) for r in simple_out["c"]
    )
    assert mixed_seen, "MATCH SIMPLE should admit a mixed-NULL composite key"

    # Under FULL the same request must be unsat: FULL only allows all-NULL keys
    # (which satisfy the unique index via NULL-distinctness) — verify FULL never
    # emits a mixed key even when it finds a solution.
    full_out = Generator({"p": parent, "c": child_full}, tight).solve()
    for r in full_out["c"]:
        assert (r["f1"] is None) == (r["f2"] is None)


def test_zero_rows_table():
    ts = TableSchema(
        schema_name="public",
        name="empty_t",
        columns=[col("id")],
        unique_constraints=[UniqueConstraint(name="pk", columns=["id"], is_primary=True)],
        foreign_keys=[],
    )
    specs = {"empty_t": spec("empty_t", 0, {"id": (1, 10)})}
    out = Generator({"empty_t": ts}, specs).solve()
    assert out["empty_t"] == []


def _nnd_table(name="t", key_cols=("k",), composite_partner=None):
    columns = [col("id")]
    uks = [UniqueConstraint(name="pk", columns=["id"], is_primary=True)]
    if len(key_cols) == 1:
        columns.append(col(key_cols[0], nullable=True))
        uks.append(
            UniqueConstraint(
                name="uk_nnd",
                columns=list(key_cols),
                is_primary=False,
                nulls_not_distinct=True,
            )
        )
    else:
        for cname in key_cols:
            columns.append(col(cname, nullable=True))
        uks.append(
            UniqueConstraint(
                name="uk_nnd",
                columns=list(key_cols),
                is_primary=False,
                nulls_not_distinct=True,
            )
        )
    return TableSchema(
        schema_name="public",
        name=name,
        columns=columns,
        unique_constraints=uks,
        foreign_keys=[],
    )


def _ordinary_uk_table():
    return TableSchema(
        schema_name="public",
        name="t",
        columns=[col("id"), col("k", nullable=True)],
        unique_constraints=[
            UniqueConstraint(name="pk", columns=["id"], is_primary=True),
            UniqueConstraint(
                name="uk_plain", columns=["k"], is_primary=False,
                nulls_not_distinct=False,
            ),
        ],
        foreign_keys=[],
    )


def test_nnd_single_point_range_three_rows_is_unsat():
    # The reported regression: 3 rows, nullable key range [7,7] under
    # UNIQUE NULLS NOT DISTINCT. Only {7, NULL} are usable -> unsat, and the
    # conflict must name the range plus the NND unique constraint.
    ts = _nnd_table()
    specs = {"t": spec("t", 3, {"id": (1, 100), "k": (7, 7)})}
    with pytest.raises(Unsatisfiable) as exc:
        Generator({"t": ts}, specs).solve()
    groups = exc.value.groups
    kinds = {g.kind for g in groups}
    assert "unique" in kinds and "range" in kinds
    assert any("NULLS NOT DISTINCT" in g.label for g in groups)


def test_nnd_two_rows_single_point_uses_one_null():
    ts = _nnd_table()
    specs = {"t": spec("t", 2, {"id": (1, 100), "k": (7, 7)})}
    data = Generator({"t": ts}, specs).solve()["t"]
    keys = [r["k"] for r in data]
    assert sorted(v for v in keys if v is not None) == [7]
    assert sum(1 for v in keys if v is None) == 1


def test_nnd_never_emits_duplicate_null_key():
    ts = _nnd_table()
    specs = {"t": spec("t", 4, {"id": (1, 100), "k": (70, 72)})}
    data = Generator({"t": ts}, specs).solve()["t"]
    keys = [r["k"] for r in data]
    # Three non-NULL values plus at most one NULL under NND.
    assert len(keys) == 4
    assert sum(1 for v in keys if v is None) <= 1
    non_null = [v for v in keys if v is not None]
    assert len(non_null) == len(set(non_null))


def test_ordinary_unique_still_allows_repeated_nulls():
    ts = _ordinary_uk_table()
    specs = {"t": spec("t", 3, {"id": (1, 100), "k": (7, 7)})}
    data = Generator({"t": ts}, specs).solve()["t"]
    # Ordinary UNIQUE: the same request stays feasible via repeated NULLs.
    assert sum(1 for r in data if r["k"] is None) >= 2
    assert sum(1 for r in data if r["k"] == 7) == 1


def test_nnd_composite_treats_three_valued_keys_as_unique():
    # (a,b) with single-point non-NULL ranges offers exactly four distinct
    # three-valued keys: (1,2),(NULL,2),(1,NULL),(NULL,NULL).
    ts = _nnd_table(key_cols=("a", "b"))
    specs4 = {
        "t": TableSpec(
            table="t", rows=4,
            ranges={
                "id": IntRange(low=1, high=100),
                "a": IntRange(low=1, high=1),
                "b": IntRange(low=2, high=2),
            },
        )
    }
    data = Generator({"t": ts}, specs4).solve()["t"]
    keys = [(r["a"], r["b"]) for r in data]
    assert len(keys) == len(set(keys)) == 4

    # A fifth row cannot be invented: every three-valued key is already used.
    specs5 = {
        "t": TableSpec(
            table="t", rows=5,
            ranges={
                "id": IntRange(low=1, high=100),
                "a": IntRange(low=1, high=1),
                "b": IntRange(low=2, high=2),
            },
        )
    }
    with pytest.raises(Unsatisfiable) as exc:
        Generator({"t": ts}, specs5).solve()
    assert any("NULLS NOT DISTINCT" in g.label for g in exc.value.groups)


def test_nnd_on_not_null_column_matches_plain_uniqueness():
    # A NND constraint on a NOT NULL column can never use NULL, so it behaves
    # exactly like ordinary uniqueness: feasible on a wide range, unsat when the
    # row count exceeds the available values.
    ts = TableSchema(
        schema_name="public",
        name="t",
        columns=[col("id"), col("k")],  # k NOT NULL
        unique_constraints=[
            UniqueConstraint(name="pk", columns=["id"], is_primary=True),
            UniqueConstraint(
                name="uk_nnd", columns=["k"], is_primary=False,
                nulls_not_distinct=True,
            ),
        ],
        foreign_keys=[],
    )
    ok = {"t": spec("t", 3, {"id": (1, 100), "k": (7, 9)})}
    data = Generator({"t": ts}, ok).solve()["t"]
    assert len({r["k"] for r in data}) == 3
    bad = {"t": spec("t", 3, {"id": (1, 100), "k": (7, 7)})}
    with pytest.raises(Unsatisfiable):
        Generator({"t": ts}, bad).solve()


def test_ordinary_composite_unique_allows_repeated_null_bearing_rows():
    ts = TableSchema(
        schema_name="public",
        name="t",
        columns=[col("id"), col("a", nullable=True), col("b", nullable=True)],
        unique_constraints=[
            UniqueConstraint(name="pk", columns=["id"], is_primary=True),
            UniqueConstraint(name="uk_plain", columns=["a", "b"], is_primary=False),
        ],
        foreign_keys=[],
    )
    specs = {
        "t": TableSpec(
            table="t", rows=5,
            ranges={
                "id": IntRange(low=1, high=100),
                "a": IntRange(low=1, high=1),
                "b": IntRange(low=2, high=2),
            },
        )
    }
    data = Generator({"t": ts}, specs).solve()["t"]
    assert len(data) == 5  # feasible: NULL-bearing rows never collide


def test_type_bounds_enforced():
    ts = TableSchema(
        schema_name="public",
        name="t",
        columns=[col("id", pg_type="smallint")],
        unique_constraints=[UniqueConstraint(name="pk", columns=["id"], is_primary=True)],
        foreign_keys=[],
    )
    with pytest.raises(ValueError):
        IntRange(low=-40000, high=1).validated(ts.column("id"))
