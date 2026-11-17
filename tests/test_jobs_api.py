"""End-to-end tests: jobs, SQL rendering, real isolated-database loading."""
from __future__ import annotations

import json
import zipfile
from io import BytesIO

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.jobs import STORE, package_zip, run_job
from app.main import app
from app.models import GenerateRequest, JobStatus

client = TestClient(app)


def _load(name):
    raw = json.load(open(f"/workspace/examples/{name}"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _run(req_dict):
    req = GenerateRequest(**req_dict)
    jid = STORE.create()
    run_job(jid, req)
    return jid, STORE.get(jid)


def test_healthz():
    assert client.get("/healthz").json() == {"status": "ok"}


def test_catalog_listing_and_structure(source_dsn):
    r = client.get("/schemas/public/tables")
    assert r.status_code == 200
    assert "employee" in r.json()["base_tables"]
    r2 = client.get("/schemas/public/tables/project/structure")
    body = r2.json()
    assert body["supported"] is True
    fks = body["structure"]["foreign_keys"]
    assert any(f["match"] == "MATCH FULL" for f in fks)


def test_cyclic_four_table_job_succeeds_and_publishes():
    jid, job = _run(_load("request_cyclic_four_tables.json"))
    assert job.status == JobStatus.SUCCEEDED
    assert set(job.selected_tables) == {
        "dept",
        "employee",
        "project",
        "assignment",
    }
    pkg = STORE.package(jid)
    assert set(pkg) == {
        "01_schema.sql",
        "02_data.sql",
        "03_constraints.sql",
        "manifest.json",
        "README.md",
    }
    # Constraints remain NOT DEFERRABLE; nothing was disabled or made deferred.
    fk_ddl = [
        ln for ln in pkg["03_constraints.sql"].splitlines()
        if ln.startswith("ALTER TABLE")
    ]
    assert fk_ddl and all("NOT DEFERRABLE" in ln for ln in fk_ddl)
    assert all("INITIALLY DEFERRED" not in ln for ln in fk_ddl)
    # FKs are post-data ALTERs so the cycle loads without deferral tricks.
    assert all("ALTER TABLE" in ln for ln in fk_ddl)
    # Zip is downloadable and well formed.
    zf = zipfile.ZipFile(BytesIO(package_zip(pkg)))
    assert set(zf.namelist()) == set(pkg)


def test_download_blocked_until_success():
    jid, job = _run(_load("request_unsat_conflict.json"))
    assert job.status == JobStatus.FAILED
    assert STORE.package(jid) is None
    r = client.get(f"/jobs/{jid}/download")
    assert r.status_code == 409


def test_unsat_reports_conflicting_rows_ranges_or_constraints():
    jid, job = _run(_load("request_unsat_conflict.json"))
    conflict = job.conflict.model_dump()
    assert conflict["kind"] == "no_solution"
    joined = json.dumps(conflict, ensure_ascii=False)
    assert "employee_dept_fk" in joined
    assert "500" in joined and "100" in joined
    assert "fk" in conflict["details"]["kinds"]


def test_fk_closure_expands_target_table():
    # Only project requested; the transitive FK closure pulls in dept and
    # employee (dept.mgr_emp -> employee), staying within the 4-table cap.
    jid, job = _run(_load("request_closure_single.json"))
    assert job.status == JobStatus.SUCCEEDED
    assert set(job.selected_tables) == {"project", "dept", "employee"}


def test_generated_package_loads_in_fresh_empty_database(
    fresh_verify_database,
):
    jid, job = _run(_load("request_cyclic_four_tables.json"))
    assert job.status == JobStatus.SUCCEEDED
    dsn, _ = fresh_verify_database
    pkg = STORE.package(jid)
    with psycopg.connect(dsn, autocommit=False) as conn:
        for f in ("01_schema.sql", "02_data.sql", "03_constraints.sql"):
            conn.cursor().execute(pkg[f])
        conn.commit()
        cur = conn.cursor()
        for t, expected in [
            ("dept", 3),
            ("employee", 5),
            ("project", 4),
            ("assignment", 6),
        ]:
            cur.execute(f"SELECT count(*) FROM public.{t}")
            assert cur.fetchone()[0] == expected


def test_zero_rows_request(source_dsn):
    req = {
        "schema_name": "public",
        "tables": [
            {
                "table": "project",
                "rows": 0,
                "ranges": {
                    "project_id": {"low": 1, "high": 9},
                    "dept_id": {"low": 1, "high": 9},
                    "dept_code": {"low": 1, "high": 9},
                    "budget": {"low": 1, "high": 9},
                },
            }
        ],
    }
    jid, job = _run(req)
    # closure pulls dept (default 5 rows); project itself stays empty
    assert job.status == JobStatus.SUCCEEDED
    pkg = STORE.package(jid)
    assert "no INSERTs" in pkg["02_data.sql"]


def test_validation_rejects_bad_interval(source_dsn):
    req = {
        "schema_name": "public",
        "tables": [
            {
                "table": "dept",
                "rows": 1,
                "ranges": {
                    "dept_id": {"low": 10, "high": 5},  # low > high
                    "dept_code": {"low": 1, "high": 2},
                    "parent_dept": {"low": 1, "high": 9},
                    "mgr_emp": {"low": 1, "high": 9},
                },
            }
        ],
    }
    # pydantic-level or service-level rejection
    r = client.post("/jobs", json=req)
    if r.status_code == 202:
        jid = r.json()["job_id"]
        job = STORE.get(jid)
        assert job.status == JobStatus.FAILED
        assert job.conflict.kind == "request"
    else:
        assert r.status_code == 422


def test_more_than_max_tables_rejected():
    r = client.post(
        "/jobs",
        json={
            "schema_name": "public",
            "tables": [
                {"table": f"t{i}", "rows": 1, "ranges": {}} for i in range(5)
            ],
        },
    )
    assert r.status_code == 422


def test_nnd_conflict_job_fails_without_package():
    # Reported regression shape: 3 rows on a nullable NND key pinned to [7,7].
    jid, job = _run(_load("request_nnd_conflict.json"))
    assert job.status == JobStatus.FAILED
    conflict = job.conflict.model_dump()
    assert conflict["kind"] == "no_solution"
    joined = json.dumps(conflict, ensure_ascii=False)
    assert "NULLS NOT DISTINCT" in joined
    assert "[7, 7]" in joined
    assert "ticket_locator_nnd_uk" in joined
    # No relaxed package is ever published for an unsatisfiable request.
    assert STORE.package(jid) is None
    r = client.get(f"/jobs/{jid}/download")
    assert r.status_code == 409


def test_nnd_feasible_job_publishes_and_loads(fresh_verify_database):
    jid, job = _run(_load("request_nnd_feasible.json"))
    assert job.status == JobStatus.SUCCEEDED
    pkg = STORE.package(jid)
    # The rebuilt DDL keeps the source constraint exactly: no silent relaxation
    # to an ordinary UNIQUE.
    assert "UNIQUE NULLS NOT DISTINCT" in pkg["01_schema.sql"]
    manifest = json.loads(pkg["manifest.json"])
    unique = {
        u["name"]: u
        for u in manifest["constraints"]["ticket"]["unique"]
    }
    assert unique["ticket_locator_nnd_uk"]["nulls_not_distinct"] is True
    assert unique["ticket_locator_uk"]["nulls_not_distinct"] is False

    # Real load into a freshly created empty database must succeed, and the
    # NND constraint must reject a duplicate NULL while ordinary UNIQUE
    # accepts one.
    dsn, _ = fresh_verify_database
    with psycopg.connect(dsn, autocommit=False) as conn:
        for f in ("01_schema.sql", "02_data.sql", "03_constraints.sql"):
            conn.cursor().execute(pkg[f])
        conn.commit()
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM public.ticket")
        assert cur.fetchone()[0] == 4
        cur.execute("SELECT count(*) FROM public.ticket WHERE locator_nnd IS NULL")
        null_rows = cur.fetchone()[0]
        # NND: at most one row may carry a NULL locator_nnd.
        assert null_rows <= 1
        cur.execute("SELECT locator_nnd FROM public.ticket WHERE locator_nnd IS NOT NULL")
        used_nnd = {r[0] for r in cur.fetchall()}
        free_nnd = next(v for v in range(70, 80) if v not in used_nnd)
        if null_rows == 0:
            # First NULL-bearing NND key: legal under NULLS NOT DISTINCT.
            conn.cursor().execute(
                "INSERT INTO public.ticket (ticket_id, tenant_ref, "
                "locator_nnd, locator_uk) VALUES (997, 9, NULL, 91)"
            )
            conn.commit()
        # A second all-NULL NND key must be rejected with 23505 regardless of
        # whether the first NULL came from the sample or the insert above.
        with pytest.raises(psycopg.Error) as exc:
            conn.cursor().execute(
                "INSERT INTO public.ticket (ticket_id, tenant_ref, "
                "locator_nnd, locator_uk) VALUES (999, 9, NULL, 92)"
            )
        assert exc.value.sqlstate == "23505"
        conn.rollback()
        # Ordinary UNIQUE: another NULL locator_uk is accepted. Use an unused
        # NND value and fresh tenant_ref so only ordinary-UNIQUE NULL semantics
        # are being exercised.
        conn.cursor().execute(
            "INSERT INTO public.ticket (ticket_id, tenant_ref, "
            "locator_nnd, locator_uk) VALUES (998, 8, %s, NULL)",
            (free_nnd,),
        )
        conn.commit()


def test_nnd_negative_probes_recorded_in_verify_log():
    jid, job = _run(_load("request_nnd_feasible.json"))
    assert job.status == JobStatus.SUCCEEDED
    log = "\n".join(job.log)
    assert "ticket_locator_nnd_uk 重复NULL" in log
    assert "普通UNIQUE 允许重复NULL" in log
