"""Behavioral tests for Data_Loading (P0-C).

These cover the DB-independent surface: the ORM exposes the PostGIS column, the
engine connection string is sourced from the environment, and the distance
ranking is correct on a known fixture. Tests requiring a live Postgres+PostGIS
service are out of scope here (skipped via DATABASE_URL conventions in CI).
"""

import math

import Data_Loading as dl


def test_orm_exposes_point_geom():
    """P0-C: the TrackingData ORM model must define the PostGIS point_geom column."""
    assert hasattr(dl.TrackingData, 'point_geom')
    assert 'point_geom' in dl.TrackingData.__table__.columns


def test_connection_string_reads_from_env(monkeypatch):
    monkeypatch.setenv('DATABASE_URL', 'postgresql://u:p@myhost:5432/mydb')
    assert dl.get_connection_string() == 'postgresql://u:p@myhost:5432/mydb'


def test_connection_string_default_when_env_absent(monkeypatch):
    monkeypatch.delenv('DATABASE_URL', raising=False)
    assert dl.get_connection_string().startswith('postgresql://')


def test_engine_uses_env_connection(monkeypatch):
    """create_sqlalchemy_engine builds an engine from the env URL. Use a sqlite
    URL so no external DB driver is required to exercise the full create path."""
    monkeypatch.setenv('DATABASE_URL', 'sqlite:///:memory:')
    engine = dl.create_sqlalchemy_engine()
    assert engine is not None
    assert str(engine.url) == 'sqlite:///:memory:'


def test_distance_calc_known_fixture():
    """Distances and nearest-first ordering must be correct on a known fixture."""
    basketball = (0.0, 0.0)
    players = [
        (10, 3.0, 4.0),   # distance 5
        (20, 0.0, 1.0),   # distance 1
        (30, 6.0, 8.0),   # distance 10
    ]

    ranked = dl.calculate_point_distances(basketball, players)

    # Sorted nearest-first by id
    assert [pid for pid, _ in ranked] == [20, 10, 30]
    # Correct Euclidean magnitudes
    assert math.isclose(dict(ranked)[20], 1.0)
    assert math.isclose(dict(ranked)[10], 5.0)
    assert math.isclose(dict(ranked)[30], 10.0)
