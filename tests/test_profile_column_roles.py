"""Profile role maps round-trip through the stored parallel column lists.
Ignored columns remain part of profile identity so matching includes every column
the user reviewed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.widgets import analysis_config_widgets as acw


def _profile(**cfg):
    return lambda *a, **k: cfg


# Legacy configs may list image_name in both fields; read it once as categorical.
PDL1 = {
    "unique_row_id_col": "cell_id",
    "fov_name_col": "image_name",
    "categorical_cols": ["treatment", "image_name"],
    "all_numerical_features": ["n.t1.mean", "Area"],
    "ignored_cols": ["notes"],
}


# ------------------------------------------------------- lists -> role map

def test_every_stored_list_becomes_a_role(monkeypatch):
    monkeypatch.setattr(acw, "_get_current_profile", lambda: "p")
    monkeypatch.setattr(acw, "_get_profile_config", _profile(**PDL1))

    assert acw.profile_column_roles() == {
        "cell_id": "row_id",
        "image_name": "categorical",
        "treatment": "categorical",
        "n.t1.mean": "numerical",
        "Area": "numerical",
        "notes": "ignore",
    }


def test_a_legacy_fov_column_reads_back_as_an_ordinary_categorical(monkeypatch):
    """A legacy FOV column becomes categorical and remains part of the profile's known columns."""
    monkeypatch.setattr(acw, "_get_current_profile", lambda: "p")
    monkeypatch.setattr(acw, "_get_profile_config", _profile(**PDL1))

    assert acw.profile_column_roles()["image_name"] == "categorical"


def test_a_migrated_profile_keeps_a_fov_column_listed_nowhere_else(monkeypatch):
    """Legacy flat configs retain their FOV name as an ordinary categorical."""
    monkeypatch.setattr(acw, "_get_current_profile", lambda: "p")
    monkeypatch.setattr(acw, "_get_profile_config", _profile(
        unique_row_id_col="cell_id", fov_name_col="image_name",
        categorical_cols=["treatment"], all_numerical_features=["feat"]))

    roles = acw.profile_column_roles()
    assert roles["image_name"] == "categorical"
    assert acw.profile_known_columns() == {"cell_id", "image_name", "treatment", "feat"}


def test_a_column_in_two_lists_resolves_by_precedence(monkeypatch):
    """Parallel lists can express nonsense; the view must still be a function."""
    monkeypatch.setattr(acw, "_get_current_profile", lambda: "p")
    monkeypatch.setattr(acw, "_get_profile_config", _profile(
        unique_row_id_col="dup", fov_name_col="",
        categorical_cols=["dup"], all_numerical_features=["dup"],
        ignored_cols=["dup"]))

    assert acw.profile_column_roles()["dup"] == "row_id"


def test_an_empty_profile_has_no_columns(monkeypatch):
    """A profile created in the sidebar and never saved is a reachable state."""
    monkeypatch.setattr(acw, "_get_current_profile", lambda: "p")
    monkeypatch.setattr(acw, "_get_profile_config", _profile())

    assert acw.profile_column_roles() == {}


def test_a_blank_identifier_names_no_column(monkeypatch):
    """Both identifiers are optional, so "" must not become a column called ""."""
    monkeypatch.setattr(acw, "_get_current_profile", lambda: "p")
    monkeypatch.setattr(acw, "_get_profile_config", _profile(
        unique_row_id_col="", fov_name_col="",
        categorical_cols=["treatment"], all_numerical_features=["feat"]))

    assert "" not in acw.profile_column_roles()


def test_a_profile_saved_before_ignored_cols_existed_still_loads(monkeypatch):
    """Profiles without ignored_cols read it as an empty list."""
    monkeypatch.setattr(acw, "_get_current_profile", lambda: "p")
    monkeypatch.setattr(acw, "_get_profile_config", _profile(
        unique_row_id_col="cell_id", fov_name_col="",
        categorical_cols=["treatment"], all_numerical_features=["feat"]))

    roles = acw.profile_column_roles()
    assert roles == {"cell_id": "row_id", "treatment": "categorical",
                     "feat": "numerical"}


# ------------------------------------------------------- role map -> lists

def test_apply_writes_each_role_to_its_list():
    cfg = {}
    acw.apply_column_roles(cfg, {
        "cell_id": "row_id",
        "well": "categorical",
        "treatment": "categorical",
        "feat": "numerical",
        "notes": "ignore",
    })

    assert cfg["unique_row_id_col"] == "cell_id"
    assert cfg["all_numerical_features"] == ["feat"]
    assert cfg["ignored_cols"] == ["notes"]
    # A field-of-view column lands here like any other category -- that is what
    # stringifies it, fills "N/A" and makes it filterable.
    assert set(cfg["categorical_cols"]) == {"treatment", "well"}
    # Saving writes the FOV column only to categorical_cols.
    assert "fov_name_col" not in cfg


def test_apply_clears_an_identifier_that_no_longer_has_a_column():
    """Unassigning Row ID must blank the name, not leave the previous one behind."""
    cfg = {"unique_row_id_col": "cell_id"}
    acw.apply_column_roles(cfg, {"treatment": "categorical", "feat": "numerical"})

    assert cfg["unique_row_id_col"] == ""


def test_roles_round_trip_through_the_stored_lists(monkeypatch):
    """The two functions are inverses on any map the review table can produce."""
    roles = {"cell_id": "row_id", "image_name": "categorical", "treatment": "categorical",
             "n.t1.mean": "numerical", "Area": "numerical", "notes": "ignore"}
    cfg = {}
    acw.apply_column_roles(cfg, roles)

    monkeypatch.setattr(acw, "_get_current_profile", lambda: "p")
    monkeypatch.setattr(acw, "_get_profile_config", lambda *a, **k: cfg)
    assert acw.profile_column_roles() == roles


# ------------------------------------------------- what matching compares

def test_all_profile_columns_reads_every_profile_not_just_the_active_one(monkeypatch):
    """Column maps include every saved profile, independent of current_profile."""
    monkeypatch.setattr(acw, "load_config", lambda *a, **k: {
        "current_profile": "iris",
        "profiles": {
            "pdl1": dict(PDL1),
            "iris": {"categorical_cols": ["species"],
                     "all_numerical_features": ["sepal_length"]},
        },
    })

    assert acw.all_profile_columns() == {
        "pdl1": {"cell_id", "image_name", "treatment", "n.t1.mean", "Area", "notes"},
        "iris": {"species", "sepal_length"},
    }


def test_the_store_feeds_the_matching_functions_directly(monkeypatch):
    """Pass profile column sets to ranking without adding config dependencies to matching."""
    from src import profile_matching

    monkeypatch.setattr(acw, "load_config", lambda *a, **k: {
        "profiles": {
            "iris": {"categorical_cols": ["species"]},
            "pdl1": dict(PDL1),
        },
    })

    fits = profile_matching.rank_profiles(
        {"cell_id", "image_name", "treatment", "n.t1.mean", "Area", "notes"},
        acw.all_profile_columns())
    assert [fit.name for fit in fits] == ["pdl1", "iris"]
    assert fits[0].is_exact


def test_known_columns_include_the_ignored_ones(monkeypatch):
    """Known columns include ignored columns as well as analysis roles."""
    monkeypatch.setattr(acw, "_get_current_profile", lambda: "p")
    monkeypatch.setattr(acw, "_get_profile_config", _profile(**PDL1))

    assert acw.profile_known_columns() == {
        "cell_id", "image_name", "treatment", "n.t1.mean", "Area", "notes"}
