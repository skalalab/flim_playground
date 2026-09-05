"""Match uploaded columns to saved analysis profiles.

Auto-apply only a unique profile with the same nonempty column set. Rank other
profiles for user selection, excluding those with no shared columns from the
chooser. Callers supply `{name: column set}` mappings; this module reads no config
and imports no UI code.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileFit:
    """A profile's shared, missing, and newly uploaded column names."""

    name: str
    shared: tuple    # the profile knows these and the file has them
    missing: tuple   # the profile knows these, the file does not have them
    new: tuple       # the file has these, the profile has never seen them

    @property
    def is_exact(self):
        """Whether file and profile have identical, nonempty column sets."""
        return bool(self.shared) and not self.missing and not self.new


def compare_columns(name, file_cols, profile_cols):
    """Compare file columns to profile columns, returning sorted tuples in ProfileFit."""
    profile_cols, file_cols = set(profile_cols), set(file_cols)
    return ProfileFit(
        name=name,
        shared=tuple(sorted(profile_cols & file_cols)),
        missing=tuple(sorted(profile_cols - file_cols)),
        new=tuple(sorted(file_cols - profile_cols)),
    )


def rank_profiles(file_cols, profiles):
    """Rank every profile by most shared columns, then fewest missing, then name.

    Ranking orders the chooser without selecting a profile or filtering candidates.
    """
    fits = [compare_columns(name, file_cols, cols) for name, cols in profiles.items()]
    return sorted(fits, key=lambda fit: (-len(fit.shared), len(fit.missing), fit.name))


def exact_match(file_cols, profiles):
    """Return the unique exact profile's name, or None for zero or multiple matches."""
    hits = [name for name, cols in profiles.items()
            if compare_columns(name, file_cols, cols).is_exact]
    return hits[0] if len(hits) == 1 else None


def chooser_options(file_cols, profiles, new_profile_label):
    """List ranked profiles sharing at least one column, then new_profile_label."""
    ranked = [fit.name for fit in rank_profiles(file_cols, profiles) if fit.shared]
    return ranked + [new_profile_label]


def chooser_is_needed(applied, file_cols, profiles):
    """Keep the chooser visible until the applied profile exactly matches the file."""
    if not applied:
        return True
    return not compare_columns(applied, file_cols, profiles.get(applied, ())).is_exact
