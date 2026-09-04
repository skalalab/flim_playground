"""Which saved analysis profile fits an uploaded file.

Strict identity: a profile auto-applies only when the columns it has seen are
*exactly* the columns the file has. Containment is deliberately not enough -- a file
carrying every column the profile knows plus one more is not the same dataset, and
auto-applying would drop a real measurement without asking about it.

Everything else goes to the chooser. The ranking there orders candidates; it never
picks one. That is what keeps a fudge factor out of this module: no threshold has to
separate "close enough" from "not close enough", because a human decides either way.
The single cutoff, in `chooser_options`, is not such a threshold -- zero shared columns
is not "far", it is "unrelated", and no tuning can move that line.

Every function takes profiles as a plain `{name: column set}` mapping, so nothing here
reads config, imports streamlit, or knows where a profile is stored. The caller pairs
these with `analysis_config_widgets.all_profile_columns()`.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileFit:
    """How one saved profile lines up against the columns of an uploaded file.

    All three sets are reported, and the UI shows all three: "16 of 18" would be
    ambiguous about which side the 18 counts.
    """

    name: str
    shared: tuple    # the profile knows these and the file has them
    missing: tuple   # the profile knows these, the file does not have them
    new: tuple       # the file has these, the profile has never seen them

    @property
    def is_exact(self):
        """Whether this profile is the one to auto-apply, with nothing left to ask.

        `shared` has to be non-empty, not just the two differences empty: a profile
        created in the sidebar and never filled in knows no columns, and two empty
        sets are equal, so identity alone would let it claim a file. The rule lives
        here rather than in `exact_match` so the chooser, which badges rows with
        this property, cannot show an "exact fit" that auto-apply then refuses.
        """
        return bool(self.shared) and not self.missing and not self.new


def compare_columns(name, file_cols, profile_cols):
    """One profile against one file, as sorted tuples so the result is displayable.

    `file_cols` first, matching rank_profiles and exact_match. Both arguments are sets
    of column names, so a transposed call raises nothing -- it just returns a fit with
    `missing` and `new` exchanged. One order across the module is the only defence.
    """
    profile_cols, file_cols = set(profile_cols), set(file_cols)
    return ProfileFit(
        name=name,
        shared=tuple(sorted(profile_cols & file_cols)),
        missing=tuple(sorted(profile_cols - file_cols)),
        new=tuple(sorted(file_cols - profile_cols)),
    )


def rank_profiles(file_cols, profiles):
    """Every profile, best fit first -- to order the chooser, not to decide it.

    Most shared columns wins, since that is the most assignments the user does not
    have to redo. Fewest missing breaks a tie. Name breaks the remainder, only so the
    list does not reorder itself between runs; a plain sort suffices there because
    this is a stability tie-break rather than a user-meaningful ordering, and the
    natural-sort helper lives in a plotting module this one should not import.

    Nothing is filtered out here: this function orders, and the one cutoff there is
    lives in `chooser_options`, so a caller that wants every profile -- the widget's
    `fits` lookup, which must resolve any name the chooser lists -- still gets one.
    What bounds the length is the profile cap, which is 20.
    """
    fits = [compare_columns(name, file_cols, cols) for name, cols in profiles.items()]
    return sorted(fits, key=lambda fit: (-len(fit.shared), len(fit.missing), fit.name))


def exact_match(file_cols, profiles):
    """The one profile to auto-apply, or None when the user has to choose.

    None covers two situations that mean the same thing to the caller: no profile
    fits, or several do (same columns, different groupings). What counts as a fit is
    ProfileFit.is_exact and nothing else, so this and the chooser can never disagree
    about a given row.
    """
    hits = [name for name, cols in profiles.items()
            if compare_columns(name, file_cols, cols).is_exact]
    return hits[0] if len(hits) == 1 else None


def chooser_options(file_cols, profiles, new_profile_label):
    """What the chooser lists, in order: the ranked profiles, then "make a new one".

    The escape hatch goes last rather than first. The list answers "which of these is
    it?", so the eye should land on the best candidate; a file that matches nothing
    still reads correctly, because the only option left is the last one.

    A profile that shares *no* column with the file is left out. The list answers
    "which of these is it?", and a profile with nothing in common is not a candidate
    under any reading -- picking one would fill the table with the file's own columns
    and nothing else, which is what the last option already does, better. An empty
    profile, saved in the sidebar and never filled in, falls out by the same test.

    The cutoff costs the one thing the chooser is otherwise the only home for: a
    profile sharing nothing with the file in hand can no longer be renamed or deleted
    from this screen. It reappears the moment a file that shares one column with it is
    uploaded, and at the cap it can still be overwritten by name from `Save as`.
    """
    ranked = [fit.name for fit in rank_profiles(file_cols, profiles) if fit.shared]
    return ranked + [new_profile_label]


def chooser_is_needed(applied, file_cols, profiles):
    """Whether the chooser has anything left to ask.

    Only when the profile in force already describes this file exactly is the answer
    no -- and then offering the others would invite the user to write this file's
    column set onto a *second* profile, which is the one state the design forbids.
    A partly-fitting pick is not enough: the list has to stay while the user is still
    deciding, or it would vanish under the cursor on the click that chose it.
    """
    if not applied:
        return True
    return not compare_columns(applied, file_cols, profiles.get(applied, ())).is_exact
