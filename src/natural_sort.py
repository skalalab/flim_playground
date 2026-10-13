"""Natural (numeric-aware) ordering of category values: ``Day 2`` before ``Day 10``.

Shared by the review table, the filters, the plots, and the exported scripts, which
embed these functions with ``inspect.getsource()``. Keep it free of internal imports.
"""
import re


def natural_key(s):
    """Return a tuple for natural sorting: (is_number, number or string)"""
    # Match decimal or integer numbers (including negative numbers)
    match = re.search(r'([-+]?\d*\.\d+|\d+)', s)
    if match:
        return (0, float(match.group(1)), s)
    else:
        return (1, s)

def tuple_natural_key(tup):
    """Return a tuple of natural keys for each element in the tuple."""
    return tuple(natural_key(str(x)) for x in tup)

def natural_tuple_sort(strings, delimiter='::'):
    """Naturally sort each field of delimited strings; callers must supply strings."""
    return sorted(strings, key=lambda x: tuple_natural_key(x.split(delimiter)))
