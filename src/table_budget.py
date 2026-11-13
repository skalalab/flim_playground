"""How much memory an upload may cost here, and the words for refusing one.

No internal imports: the reader and the spreadsheet child both use it.
"""
import psutil

# An upload may use this share of the memory that is free when it arrives.
MEMORY_SHARE = 0.5
# Peak memory of a delimited table through read, review and interpretation, as bytes per
# cell plus bytes per row (the row ID and per-row overheads). Fitted to three measured
# 1M-row-scale tables: 13 mixed columns 1.05 GB, 57 float columns 2.6 GB, 30M x 2 6.6 GB.
# A single per-cell figure had to be 110 to cover the tall table and then refused wide
# numeric tables a laptop can hold.
TEXT_BYTES_PER_CELL = 45
TEXT_BYTES_PER_ROW = 135
# The online demo's container is documented at about 2.7 GB; the host's own free-memory
# figure is not the container's, so the demo counts down from this instead.
HOSTED_MEMORY_BYTES = 2_400_000_000
HOSTED_MAX_CELLS = 5_000_000
# A desktop never refuses a table this small, whatever the OS reports as free.
DESKTOP_MIN_BUDGET_BYTES = 256_000_000
# Below this a workbook child cannot even start: Python plus pandas is about 100 MB.
WORKBOOK_MIN_BYTES = 200_000_000
DESKTOP_WORKBOOK_SECONDS = 600.0
HOSTED_WORKBOOK_SECONDS = 60.0


def memory_budget_bytes(hosted):
    """Bytes one upload may cost right now."""
    if hosted:
        # Shrinks as other visitors' tables fill the server; no floor.
        return int(MEMORY_SHARE * max(0, HOSTED_MEMORY_BYTES - psutil.Process().memory_info().rss))
    return max(DESKTOP_MIN_BUDGET_BYTES, int(MEMORY_SHARE * psutil.virtual_memory().available))


def format_bytes(count):
    return f"{count / 1e9:.1f} GB" if count >= 1e9 else f"{count / 1e6:.0f} MB"


def busy_refusal(filename):
    return (f"Error: the online demo is holding other visitors' tables and cannot open '{filename}' "
            "right now. Try again in a few minutes, or open it in the desktop app.")


def cells_refusal(filename, rows, columns):
    return (f"Error: '{filename}' is too large for the online demo. It has about {rows:,} rows × "
            f"{columns:,} columns ({rows * columns:,} cells), and the demo opens tables of up to "
            f"{HOSTED_MAX_CELLS:,} cells. Filter or split the table, or open it in the desktop app.")


def text_refusal(filename, file_bytes, rows, columns, *, hosted, budget):
    """Why a delimited file of this size is not parsed, or "" when it may be."""
    if hosted and rows * columns > HOSTED_MAX_CELLS:
        return cells_refusal(filename, rows, columns)
    # The server holds the uploaded bytes as well as the table parsed from them.
    need = file_bytes + rows * columns * TEXT_BYTES_PER_CELL + rows * TEXT_BYTES_PER_ROW
    if need <= budget:
        return ""
    if hosted:
        return busy_refusal(filename)
    return (f"Error: '{filename}' is too large to open on this computer. It has about {rows:,} rows × "
            f"{columns:,} columns ({rows * columns:,} cells), which needs about {format_bytes(need)} of "
            f"memory; {format_bytes(budget)} can be used right now. Filter or split the table and "
            "upload it again.")


def workbook_refusal(filename, reason, *, hosted, budget, seconds):
    """The message for a spreadsheet the capped child did not return: memory, time or failed."""
    if reason == "time":
        return (f"Error: '{filename}' was still being read after {seconds:.0f} seconds, so reading "
                "was stopped. Save the table as CSV and upload that.")
    if reason == "failed":
        return (f"Error: '{filename}' could not be opened safely, so it was not read. "
                "Save the table as CSV and upload that.")
    return (f"Error: the first sheet of '{filename}' needs more memory than can be used here "
            f"({format_bytes(budget)}), so it was not opened. If the table itself is small, the sheet "
            "holds content far outside it or very long repeated text: copy the table into a new "
            "workbook, or save it as CSV."
            + (" Larger tables open in the desktop app." if hosted else ""))
