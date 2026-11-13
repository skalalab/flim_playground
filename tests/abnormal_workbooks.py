"""Tiny workbooks, written part by part, that cost far more to read than their size suggests.

Each is a few kilobytes at most. Measured against the in-process reader before the
capped child existed: a far-away valued cell passes 8 GB, 200,000 x 50 ODS repeats cost
1.3 GB, and 3,000 cells pointing at one 100,000-character shared text cost 0.7 GB.
"""
import io
import zipfile

import numpy as np
import pandas as pd

_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG = "http://schemas.openxmlformats.org/package/2006"
_XML = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
_ODS_NS = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
           'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
           'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"')


def _xlsx(sheet_data, shared_strings=None):
    """A one-sheet workbook around the given <sheetData> content."""
    shared_type = ('<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.'
                   'openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>') if shared_strings else ""
    shared_rel = (f'<Relationship Id="rId2" Type="{_REL}/sharedStrings" '
                  'Target="sharedStrings.xml"/>') if shared_strings else ""
    parts = {
        "[Content_Types].xml": (
            f'{_XML}<Types xmlns="{_PKG}/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-'
            'officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-'
            f'officedocument.spreadsheetml.worksheet+xml"/>{shared_type}</Types>'),
        "_rels/.rels": (
            f'{_XML}<Relationships xmlns="{_PKG}/relationships">'
            f'<Relationship Id="rId1" Type="{_REL}/officeDocument" Target="xl/workbook.xml"/></Relationships>'),
        "xl/workbook.xml": (
            f'{_XML}<workbook xmlns="{_MAIN}" xmlns:r="{_REL}">'
            '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>'),
        "xl/_rels/workbook.xml.rels": (
            f'{_XML}<Relationships xmlns="{_PKG}/relationships">'
            f'<Relationship Id="rId1" Type="{_REL}/worksheet" Target="worksheets/sheet1.xml"/>'
            f'{shared_rel}</Relationships>'),
        "xl/worksheets/sheet1.xml": f'{_XML}<worksheet xmlns="{_MAIN}"><sheetData>{sheet_data}</sheetData></worksheet>',
    }
    if shared_strings:
        parts["xl/sharedStrings.xml"] = shared_strings
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in parts.items():
            archive.writestr(name, text)
    return buffer.getvalue()


_SMALL_TABLE = ('<row r="1"><c r="A1" t="inlineStr"><is><t>a</t></is></c>'
                '<c r="B1" t="inlineStr"><is><t>b</t></is></c></row>'
                '<row r="2"><c r="A2"><v>1</v></c><c r="B2"><v>2</v></c></row>')


def far_cell_xlsx(column, row):
    """A 1 x 2 table plus one valued cell far away: the reader allocates the whole rectangle."""
    return _xlsx(f'{_SMALL_TABLE}<row r="{row}"><c r="{column}{row}"><v>3</v></c></row>')


def formatting_only_far_cell_xlsx():
    """The same far cell holding only formatting, as Excel leaves behind: the reader ignores it."""
    return _xlsx(f'{_SMALL_TABLE}<row r="1048576"><c r="XFD1048576" s="0"/></row>')


def long_shared_text_xlsx(cells, length):
    """One long shared text that every cell points at: a small grid, a large table."""
    rows = "".join(f'<row r="{r}"><c r="A{r}" t="s"><v>1</v></c></row>' for r in range(2, cells + 2))
    shared = (f'{_XML}<sst xmlns="{_MAIN}" count="{cells + 1}" uniqueCount="2">'
              f'<si><t>h</t></si><si><t>{"A" * length}</t></si></sst>')
    return _xlsx(f'<row r="1"><c r="A1" t="s"><v>0</v></c></row>{rows}', shared)


def _ods(rows_xml):
    content = (f'<?xml version="1.0" encoding="UTF-8"?><office:document-content {_ODS_NS} office:version="1.2">'
               '<office:body><office:spreadsheet><table:table table:name="Sheet1">'
               f'{rows_xml}</table:table></office:spreadsheet></office:body></office:document-content>')
    manifest = ('<?xml version="1.0" encoding="UTF-8"?><manifest:manifest '
                'xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.2">'
                '<manifest:file-entry manifest:full-path="/" '
                'manifest:media-type="application/vnd.oasis.opendocument.spreadsheet"/>'
                '<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>'
                '</manifest:manifest>')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(zipfile.ZipInfo("mimetype"), "application/vnd.oasis.opendocument.spreadsheet")
        archive.writestr("META-INF/manifest.xml", manifest, zipfile.ZIP_DEFLATED)
        archive.writestr("content.xml", content, zipfile.ZIP_DEFLATED)
    return buffer.getvalue()


def _ods_cell(value, kind="float", repeat=1):
    typed = f'office:value-type="{kind}"' + (f' office:value="{value}"' if kind == "float" else "")
    return f'<table:table-cell table:number-columns-repeated="{repeat}" {typed}><text:p>{value}</text:p></table:table-cell>'


def repeat_ods(rows, columns):
    """Repeat counts describe rows x columns valued cells in under a kilobyte."""
    return _ods(f'<table:table-row>{_ods_cell("h", "string", columns)}</table:table-row>'
                f'<table:table-row table:number-rows-repeated="{rows}">{_ods_cell(1, repeat=columns)}</table:table-row>')


def libreoffice_filler_ods():
    """A 3 x 2 table followed by the repeated empty rows and cells LibreOffice always writes."""
    tail = '<table:table-cell table:number-columns-repeated="16382"/>'
    rows = f'<table:table-row>{_ods_cell("a", "string")}{_ods_cell("b", "string")}{tail}</table:table-row>'
    for i in range(3):
        rows += f'<table:table-row>{_ods_cell(i)}{_ods_cell(i * 2)}{tail}</table:table-row>'
    rows += ('<table:table-row table:number-rows-repeated="1048572">'
             '<table:table-cell table:number-columns-repeated="16384"/></table:table-row>')
    return _ods(rows)


def zip_without_a_workbook():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("hello.txt", "not a workbook")
    return buffer.getvalue()


def mixed_workbook(rows=2_000):
    """An ordinary workbook: text IDs, a numeric header, floats, dates and blanks."""
    frame = pd.DataFrame({
        "id": [f"c{i}" for i in range(rows)],
        7: range(rows),
        "x": np.linspace(0, 1, rows),
        "when": pd.date_range("2026-01-01", periods=rows),
        "note": [None if i % 3 else "a" for i in range(rows)],
    })
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False)
    return buffer.getvalue()
