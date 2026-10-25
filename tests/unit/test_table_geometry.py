"""Lines, segments, blocks and side-by-side splitting (Fase 3.1; targets F4/F5)."""
from __future__ import annotations

import pymupdf
import pytest

from app.services.candidates.context import build_page
from app.services.candidates.schema import Token
from app.services.candidates.tables.geometry import build_lines, split_blocks, text_blocks
from app.tools.table_eval.synthetic import build_fixture, to_evidence_page


def _page(name: str, mode: str = "vector", index: int = 0):
    document = pymupdf.open(stream=build_fixture(name).pdf, filetype="pdf")
    return build_page(to_evidence_page(document[index], index + 1, mode))


def _token(text, x0, y0, x1, y1):
    return Token(text, (x0, y0, x1, y1))


def test_tokens_with_jittered_centres_share_a_line_and_distant_ones_do_not():
    tokens = [_token("A", 0.1, 0.100, 0.15, 0.112), _token("B", 0.2, 0.102, 0.25, 0.114), _token("C", 0.1, 0.140, 0.15, 0.152)]
    lines = build_lines(tokens, 0.7071)
    assert [line.text for line in lines] == ["A B", "C"]


def test_a_tall_title_does_not_drag_neighbouring_small_text_into_its_line():
    tokens = [_token("TITLE", 0.1, 0.05, 0.4, 0.09)] + [_token(str(i), 0.1 + i * 0.05, 0.12, 0.13 + i * 0.05, 0.132) for i in range(4)]
    lines = build_lines(tokens, 0.7071)
    assert len(lines) == 2


def test_segments_split_on_wide_gaps_only():
    tokens = [_token("Harga", 0.50, 0.1, 0.56, 0.112), _token("Satuan", 0.565, 0.1, 0.63, 0.112), _token("Jumlah", 0.80, 0.1, 0.87, 0.112)]
    (line,) = build_lines(tokens, 0.7071)
    assert [seg.text for seg in line.segs] == ["Harga Satuan", "Jumlah"]


def test_segment_gap_scales_with_text_size_not_with_page_units():
    small = [_token("a", 0.10, 0.1, 0.12, 0.106), _token("b", 0.135, 0.1, 0.155, 0.106)]  # 0.015 gap, tiny text
    large = [_token("a", 0.10, 0.1, 0.20, 0.140), _token("b", 0.215, 0.1, 0.315, 0.140)]  # 0.015 gap, huge text
    assert len(build_lines(small, 0.7071)[0].segs) == 2
    assert len(build_lines(large, 0.7071)[0].segs) == 1


def test_header_of_constructed_table_has_one_segment_per_column():
    page = _page("single_unruled")
    lines = build_lines(page.tokens, page.aspect)
    header = next(line for line in lines if line.text.startswith("No Deskripsi"))
    assert [seg.text for seg in header.segs] == ["No", "Deskripsi", "Qty", "Harga Satuan", "Jumlah"]


@pytest.mark.parametrize("mode", ["vector", "phrase"])
def test_stacked_tables_fall_in_different_blocks(mode):
    page = _page("stacked_two_tables", mode)
    blocks, _ = text_blocks(page.tokens, page.aspect)
    texts = [" | ".join(line.text for line in block.lines) for block in blocks]
    first = next(i for i, t in enumerate(texts) if "Deskripsi" in t)
    second = next(i for i, t in enumerate(texts) if "Keterangan" in t)
    assert first != second


@pytest.mark.parametrize("mode", ["vector", "phrase"])
def test_side_by_side_tables_are_split_into_left_and_right_blocks(mode):
    page = _page("side_by_side", mode)
    blocks, _ = text_blocks(page.tokens, page.aspect)
    tables = [block for block in blocks if len(block.lines) >= 4]
    assert len(tables) == 2
    left, right = sorted(tables, key=lambda b: b.bbox[0])
    assert left.bbox[2] < right.bbox[0]
    assert "Tujuan" in " ".join(l.text for l in left.lines) and "Asal" in " ".join(l.text for l in right.lines)


@pytest.mark.parametrize("name", ["single_ruled", "single_unruled", "wrapped_cells", "blank_column", "header_two_line", "items_total_rows_inside"])
def test_a_single_table_is_never_split_horizontally(name):
    page = _page(name)
    blocks, _ = text_blocks(page.tokens, page.aspect)
    big = [b for b in blocks if len(b.lines) >= 4]
    assert len(big) == 1


def test_vertical_gap_separates_blocks_but_row_padding_does_not():
    page = _page("single_ruled")
    lines = build_lines(page.tokens, page.aspect)
    height = sorted(t.bbox[3] - t.bbox[1] for t in page.tokens)[len(page.tokens) // 2]
    blocks = split_blocks(lines, height)
    table_block = max(blocks, key=lambda b: len(b.lines))
    assert len(table_block.lines) == 7  # header + 6 padded rows stay together
