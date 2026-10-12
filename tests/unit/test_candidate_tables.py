from pathlib import Path

from app.services.candidates.context import build_page
from app.services.candidates.line_items import LineItemBuilder
from app.services.candidates.tables import TokenGridTableProvider


def _page():
    rows = [
        (200, [(40, "No"), (100, "Nama Barang"), (330, "Koli"), (440, "Berat (Kg)"), (620, "Catatan")]),
        (230, [(40, "1"), (100, "Dus Mie Instan"), (330, "10"), (440, "25,5"), (620, "Fragile")]),
        (260, [(40, "2"), (100, "Air Mineral 600ml"), (330, "5"), (440, "12,0")]),
    ]
    lines = []
    for y, cells in rows:
        tokens = []
        for x, text in cells:
            words = text.split()
            for offset, word in enumerate(words):
                start = x + offset * 10
                tokens.append({"text": word, "confidence": .95, "bbox_norm": [start, y, start + 8, y + 14]})
        lines.append({"text": " ".join(token["text"] for token in tokens), "bbox_norm": [0, y, 800, y + 14], "tokens": tokens})
    return build_page({"page_number": 1, "engine_used": "pymupdf_vector", "lines": lines})


def test_token_grid_and_line_item_outputs():
    config = Path("config/candidates")
    tables = TokenGridTableProvider(config / "line_item_aliases.yaml", config / "abbreviations.yaml").tables(_page(), "SURAT_JALAN", "D1")
    assert len(tables) == 1
    assert len(tables[0].columns) == 5
    assert len(tables[0].rows) == 2
    result = LineItemBuilder(config).build("D1", "SURAT_JALAN", tables)
    assert result["columns"]
    assert any(row["li_field"] == "quantity" for row in result["columns"])
    assert any(row["value_parsed"] == "10" for row in result["cells"])