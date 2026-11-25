from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag.ingestion import (
    booklet_info,
    chunk_id,
    clean_page,
    join_pages,
    join_paragraphs,
    page_at,
    remove_footnote_markers,
    remove_footnotes,
    remove_toc,
    remove_topic_list,
    split_documents,
)


# --- booklet_info ---

def test_booklet_info_parses_filename():
    path = Path("41 A társasági adó legfontosabb szabályai 20260202.pdf")

    assert booklet_info(path) == {
        "booklet": "41",
        "title": "A társasági adó legfontosabb szabályai",
        "published": "2026-02-02",
    }


def test_booklet_info_falls_back_to_file_name():
    assert booklet_info(Path("random.pdf")) == {"title": "random"}


# --- remove_footnotes ---

@pytest.mark.parametrize("footnotes, numbers", [
    (["5 2019. évi XIII. törvény"], {5}),
    (["44Tao. tv. 7. §"], {44}),
    (["12 479-es számú határozat"], {12}),
    (["3 Első lábjegyzet", "4 Második lábjegyzet"], {3, 4}),
])
def test_remove_footnotes_removes_last_block(footnotes, numbers):
    lines = ["Szöveg a törvény5 alapján.", "", *footnotes]

    kept, found = remove_footnotes(lines)

    assert kept == ["Szöveg a törvény5 alapján.", ""]
    assert found == numbers


@pytest.mark.parametrize("last_block", [
    "A társasági adó mértéke 9 százalék.",
    "2026. január 1-jétől új szabály lép életbe.",
])
def test_remove_footnotes_keeps_normal_paragraph(last_block):
    lines = ["Első bekezdés.", "", last_block]

    kept, found = remove_footnotes(lines)

    assert kept == lines
    assert found == set()


# --- remove_footnote_markers ---

def test_remove_footnote_markers_removes_known_marker():
    lines = ["Az alapítványt külön törvény5 szabályozza."]

    assert remove_footnote_markers(lines, {5}) == ["Az alapítványt külön törvény szabályozza."]


def test_remove_footnote_markers_needs_footnote_on_page():
    lines = ["Az alapítványt külön törvény5 szabályozza."]

    assert remove_footnote_markers(lines, set()) == lines


@pytest.mark.parametrize("line", [
    "2019. március óta létezik.",
    "Budapesten 30 százalékának megfelelő mérték.",
    "A Tao. tv. 7. § szerint.",
])
def test_remove_footnote_markers_keeps_real_numbers(line):
    assert remove_footnote_markers([line], {5, 7, 30}) == [line]


# --- remove_toc ---

def test_remove_toc_removes_contents_block():
    lines = [
        "Bevezető szöveg.",
        "Tartalom",
        "1. Bevezetés ........ 3",
        "2. Az adó alanyai ........ 5",
        "",
        "Szöveg.",
    ]

    assert remove_toc(lines) == ["Bevezető szöveg.", "", "Szöveg."]


def test_remove_toc_keeps_single_dot_leader_line():
    lines = ["Szöveg.", "Lásd a mellékletet ........ 3", "Folytatás."]

    assert remove_toc(lines) == lines


# --- join_paragraphs / remove_topic_list ---

@pytest.mark.parametrize("lines, paragraphs", [
    (["Első sor", "folytatás.", "", "Második."], ["Első sor folytatás.", "Második."]),
    (["adó-", "alap"], ["adó-alap"]),
    (["Az alábbiak:", "− a) első,", "− b) második."], ["Az alábbiak:\n− a) első,\n− b) második."]),
    (["", "Szöveg.", "", ""], ["Szöveg."]),
])
def test_join_paragraphs(lines, paragraphs):
    assert join_paragraphs(lines) == paragraphs


def test_remove_topic_list_drops_intro_and_list():
    paragraphs = ["Bevezető.", "Ebben a füzetben az alábbiakról olvashat:", "Téma 1, Téma 2", "Folytatás."]

    assert remove_topic_list(paragraphs) == ["Bevezető.", "Folytatás."]


# --- clean_page (whole chain) ---

def test_clean_page_end_to_end():
    text = (
        "12\n"
        "A vagyonkezelő alapítványt külön\n"
        "törvény5 szabályozza.\n"
        "\n"
        "5 2019. évi XIII. törvény"
    )

    assert clean_page(text) == "A vagyonkezelő alapítványt külön törvény szabályozza."


# --- join_pages / page_at ---

def page(text: str, number: int, source: str = "a.pdf") -> Document:
    return Document(page_content=text, metadata={"source": source, "title": source, "page": number})


@pytest.mark.parametrize("first, separator", [
    ("Az első oldal vége.", "\n\n"),
    ("Az első oldal mondata", " "),
])
def test_join_pages_separator(first, separator):
    text, offsets, page_numbers = join_pages([page(first, 1), page("második oldal.", 2)])

    assert text == first + separator + "második oldal."
    assert offsets == [0, len(first) + len(separator)]
    assert page_numbers == [1, 2]


@pytest.mark.parametrize("offset, expected", [(0, 1), (17, 1), (18, 2), (30, 2)])
def test_page_at(offset, expected):
    assert page_at(offset, [0, 18], [1, 2]) == expected


# --- split_documents ---

def sentences(word: str, page_number: int) -> str:
    return " ".join(f"Ez a {word} füzet {page_number}. oldalának {i}. mondata." for i in range(6))


@pytest.fixture
def small_chunks(settings):
    return settings.model_copy(update={"chunk_size": 200, "chunk_overlap": 50})


def test_split_documents(small_chunks):
    pages = [
        page(sentences("ALFA", 1), 1, "alfa.pdf"),
        page(sentences("ALFA", 2), 2, "alfa.pdf"),
        page(sentences("BETA", 1), 1, "beta.pdf"),
    ]

    chunks = split_documents(pages, small_chunks)

    assert len(chunks) > 3
    for chunk in chunks:
        assert len(chunk.page_content) <= 200
        assert {"source", "title", "page", "page_end", "start_index"} <= chunk.metadata.keys()
        assert chunk.metadata["page"] <= chunk.metadata["page_end"]
        word = "ALFA" if chunk.metadata["source"] == "alfa.pdf" else "BETA"
        other = "BETA" if word == "ALFA" else "ALFA"
        assert word in chunk.page_content
        assert other not in chunk.page_content
    assert any(c.metadata["page"] < c.metadata["page_end"] for c in chunks if c.metadata["source"] == "alfa.pdf")


# --- chunk_id ---

def chunk(source: str, start_index: int) -> Document:
    return Document(page_content="x", metadata={"source": source, "start_index": start_index})


def test_chunk_id_is_stable():
    assert chunk_id(chunk("a.pdf", 100)) == chunk_id(chunk("a.pdf", 100))


@pytest.mark.parametrize("other", [chunk("a.pdf", 101), chunk("b.pdf", 100)])
def test_chunk_id_differs(other):
    assert chunk_id(chunk("a.pdf", 100)) != chunk_id(other)
