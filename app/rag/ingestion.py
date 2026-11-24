import logging
import re
from bisect import bisect_right
from itertools import groupby
from pathlib import Path

import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# regexes
PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,3}\s*$")
DOT_LEADER = re.compile(r"\.{4,}\s*\d*\s*$")
NUMBERED_ENTRY = re.compile(r"^\s*\d+(\.\d+)*\.?\s.*\s\d{1,3}\s*$")
TOC_HEADING = re.compile(r"^\s*tartalom(jegyzék)?\s*:?\s*$", re.IGNORECASE)
MIN_TOC_LINES = 2

FOOTNOTE_LINE = re.compile(r"^\s*\d{1,3}(\s+\S|[A-ZÁÉÍÓÖŐÚÜŰ])")
LIST_ITEM = re.compile(r"^\s*([•\-–−]|[a-z]\))\s")
SENTENCE_END = re.compile(r"[.!?:;]\d*\s*$")
SEPARATORS = [r"(?<=[.!?;:])\s+", r"\n", " ", ""]
FOOTNOTE_NUMBER = re.compile(r"^\s*(\d{1,3})")
FOOTNOTE_MARKER = re.compile(r"(?:(?<=[^\W\d_])|(?<=[^\W\d_][.,;:)”]))(\d{1,3})(?=[\s.,;:)”]|$)")
TOPIC_LIST_INTRO = re.compile(r"az alábbiakról olvashat", re.IGNORECASE)
FILENAME = re.compile(r"^(\d+)\s+(.+?)\s+(\d{8})$")


# --- 1. Load ---------------------------------------------------------------

def booklet_info(path: Path) -> dict[str, str]:
    match = FILENAME.match(path.stem)
    if not match:
        return {"title": path.stem}
    number, title, date = match.groups()
    return {
        "booklet": number,
        "title": title,
        "published": f"{date[:4]}-{date[4:6]}-{date[6:]}",
    }


def load_pdf(path: Path) -> list[Document]:
    try:
        reader = pypdf.PdfReader(path)
    except Exception:
        logger.warning("Skipping unreadable PDF %s", path.name, exc_info=True)
        return []

    info = booklet_info(path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": path.name, **info, "page": index + 1},
        )
        for index, page in enumerate(reader.pages)
    ]


def load_documents(data_dir: Path) -> list[Document]:
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.warning("No PDF files found in %s", data_dir)

    pages: list[Document] = []
    for path in pdf_paths:
        pages.extend(load_pdf(path))
    logger.info("Loaded %d pages from %d PDFs", len(pages), len(pdf_paths))
    return pages


# --- 2. Clean --------------------------------------------------------------

def remove_toc(lines: list[str]) -> list[str]:
    if sum(bool(DOT_LEADER.search(line)) for line in lines) < MIN_TOC_LINES:
        return lines

    toc_like = [
        i for i, line in enumerate(lines)
        if DOT_LEADER.search(line) or NUMBERED_ENTRY.match(line) or TOC_HEADING.match(line)
    ]
    first, last = toc_like[0], toc_like[-1]
    return lines[:first] + lines[last + 1:]


def remove_footnotes(lines: list[str]) -> tuple[list[str], set[int]]:
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    blanks = [i for i, line in enumerate(lines) if not line.strip()]
    block_start = blanks[-1] + 1 if blanks else 0
    if block_start < len(lines) and FOOTNOTE_LINE.match(lines[block_start]):
        numbers = {
            int(FOOTNOTE_NUMBER.match(line).group(1))
            for line in lines[block_start:]
            if FOOTNOTE_LINE.match(line)
        }
        return lines[:block_start], numbers
    return lines, set()


def remove_footnote_markers(lines: list[str], numbers: set[int]) -> list[str]:
    if not numbers:
        return lines

    def replace(match: re.Match) -> str:
        return "" if int(match.group(1)) in numbers else match.group(0)

    return [FOOTNOTE_MARKER.sub(replace, line) for line in lines]


def remove_topic_list(paragraphs: list[str]) -> list[str]:
    for i, paragraph in enumerate(paragraphs):
        if TOPIC_LIST_INTRO.search(paragraph):
            return paragraphs[:i] + paragraphs[i + 2:]
    return paragraphs


def join_paragraphs(lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            if current:
                paragraphs.append("\n".join(current))
                current = []
        elif not current or LIST_ITEM.match(line):
            current.append(line)
        elif current[-1].endswith("-"):
            current[-1] += line
        else:
            current[-1] += " " + line

    if current:
        paragraphs.append("\n".join(current))
    return paragraphs


def clean_page(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]

    for i, line in enumerate(lines):
        if line.strip():
            if PAGE_NUMBER_LINE.match(line):
                lines[i] = ""
            break

    lines = remove_toc(lines)
    lines, footnote_numbers = remove_footnotes(lines)
    lines = remove_footnote_markers(lines, footnote_numbers)
    paragraphs = remove_topic_list(join_paragraphs(lines))
    return "\n\n".join(paragraphs)


def clean_documents(pages: list[Document]) -> list[Document]:
    cleaned = []
    for page in pages:
        text = clean_page(page.page_content)
        if text:
            cleaned.append(Document(page_content=text, metadata=page.metadata))
    logger.info("Kept %d of %d pages after cleaning", len(cleaned), len(pages))
    return cleaned


# --- 3. Chunk --------------------------------------------------------------

def join_pages(pages: list[Document]) -> tuple[str, list[int], list[int]]:
    text = ""
    offsets: list[int] = []
    page_numbers: list[int] = []
    for page in pages:
        if text:
            text += "\n\n" if SENTENCE_END.search(text) else " "
        offsets.append(len(text))
        page_numbers.append(page.metadata["page"])
        text += page.page_content
    return text, offsets, page_numbers


def page_at(offset: int, offsets: list[int], page_numbers: list[int]) -> int:
    return page_numbers[bisect_right(offsets, offset) - 1]


def split_documents(pages: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=SEPARATORS,
        is_separator_regex=True,
        keep_separator="end",
        add_start_index=True,
    )

    chunks: list[Document] = []
    for _, booklet_pages in groupby(pages, key=lambda page: page.metadata["source"]):
        booklet_pages = list(booklet_pages)
        text, offsets, page_numbers = join_pages(booklet_pages)
        booklet_metadata = {k: v for k, v in booklet_pages[0].metadata.items() if k != "page"}
        for chunk in splitter.create_documents([text], metadatas=[booklet_metadata]):
            start = chunk.metadata["start_index"]
            end = start + len(chunk.page_content) - 1
            chunk.metadata["page"] = page_at(start, offsets, page_numbers)
            chunk.metadata["page_end"] = page_at(end, offsets, page_numbers)
            chunks.append(chunk)

    logger.info("Split %d pages into %d chunks", len(pages), len(chunks))
    return chunks

def ingest_documents(data_dir: Path, settings: Settings) -> list[Document]:
    pages = clean_documents(load_documents(data_dir))
    return split_documents(pages, settings)


# --- Preview: python -m rag.ingestion ---------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()

    pages = clean_documents(load_documents(settings.data_dir))
    chunks = split_documents(pages, settings)

    for chunk in chunks[:3]:
        print("-" * 80)
        print(chunk.metadata)
        print(chunk.page_content)
