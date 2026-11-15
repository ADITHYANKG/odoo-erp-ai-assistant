"""Split an extracted document into chunks worth embedding.

ONE generic splitter, deliberately. Documents vary — a contract is not a
Human Resources policy is not a set of board minutes — and a parser per
document type works on the three documents it was written against and
breaks on the fourth. Nothing here knows what kind of document it is
reading.

The heuristic that carries it: a heading is a SHORT line with no terminal
punctuation, followed by prose. That single rule catches
``44. Harassment``, ``ARTICLE V - TERMINATION``, ``7.2 Payment Terms`` and
``Agenda Item 3`` without being told about any of them.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

#: Bumped whenever a change here would produce DIFFERENT chunks for the same
#: input. Stored on every chunk so the indexer can tell that existing rows
#: were built by older logic and rebuild them. Without it, improving the
#: splitter silently never reaches documents already indexed - their file
#: checksum has not changed, so nothing marks them stale, and a client would
#: keep the chunking bugs they were shipped with for ever.
#:
#: 2 - section numbers must be dotted ("44." / "7.2"), page markers are not
#:     headings, and an unnumbered Title Case line needs prose after it.
#: 3 - a numbered heading's title must start with a capital, so the money
#:     range "250.01 to 1,000 USD" is no longer read as section 250.01.
CHUNKER_VERSION = 3

#: A heading is short. Longer than this and it is a sentence that happens to
#: lack a full stop — a line wrap, a table row, a bullet.
MAX_HEADING_CHARS = 80

#: Lines ending in these are prose, never headings.
_SENTENCE_ENDINGS = ('.', ',', ';', ':', '!', '?')

#: Shapes a heading may take. Checked in order; the first match wins.
_HEADING_PATTERNS = (
    # "44. Harassment" / "7.2 Payment Terms" / "12.3.1 Scope".
    # Three constraints, each earned from a real false positive in the
    # expense policy (2026-09-02):
    #   - the number must be dotted ("44.") or sub-numbered ("7.2"), never
    #     bare, or "125 USD" becomes a section number;
    #   - the title must begin with a CAPITAL, or the money range
    #     "250.01 to 1,000 USD" parses as section 250.01;
    #   - a title is words, not more digits.
    re.compile(r'^(?:\d+\.|\d+(?:\.\d+)+\.?)\s+[A-Z]\S*.*$'),
    # "ARTICLE V - TERMINATION"  /  "SECTION 4"  /  "APPENDIX B"
    re.compile(r'^[A-Z][A-Z0-9 &/,\'()-]{3,}$'),
    # "Payment Terms" / "Agenda Item 3" — Title Case, at least two words,
    # optionally ending in a number ("Schedule 2", "Appendix 4").
    re.compile(r'^(?:[A-Z][a-z0-9\']*\s+){1,7}(?:[A-Z][A-Za-z0-9\']*|\d+)$'),
)

#: Page furniture that VARIES, so the repeat detector cannot see it: every
#: "Page 2", "Page 3" is a different string and each appears once.
_PAGE_MARKER = re.compile(r'^(?:page|p\.?)\s*\d+(?:\s*(?:of|/)\s*\d+)?$', re.I)

#: A heading introduces prose. A line of Title Case followed by more SHORT
#: lines is a table row, not a section — which is how the expense policy
#: ended up with chunks headed "Additional Evidence" and "125 USD".
MIN_PROSE_CHARS = 60
PROSE_LOOKAHEAD_LINES = 4


#: Symbol-font glyphs (Wingdings bullets and friends) extract as private-use
#: codepoints - 62 of them in the sample policy - and mean nothing to a
#: reader or to a search index. Built with chr() so no unprintable
#: character ever sits in this source file.
_PRIVATE_USE = re.compile('[' + chr(0xE000) + '-' + chr(0xF8FF) + ']')

#: A running header or footer repeats once per page, so it appears many
#: times AT REGULAR INTERVALS. Frequency alone is not enough to identify
#: it: in the sample policy the word "Approve" occurs 11 times
#: legitimately, inside tables, and dropping it would delete real content.
MIN_BOILERPLATE_REPEATS = 5
#: How irregular the gaps may be and still count as "once per page",
#: measured as the coefficient of variation. Calibrated against the sample
#: policy on 2026-09-02, where the separation is wide and unambiguous:
#: the two genuine running headers scored 0.43, while the most regular
#: piece of REAL repeated content ("Approve", a table cell) scored 0.79.
#: Real text repeats in clusters; page furniture repeats on a rhythm.
MAX_BOILERPLATE_GAP_VARIATION = 0.6
#: Page furniture is separated by a page's worth of lines. A tight run of
#: identical adjacent lines - repeated table cells, a list of the same word -
#: has gaps of 1 and therefore PERFECT regularity, so the variation test
#: alone would call it boilerplate and delete real content.
MIN_BOILERPLATE_GAP = 15


@dataclass(frozen=True)
class Chunk:
    """One embeddable piece of a document."""

    sequence: int
    heading: str
    body: str

    @property
    def content(self) -> str:
        """What actually gets embedded.

        The heading is prepended rather than kept only as metadata: chunks
        here average a few hundred characters, so the heading carries a
        large share of the meaning, and it is what makes the obvious
        question ("what is the harassment policy?") match the right
        section.
        """
        return f'{self.heading}\n\n{self.body}' if self.heading else self.body


def _boilerplate_lines(lines: list[str]) -> set[str]:
    """Running headers and footers, identified by their regular spacing.

    Extracted PDF text carries the page furniture — ``MyCompany | Human
    Resources Policy`` appeared 24 times in the sample, once per page. Left
    in, it is embedded into every chunk, so every chunk looks slightly like
    every other one and similarity search gets noisier the longer the
    document is.
    """
    positions: defaultdict[str, list[int]] = defaultdict(list)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped and len(stripped) < MAX_HEADING_CHARS:
            positions[stripped].append(index)

    boilerplate = set()
    for line, at in positions.items():
        if len(at) < MIN_BOILERPLATE_REPEATS:
            continue
        gaps = [second - first for first, second in zip(at, at[1:], strict=False)]
        mean = statistics.fmean(gaps)
        if mean < MIN_BOILERPLATE_GAP:
            continue
        # Regular spacing => page furniture. Prose that merely repeats
        # (a table label, a recurring clause) clusters instead.
        if statistics.pstdev(gaps) / mean <= MAX_BOILERPLATE_GAP_VARIATION:
            boilerplate.add(line)
    return boilerplate


def clean_text(text: str) -> str:
    """Strip extraction artefacts before anything is split or embedded."""
    text = (text or '').replace('\r\n', '\n')
    # Private-use codepoints are symbol-font glyphs (Wingdings bullets and
    # friends) that extract as meaningless characters — 62 of them in the
    # sample policy. They carry no meaning for search.
    text = _PRIVATE_USE.sub(' ', text)
    lines = text.split('\n')
    drop = _boilerplate_lines(lines)
    kept = [line for line in lines if line.strip() not in drop]
    # Collapse the runs of blank lines that removal leaves behind.
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(kept)).strip()


def looks_like_heading(line: str, following: Sequence[str] = ()) -> bool:
    """Whether one line reads as a section heading.

    :param following: the lines after it. A heading introduces PROSE, so a
        candidate followed only by more short lines is a table row. Without
        this check a table-heavy document produces chunks labelled with cell
        values, and a citation quotes the cell instead of the section.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > MAX_HEADING_CHARS:
        return False
    if stripped.endswith(_SENTENCE_ENDINGS):
        return False
    if _PAGE_MARKER.match(stripped):
        return False
    # A line that is mostly digits is a page number or a table row.
    letters = sum(character.isalpha() for character in stripped)
    if letters < 3:
        return False
    if not any(pattern.match(stripped) for pattern in _HEADING_PATTERNS):
        return False
    if not following:
        return True
    # Numbered sections are trustworthy on their own; an unnumbered Title
    # Case line is only a heading if prose actually follows it.
    if _HEADING_PATTERNS[0].match(stripped):
        return True
    upcoming = [text.strip() for text in following[:PROSE_LOOKAHEAD_LINES] if text.strip()]
    return any(len(text) >= MIN_PROSE_CHARS for text in upcoming)


def _split_oversized(text: str, limit: int) -> list[str]:
    """Break a too-long block on paragraph, then sentence, then hard.

    A real section can run five pages, so a heading boundary alone is not
    enough — without this one chunk swallows a chapter and retrieves for
    everything.
    """
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    current = ''
    for paragraph in re.split(r'\n\s*\n', text):
        candidate = f'{current}\n\n{paragraph}' if current else paragraph
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if len(paragraph) <= limit:
            current = paragraph
            continue
        # A single paragraph over the limit: sentences, then characters.
        current = ''
        for sentence in re.split(r'(?<=[.!?])\s+', paragraph):
            candidate = f'{current} {sentence}'.strip()
            if len(candidate) <= limit:
                current = candidate
                continue
            if current:
                pieces.append(current)
            while len(sentence) > limit:
                pieces.append(sentence[:limit])
                sentence = sentence[limit:]
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def _with_overlap(pieces: list[str], overlap: int) -> list[str]:
    """Carry the tail of each piece into the next.

    Only used on the no-headings path. With headings, a boundary is a real
    section break and overlapping across it just duplicates text; without
    them the boundary is arbitrary, and overlap is what stops a clause
    being cut in half so that neither half means what the whole did.
    """
    if overlap <= 0 or len(pieces) < 2:
        return pieces
    joined = [pieces[0]]
    for previous, piece in zip(pieces, pieces[1:], strict=False):
        tail = previous[-overlap:]
        joined.append(f'{tail}\n\n{piece}' if tail.strip() else piece)
    return joined


def chunk_text(
    text: str,
    max_chars: int = 1800,
    min_chars: int = 250,
    overlap_chars: int = 200,
) -> list[Chunk]:
    """Split extracted document text into chunks.

    :param max_chars: hard ceiling on one chunk.
    :param min_chars: chunks below this are merged forward — an 80-character
        section retrieves badly and means nothing on its own.
    :param overlap_chars: applied ONLY when no headings were found.
    """
    text = (text or '').replace('\r\n', '\n').strip()
    if not text:
        return []

    # --- pass 1: cut at headings ------------------------------------------
    sections: list[tuple[str, list[str]]] = []
    heading = ''
    body: list[str] = []
    lines = text.split('\n')
    for index, line in enumerate(lines):
        if looks_like_heading(line, lines[index + 1 :]):
            if heading or any(part.strip() for part in body):
                sections.append((heading, body))
            heading, body = line.strip(), []
        else:
            body.append(line)
    if heading or any(part.strip() for part in body):
        sections.append((heading, body))

    found_headings = any(head for head, _ in sections)

    # --- pass 2: size --------------------------------------------------
    sized: list[tuple[str, str]] = []
    for head, lines in sections:
        block = '\n'.join(lines).strip()
        if not block:
            # A heading with no body: keep it, the merge below will attach
            # it to what follows rather than embedding a bare title.
            sized.append((head, ''))
            continue
        for piece in _split_oversized(block, max_chars):
            sized.append((head, piece))
            head = ''  # continuation pieces inherit the section, not the title

    if not found_headings:
        pieces = _with_overlap([body for _, body in sized if body], overlap_chars)
        sized = [('', piece) for piece in pieces]

    # --- pass 3: merge the too-small forward -------------------------------
    # Short neighbouring sections are merged because an 80-character chunk
    # retrieves badly and means nothing alone. But the merged chunk must
    # then report EVERY section it covers: sections 76-79 of the sample
    # policy merge into one chunk, and labelling it only "76.
    # Organizational Change" would have the bot cite section 76 for a
    # question actually answered by 79. Notice Period.
    merged: list[tuple[list[str], str]] = []
    for head, block in sized:
        combined = f'{head}\n\n{block}'.strip() if head else block
        if merged and len(combined) < min_chars:
            previous_heads, previous_block = merged[-1]
            merged[-1] = (
                previous_heads + ([head] if head else []),
                f'{previous_block}\n\n{combined}'.strip(),
            )
            continue
        merged.append(([head] if head else [], block))

    # A lone trailing fragment would otherwise stand as its own chunk.
    if len(merged) > 1:
        last_heads, last_block = merged[-1]
        if len(''.join(last_heads) + last_block) < min_chars:
            heads, block = merged.pop()
            previous_heads, previous_block = merged[-1]
            tail = f'{heads[0]}\n\n{block}'.strip() if heads else block
            merged[-1] = (previous_heads + heads, f'{previous_block}\n\n{tail}'.strip())

    return [
        Chunk(sequence=index, heading=_span(heads), body=block)
        for index, (heads, block) in enumerate(merged)
        if block.strip() or heads
    ]


def _span(headings: list[str]) -> str:
    """Label a chunk by the sections it covers.

    One heading is itself; several become ``"76. Organizational Change - 79.
    Notice Period"`` so a citation names the range that was actually read
    rather than only where it started.
    """
    kept = [head for head in headings if head]
    if not kept:
        return ''
    if len(kept) == 1:
        return kept[0]
    return f'{kept[0]} - {kept[-1]}'
