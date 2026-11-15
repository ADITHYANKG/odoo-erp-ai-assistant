# ont_mcp_rag — document search for the AI assistant

Makes the contents of Odoo Documents answerable in chat. *"What is the
harassment policy?"* is answered from the company's own PDF, with a citation,
instead of from the language model's general knowledge — which is the whole
point, because a plausible answer that is **not this company's policy** is
worse than no answer.

## Setting it up

1. Install the module.
2. Set the system parameter `ont_mcp_rag.openrouter_api_key`.
3. **Settings → Technical → AI Document Search** → tick "Searchable by AI"
   on a folder.

That is all. A scheduled action runs every 10 minutes; uploading a document
to a ticked folder makes it answerable within minutes with no further action.
The same screen shows an indexed-passage count per folder, so "is it
working?" is answerable without a database console.

## What was already there (and what that saved)

Investigated before building, on 2026-09-02:

- **Odoo already extracts the text.** `attachment_indexation` fills
  `ir_attachment.index_content` on upload — 44,975 characters for the sample
  25-page HR policy. There is no PDF parsing in this module and no OCR
  dependency. A scanned document has no extractable text, so it is skipped
  rather than retried for ever.
- **pgvector is created by the module's pre-init hook**; the `vector(1536)`
  column and both indexes are created in `ont.document.chunk.init()`, since
  Odoo 17 has no pgvector ORM field.
- **`read_resource` already exists** for fetching a whole document.

**Odoo's own `ai.embedding` could not be reused** despite having exactly the
right shape. It is driven by `ai.agent.source` records, and its
`embedding_model` is a `Selection` restricted to first-party providers, which
does not include OpenRouter. This module copies the proven shape — attachment
foreign key with cascade delete, `sequence`, `content`, `vector(1536)`,
ivfflat cosine index — and reuses only the field type.

## Chunking

**One generic splitter. No per-document-type parsers** — a contract is not an
HR policy is not a set of board minutes, and a parser per type works on the
three documents it was written against and breaks on the fourth.

A heading is detected as *a short line, no terminal punctuation, followed by
prose*, which catches `44. Harassment`, `ARTICLE V - TERMINATION`,
`7.2 Payment Terms` and `Agenda Item 3` without knowing anything about the
document.

Three details that came from running it against the real policy:

- **Merged chunks report their full span.** Short neighbouring sections are
  merged (an 80-character chunk retrieves badly), but the label becomes
  `76. Organizational Change - 79. Notice Period`. Labelling it only by where
  it starts had the assistant citing "Organizational Change" for a
  notice-period question.
- **Running headers are stripped by their rhythm**, not their frequency.
  `MyCompany | Human Resources Policy` appeared 24 times, once per page; the
  word "Approve" appeared 11 times as real table content. Page furniture
  repeats at regular intervals (coefficient of variation 0.43) while real
  text clusters (0.79), and adjacent duplicates are excluded by a minimum
  gap. 1,393 characters of furniture removed from the sample.
- **Symbol-font glyphs are dropped** — 62 private-use codepoints (Wingdings
  bullets) that mean nothing to a reader or a search index.

## Retrieval

**Hybrid: cosine similarity and PostgreSQL full-text, fused by reciprocal
rank.** Neither half is sufficient — similarity misses exact terms (a clause
number, a defined term) and keywords miss paraphrase, which is most of how
people actually ask: *"can I work from home?"* against a section headed
*"Remote and Hybrid Work"*.

**Neighbouring sections come back too**, for the strongest few hits. *"What
happens if I am harassed?"* is genuinely answered by the definition, the
grievance procedure and the non-retaliation clause together; one passage
would read as a complete answer while being an incomplete one.

**Permissions are the folder's.** Search runs as the calling user through the
ORM, so `documents.document` record rules decide what is findable — the same
rules as the web client, with nothing re-implemented here.

### The similarity gate is deliberately low, and this is the important part

Measured over 14 questions the sample document answers and 6 it does not:

| | score |
|---|---|
| lowest genuine question ("can I take a second job?") | **0.379** |
| highest absent question ("what is our crypto policy?") | **0.357** |

The two populations separate by 0.022. **A threshold in that band would fit
the sample and break on the next phrasing.** Cosine similarity is not a
reliable relevance classifier: an absent question that shares the document's
vocabulary ("policy", "company") outscores a genuine one that does not.

So the gate sits at **0.30** and only removes what is unambiguously
unrelated — coffee machines (0.137), laser cutters (0.180), scuba depths
(0.184), bitcoin refunds (0.230) — leaving comfortable headroom under every
genuine question. **Judging the borderline is the model's job**, and a
`SYSTEM_PROMPT` rule requires it to decline when the returned passages do not
address the question. It can read a passage headed "1. Purpose" and tell that
it does not answer a question about crypto trading; no scalar threshold can.

## Measured results

`experiments/13_probe_policy_questions.py`, live against the HR policy:

```
search_documents called : 11/11   <-- the headline number
expected section cited  : 10/10
absent-policy control   : declined honestly, naming the folder searched
```

**The tool-call rate is the headline** because the worst failure this feature
can have is invisible to every other measure: the model never searches,
answers from general knowledge, and produces something that reads perfectly
while not being this company's policy.

**`query` is declared REQUIRED in the tool schema** for the same reason. Left
to inference, the model opened with `search_documents({})`, was told there was
no search text, and retried with variations — three calls where one does, and
on slower questions enough to exhaust the recursion limit and degrade the
answer entirely. Declaring it required took that from three calls to one.

## Deduplication and staleness

Both keyed on the attachment checksum. The same file uploaded three times
(which is exactly what was in the live database) is indexed **once**;
replacing a file changes its checksum and its chunks are rebuilt; deleting a
document removes its chunks by foreign-key cascade.

## Known limitations

- **Tables flatten.** In the sample policy the header table's labels and
  values extracted as two separate blocks, so "who approved this policy?" has
  the labels next to the wrong values. Prose is unaffected; budget packs
  would not be.
- **Scanned documents are skipped**, having no extractable text. OCR is out
  of scope; `AGENT_UPLOAD_PDF_ENGINE` is the escalation path if one appears.
- The vector width is fixed in the column type (as it is in Odoo's own
  `ai.embedding`). `ont_mcp_rag.embed_dims` exists to *validate* that the
  configured model agrees with it; moving to a differently-sized model means
  changing the column and re-indexing.

## Tests

23, on `ont_test`:

```
docker exec odoo19-ent-2 sh -c "/opt/odoo/odoo-bin -c /opt/odoo/odoo.conf -d ont_test --http-port=8072 -u ont_mcp_rag --test-enable --test-tags /ont_mcp_rag --stop-after-init"
```
