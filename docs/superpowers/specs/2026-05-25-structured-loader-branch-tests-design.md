# Structured Loader Branch Tests Design

## Goal

Rewrite the existing tests for `structured_artifacts`, `structured_docx_loader`,
and `structured_html_loader` so that they deliberately exercise conditional
branches while still verifying the primary public behaviors of each module.

## Scope

Only these tracked test files will be replaced:

- `tests/test_structured_artifacts.py`
- `tests/test_structured_docx_loader.py`
- `tests/test_structured_html_loader.py`

Production code is out of scope. Existing unrelated or untracked working tree
changes, including `tests/test_requested_line_coverage.py`, are not modified.

## Test Organization

Each test file remains owned by the corresponding module and is organized
around three concerns where applicable:

1. Public parsing or persistence behavior that proves the module still works
   through its normal entry points.
2. Conditional helper behavior, using small inputs that force true and false
   branches without unnecessary integration setup.
3. Boundary cases such as missing content, invalid image sources, duplicate
   keys, or absent artifacts.

## Coverage Targets

### `structured_artifacts`

Cover prefix validation for empty, accepted, and rejected inputs; artifact and
image metadata extraction with absent and duplicate values; empty persistence
paths; parsed markdown selection from direct text or blocks; section grouping
from custom IDs, headers, and fallback titles; table reference application;
and image key inclusion or omission in summaries.

### `structured_docx_loader`

Cover empty document parsing; heading, normal, image-only, and empty paragraph
handling; table dispatch; heading parsing failures and missing nesting levels;
image relation de-duplication; missing and non-image parts; markdown and object
key collection; block flushing around headings and tables; and image part type
detection.

### `structured_html_loader`

Cover event dispatch for text, skipped nodes, headings, tables, images, and
nested nodes; empty text block flushing; header nesting; remote, data, and
local image handling; invalid or absent image input; safe local path
validation; uploaded object key recording; and markdown text/image joining.

## Verification

Run the rewritten three test modules directly with `unittest`. If a coverage
runner is available, report branch coverage for the three target production
modules; if it is unavailable in the environment, report that limitation
explicitly after confirming all targeted tests pass.
