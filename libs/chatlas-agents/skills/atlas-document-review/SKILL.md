---
name: atlas-document-review
description: "Review ATLAS scientific document drafts. Use when reading, checking, or revising ATLAS LaTeX notes, papers, internal documents, analysis notes, review drafts, or technical writeups that need physics-content review, ATLAS style checks, citation verification, and inline comment generation."
---

# ATLAS Document Review

## Workflow

1. Read the main LaTeX document and identify the structure, bibliography, included files, and whether ATLAS TODO notes are enabled.
2. Gather relevant context from ATLAS public sources, internal documentation, and the paper's cited references when the review needs physics or methodology support.
3. Review the draft for:
   - physics and analysis consistency
   - logical flow and missing context
   - citation completeness and relevance
   - ATLAS style, terminology, and LaTeX conventions
   - wording, grammar, spelling, and readability
4. Produce comments grouped by category:
   - `CONTENT|CRITICAL`
   - `CONTENT|SUGGESTION`
   - `CONTENT|QUESTION`
   - `CONTENT|MINOR`
   - `STYLE|FORMATTING`
   - `STYLE|SPELLING`
   - `STYLE|GRAMMAR`
   - `STYLE|PHRASING`
5. If inline comments are requested, insert them without changing document structure, variable names, or package dependencies.

## Review guidance

- Use the ATLAS Style Guide for style-related comments.
- Prefer public ATLAS papers and standard references for detector, calibration, and analysis-method claims when they are clearer than internal notes.
- Check the paper's abstract and introduction references first when looking for supporting literature.
- Keep comments constructive, specific, and actionable.

## Commenting rules

- If `atlastodo` is enabled, add the reviewer note in the preamble and use `\AIinote` comments.
- Otherwise, insert LaTeX comments with the `%(AI-COMMENT:COMMENTCATEGORY)` template.
- For bibliography feedback, use `%(AI-COMMENT:BIBLIOGRAPHY)`.
- Keep LaTeX syntax valid in inserted comments; use math mode for special symbols.

## Optional reference files

- Read `references/review_checklist.md` for a compact review checklist.
- Read `references/atlas_style_guide.md` for style and wording guidance.
- Read `references/physics_comments.md` for physics-review framing.
- Read `references/astyle_rules.txt` for style-check patterns and common issues.

