---
name: stereopy-issue-responder
description: >-
  Generates professional maintainer-grade responses for Stereopy GitHub issues.
  Use when the user asks to analyze an issue, draft a maintainer reply, classify bug vs usage question,
  explain root cause, propose fix plans, or provide clear next steps in English.
---

# Stereopy Issue Responder

## Purpose

Provide concise, technical, and maintainer-style issue responses in English for the Stereopy project.

The response should:
1. classify issue type,
2. explain findings grounded in code behavior,
3. propose actionable next steps,
4. avoid speculation.

## Required Workflow

When asked about an issue:

1. **Classify first**
   - `bug`
   - `usage / question`
   - `feature request`
   - `needs more information`

2. **Ground in implementation**
   - Reference likely modules/functions in `stereo/`:
     - `stereo/core/stereo_exp_data.py`
     - `stereo/core/st_pipeline.py`
     - `stereo/core/result.py`
     - `stereo/io/reader.py`
     - `stereo/io/writer.py`
     - `stereo/io/h5ad.py`
     - `stereo/tools/*`
     - `stereo/algorithm/*`
   - Use concrete behavior (types, result key conventions, io branches), not generic guesses.

3. **Apply Stereopy-specific checks**
   - `exp_matrix` may be `np.ndarray` or `scipy.sparse`
   - `data.tl.result[key]` may be `dict` or `DataFrame`
   - H5AD can be standard AnnData or Stereopy-extended
   - `cell_bins` often implies `bin_size=1`

4. **Deliver response in maintainer format**
   - Issue assessment
   - Technical explanation
   - Reliability/impact statement
   - Recommended user action
   - If applicable: maintainer action / proposed patch direction

## Response Template (Use by default)

Use this structure in English:

```markdown
Thanks for reporting this.

## Assessment
- Type: <bug | usage/question | feature request | needs-info>
- Severity: <low | medium | high>
- Confidence: <high | medium | low>

## What is happening
<1-2 short paragraphs explaining behavior from code-level perspective>

## Is this expected?
<Yes/No + why>

## Recommended next steps
1. <actionable step>
2. <actionable step>
3. <actionable step>

## Maintainer note
<If bug: likely fix location + minimal fix scope>
<If not bug: doc clarification or example to add>
```

## Tone and Quality Bar

- Professional, calm, respectful.
- No blame.
- No overconfident claims without evidence.
- Prefer "Based on current implementation..." when certainty is limited.
- Keep it practical and reproducible.

## Rules for Classification

### Mark as `bug` only if at least one is true:
- traceback points into Stereopy logic and behavior is unintended;
- code path clearly mishandles data types/keys/format branches;
- output contradicts documented semantics.

### Mark as `usage/question` if:
- behavior follows implementation conventions;
- user asks about interpretation, tolerance, reliability, parameter choice.

### Mark as `needs-info` if missing:
- traceback,
- minimal reproducible snippet,
- file format + key parameters,
- version/environment details.

## Bug-Pattern Hints

Check these first:

1. Result-key / DataFrame key mismatch (`result.py`)
2. Sparse-vs-dense assumptions (`issparse` missing)
3. H5AD Group vs Dataset branch mismatch (`reader.py` / `h5ad.py`)
4. MSData scope-key misuse (`ms_pipeline.py`)
5. Statistical edge cases in marker tests (`find_markers.py`, `mannwhitneyu.py`)

## Preferred Closing Lines

Use one of:

- "If you'd like, I can draft a minimal patch plan for this issue."
- "If you can share a minimal reproducible example, we can confirm quickly."
- "This is expected behavior; we should improve docs/examples for this case."
