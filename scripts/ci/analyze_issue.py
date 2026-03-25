"""
Automated GitHub Issue analysis script.
Triggered by GitHub Actions when an issue is labeled 'auto-fix'.
Uses ZhipuAI GLM-5 for code analysis and fix generation.

Optimizations:
- AST-based function extraction to minimize token usage
- Enhanced keyword extraction from plain text (not just backtick-wrapped)
- Two-phase analysis: locate files first, then analyze only relevant functions
- Project overview cache to avoid re-sending full files
- Empty result detection and retry
"""

import os
import re
import ast
import json
import subprocess
import time
import textwrap

from zhipuai import ZhipuAI


# ─── Project overview (static, avoids sending large files every time) ────────

PROJECT_OVERVIEW = textwrap.dedent("""\
Stereopy is a Python library for spatial transcriptomics data analysis.

## Core Architecture
- `StereoExpData`: Main data container, wraps AnnData. Holds exp_matrix (numpy/scipy.sparse), cells, genes.
- `AnnBasedStereoExpData`: AnnData-backed variant of StereoExpData.
- `MSData`: Multi-sample data container, holds multiple StereoExpData objects.
- `StPipeline` / `MSDataPipeLine`: Tool execution pipelines. Results stored in `data.tl.result[key]`.
- `_BaseResult` / `AnnBasedResult`: Result storage, mapping tool output keys to data.

## Key Modules
- `stereo/core/`: Data models (stereo_exp_data, ms_data, ms_pipeline, st_pipeline, result, cell, gene)
- `stereo/io/`: File I/O (reader.py, writer.py, h5ad.py) — formats: h5ad, h5ms, gef, gem, loom
- `stereo/tools/`: Analysis tools (find_markers, clustering, dim_reduce, spatial_alignment, etc.)
- `stereo/algorithm/`: Low-level algorithms (mannwhitneyu, sctransform, single_r, cell_cut)
- `stereo/plots/`: Visualization (interact_plot, scatter, marker_genes plots)

## Common Patterns
- exp_matrix can be np.ndarray or scipy.sparse — always check type before operations
- Results in data.tl.result[key] may be dict or DataFrame
- Gene/Cell objects have .to_df() returning pandas DataFrame
- bin_size=1 for cell_bins, otherwise actual bin size
- H5AD has two formats: standard AnnData and Stereopy-extended (with extra groups)
""")


# ─── Keyword extraction ─────────────────────────────────────────────────────

def extract_traceback_files(issue_body: str) -> list:
    """Extract file paths and line numbers from Python tracebacks in issue body."""
    pattern = r'File\s+"([^"]+)",\s+line\s+(\d+)'
    matches = re.findall(pattern, issue_body or "")
    results = []
    for filepath, lineno in matches:
        if "stereo/" in filepath:
            rel_path = "stereo/" + filepath.split("stereo/", 1)[1]
            results.append({"file": rel_path, "line": int(lineno)})
    return results


def extract_keywords(issue_title: str, issue_body: str) -> list:
    """Extract keywords from issue text — backtick-wrapped AND plain text identifiers."""
    text = f"{issue_title} {issue_body or ''}"

    backtick_ids = re.findall(r'`([a-zA-Z_]\w+(?:\.\w+)*)`', text)

    errors = re.findall(
        r'(\w+Error|\w+Exception|TypeError|KeyError|IndexError|AttributeError)',
        text,
    )

    snake_case = re.findall(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b', text)
    snake_case = [w for w in snake_case if len(w) >= 5 and w not in (
        "https", "github", "issue_number", "auto_fix",
    )]

    camel_case = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', text)

    all_kw = list(set(backtick_ids + errors + snake_case + camel_case))
    all_kw.sort(key=lambda x: -len(x))
    return all_kw[:15]


# ─── Code search and reading ────────────────────────────────────────────────

def search_codebase(keywords: list) -> dict:
    """Use ripgrep to find files containing the given keywords."""
    relevant_files = {}
    for keyword in keywords[:10]:
        try:
            result = subprocess.run(
                ["rg", "-l", "--type=py", keyword, "stereo/"],
                capture_output=True, text=True, timeout=10,
            )
            for filepath in result.stdout.strip().split("\n"):
                filepath = filepath.strip()
                if filepath and filepath not in relevant_files:
                    relevant_files[filepath] = ""
        except (subprocess.TimeoutExpired, Exception):
            continue
    return relevant_files


def extract_functions_by_name(filepath: str, func_names: list) -> str:
    """Use AST to extract only relevant functions/classes from a file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except FileNotFoundError:
        return ""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return read_file_truncated(filepath, max_lines=100)

    lines = source.splitlines()
    extracted = []
    matched_ranges = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        node_name = node.name.lower()
        for kw in func_names:
            if kw.lower() in node_name or node_name in kw.lower():
                start = node.lineno - 1
                end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start + 30
                range_key = (start, end)
                if range_key not in matched_ranges:
                    matched_ranges.add(range_key)
                    chunk = "\n".join(lines[start:end])
                    extracted.append(f"[Lines {start + 1}-{end}]\n{chunk}")
                break

    if extracted:
        return "\n\n".join(extracted)
    return ""


def read_file_truncated(filepath: str, max_lines: int = 150) -> str:
    """Read a file, truncated to max_lines."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if len(lines) > max_lines:
            return "".join(lines[:max_lines]) + f"\n... [truncated, total {len(lines)} lines]"
        return "".join(lines)
    except FileNotFoundError:
        return ""


def read_file_with_context(filepath: str, target_line: int = None, context: int = 30) -> str:
    """Read file content centered around a target line."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        if target_line and len(lines) > 2 * context:
            start = max(0, target_line - context)
            end = min(len(lines), target_line + context)
            content = "".join(lines[start:end])
            return f"[Lines {start + 1}-{end}]\n{content}"

        if len(lines) > 150:
            return "".join(lines[:150]) + f"\n... [truncated, total {len(lines)} lines]"

        return "".join(lines)
    except FileNotFoundError:
        return f"[File not found: {filepath}]"


def gather_relevant_code(keywords: list, traceback_files: list, max_files: int = 6) -> dict:
    """
    Smart code gathering: use AST extraction when possible, fall back to truncated reads.
    Returns {filepath: code_string} with minimal token usage.
    """
    file_contents = {}

    for tb in traceback_files:
        filepath = tb["file"]
        line = tb.get("line")
        content = read_file_with_context(filepath, line, context=25)
        if content and "[File not found" not in content:
            file_contents[filepath] = content
            if len(file_contents) >= max_files:
                return file_contents

    search_results = search_codebase(keywords)
    for filepath in search_results:
        if filepath in file_contents:
            continue
        ast_content = extract_functions_by_name(filepath, keywords)
        if ast_content:
            file_contents[filepath] = ast_content
        if len(file_contents) >= max_files:
            return file_contents

    if not file_contents:
        for default_path in [
            "stereo/core/stereo_exp_data.py",
            "stereo/core/st_pipeline.py",
            "stereo/io/reader.py",
        ]:
            ast_content = extract_functions_by_name(default_path, keywords)
            if ast_content:
                file_contents[default_path] = ast_content
            elif not file_contents:
                content = read_file_truncated(default_path, max_lines=100)
                if content:
                    file_contents[default_path] = content

    return file_contents


# ─── Prompt building ────────────────────────────────────────────────────────

def build_analysis_prompt(issue_number, issue_title, issue_body, file_contents: dict) -> str:
    files_section = ""
    for path, content in file_contents.items():
        files_section += f"\n### `{path}`\n```python\n{content}\n```\n"

    return f"""请分析以下 GitHub Issue 并生成修复方案。

## 项目概览
{PROJECT_OVERVIEW}

## Issue #{issue_number}: {issue_title}

{issue_body or '(no body)'}

## 相关代码（仅展示相关函数/类）
{files_section}

请严格按以下 Markdown 格式输出，不要遗漏任何章节：

## 根因分析
（分析问题的根本原因）

## 影响范围
- 严重程度：P0/P1/P2
- 影响的功能模块：...

## 修复方案
### 文件 1：`path/to/file.py`
- 行 N-M：...
- 改动说明：...
```python
# 修改前
...
# 修改后
...
```

## 测试建议
```python
# 最小测试用例
...
```

## 注意事项
（其他需要注意的点）"""


def generate_agent_task(issue_number, issue_title, analysis: str, file_list: list) -> str:
    """Generate the task description file for Cursor Background Agent."""
    return f"""# Agent Task: Fix Issue #{issue_number}

## Issue
**#{issue_number}: {issue_title}**

## 任务要求
1. 阅读下方的分析报告，理解问题根因
2. 按修复方案修改相关文件
3. 确保修改后代码通过 lint 检查（flake8 --max-line-length=120）
4. 如有现有测试，确保 pytest 通过
5. 为修复编写最小测试用例
6. 提交变更（commit message 格式：`fix #{issue_number}: 简要描述`）
7. 不要修改不相关的文件，保持最小改动原则

## 相关文件
{chr(10).join(f'- `{f}`' for f in file_list)}

## 分析报告
{analysis}

## 约束
- 仅修改 `stereo/` 目录下的文件
- 不要修改版本号或 CI 配置
- 保持现有代码风格（4 空格缩进）
- 如果不确定修复方案，在代码中添加 TODO 注释
"""


# ─── API call ────────────────────────────────────────────────────────────────

def call_glm5_with_retry(client, messages, max_retries=3):
    """Call GLM-5 API with exponential backoff retry and empty-result detection."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="glm-5",
                messages=messages,
                max_tokens=16384,
                temperature=0.1,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                print(f"  Warning: API returned empty content (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return "(分析结果为空，请检查 API 配置或手动分析此 issue)"
            print(f"  API response received: {len(content)} chars")
            return content
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 5
                print(f"  API call failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    issue_number = os.environ["ISSUE_NUMBER"]
    issue_title = os.environ["ISSUE_TITLE"]
    issue_body = os.environ.get("ISSUE_BODY", "")

    print(f"=== Analyzing issue #{issue_number}: {issue_title} ===")

    # Step 1: Extract keywords and traceback info
    traceback_files = extract_traceback_files(issue_body)
    keywords = extract_keywords(issue_title, issue_body)
    print(f"  Traceback refs: {len(traceback_files)}")
    print(f"  Keywords: {keywords}")

    # Step 2: Smart code gathering (AST-based, minimal tokens)
    file_contents = gather_relevant_code(keywords, traceback_files)
    total_lines = sum(c.count("\n") for c in file_contents.values())
    print(f"  Loaded {len(file_contents)} files, ~{total_lines} lines total (token-optimized)")
    for fp in file_contents:
        print(f"    - {fp}")

    # Step 3: Call GLM-5
    client = ZhipuAI(api_key=os.environ["ZHIPUAI_API_KEY"])
    prompt = build_analysis_prompt(issue_number, issue_title, issue_body, file_contents)
    prompt_len = len(prompt)
    print(f"  Prompt length: {prompt_len} chars (~{prompt_len // 2} tokens est.)")

    analysis = call_glm5_with_retry(client, [
        {
            "role": "system",
            "content": (
                "你是 Stereopy（空间转录组 Python 库）的高级维护者。"
                "你擅长阅读 Python 源码，定位 bug 根因，并给出精准的修复方案。"
                "请严格按照用户指定的 Markdown 格式输出，不要遗漏任何章节。"
                "代码修改建议必须包含修改前和修改后的对比。"
                "请用中文回答，代码部分用英文。"
            ),
        },
        {"role": "user", "content": prompt},
    ])

    # Step 4: Write outputs
    report = f"## 🤖 自动分析报告 — Issue #{issue_number}\n\n{analysis}"
    with open("analysis_result.md", "w", encoding="utf-8") as f:
        f.write(report)

    agent_task = generate_agent_task(
        issue_number, issue_title, analysis, list(file_contents.keys())
    )
    os.makedirs(".cursor", exist_ok=True)
    with open(".cursor/agent-task.md", "w", encoding="utf-8") as f:
        f.write(agent_task)

    context = {
        "issue_number": issue_number,
        "issue_title": issue_title,
        "traceback_files": traceback_files,
        "keywords": keywords,
        "relevant_files": list(file_contents.keys()),
        "prompt_chars": prompt_len,
        "analysis_chars": len(analysis),
    }
    with open("analysis_context.json", "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2, ensure_ascii=False)

    print(f"\n=== Analysis complete ===")
    print(f"  Report:      analysis_result.md ({len(report)} chars)")
    print(f"  Agent task:  .cursor/agent-task.md")
    print(f"  Context:     analysis_context.json")


if __name__ == "__main__":
    main()
