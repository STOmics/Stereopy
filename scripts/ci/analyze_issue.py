"""
Automated GitHub Issue analysis script.
Triggered by GitHub Actions when an issue is labeled 'auto-fix'.
Uses ZhipuAI GLM-5 for code analysis and fix generation.
"""

import os
import re
import json
import subprocess
import time

from zhipuai import ZhipuAI


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
    """Extract function/class names and error types from issue text."""
    text = f"{issue_title} {issue_body or ''}"
    identifiers = re.findall(r'`([a-zA-Z_]\w+(?:\.\w+)*)`', text)
    errors = re.findall(
        r'(\w+Error|\w+Exception|TypeError|KeyError|IndexError|AttributeError)',
        text
    )
    return list(set(identifiers + errors))


def search_codebase(keywords: list) -> dict:
    """Use ripgrep to find files containing the given keywords."""
    relevant_files = {}
    for keyword in keywords[:10]:
        try:
            result = subprocess.run(
                ["rg", "-l", "--type=py", keyword, "stereo/"],
                capture_output=True, text=True, timeout=10
            )
            for filepath in result.stdout.strip().split("\n"):
                filepath = filepath.strip()
                if filepath and filepath not in relevant_files:
                    relevant_files[filepath] = ""
        except (subprocess.TimeoutExpired, Exception):
            continue
    return relevant_files


def read_file_with_context(filepath: str, target_line: int = None, context: int = 40) -> str:
    """Read file content, optionally centered around a target line."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        if target_line and len(lines) > 2 * context:
            start = max(0, target_line - context)
            end = min(len(lines), target_line + context)
            content = "".join(lines[start:end])
            return f"[Lines {start + 1}-{end}]\n{content}"

        if len(lines) > 250:
            return "".join(lines[:250]) + f"\n... [truncated, total {len(lines)} lines]"

        return "".join(lines)
    except FileNotFoundError:
        return f"[File not found: {filepath}]"


def build_analysis_prompt(issue_number, issue_title, issue_body, file_contents: dict) -> str:
    files_section = ""
    for path, content in file_contents.items():
        files_section += f"\n### `{path}`\n```python\n{content}\n```\n"

    return f"""你是 Stereopy（空间转录组 Python 库）的高级维护者。
请分析以下 GitHub Issue 并生成：
1. **根因分析**：问题的根本原因是什么
2. **影响范围**：哪些功能/用户受影响
3. **修复方案**：具体要修改哪些文件的哪些行，怎么改
4. **测试建议**：最小的验证方式

## Issue #{issue_number}: {issue_title}

{issue_body or '(no body)'}

## 相关代码文件
{files_section}

请严格按以下 Markdown 格式输出，不要遗漏任何章节：

## 根因分析
...

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

### 文件 2：...

## 测试建议
```python
# 最小测试用例
...
```

## 注意事项
..."""


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


def call_glm5_with_retry(client, messages, max_retries=3):
    """Call GLM-5 API with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="glm-5",
                messages=messages,
                max_tokens=4096,
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 5
                print(f"API call failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def main():
    issue_number = os.environ["ISSUE_NUMBER"]
    issue_title = os.environ["ISSUE_TITLE"]
    issue_body = os.environ.get("ISSUE_BODY", "")

    print(f"Analyzing issue #{issue_number}: {issue_title}")

    # Step 1: Extract info from issue
    traceback_files = extract_traceback_files(issue_body)
    keywords = extract_keywords(issue_title, issue_body)
    print(f"  Found {len(traceback_files)} traceback refs, {len(keywords)} keywords")

    # Step 2: Search codebase
    search_results = search_codebase(keywords)
    all_files = {}
    for tb in traceback_files:
        all_files[tb["file"]] = tb.get("line")
    for f in search_results:
        if f not in all_files:
            all_files[f] = None
    print(f"  Found {len(all_files)} potentially relevant files")

    # Step 3: Read file contents (cap at 8 files)
    file_contents = {}
    for filepath, line in list(all_files.items())[:8]:
        content = read_file_with_context(filepath, line)
        if "[File not found" not in content:
            file_contents[filepath] = content

    if not file_contents:
        for default_path in [
            "stereo/core/stereo_exp_data.py",
            "stereo/core/st_pipeline.py",
            "stereo/io/reader.py",
        ]:
            content = read_file_with_context(default_path)
            if "[File not found" not in content:
                file_contents[default_path] = content

    print(f"  Loaded {len(file_contents)} files for analysis")

    # Step 4: Call GLM-5
    client = ZhipuAI(api_key=os.environ["ZHIPUAI_API_KEY"])
    prompt = build_analysis_prompt(issue_number, issue_title, issue_body, file_contents)

    analysis = call_glm5_with_retry(client, [
        {
            "role": "system",
            "content": (
                "你是 Stereopy（空间转录组 Python 库）的高级维护者。"
                "你擅长阅读 Python 源码，定位 bug 根因，并给出精准的修复方案。"
                "请严格按照用户指定的 Markdown 格式输出，不要遗漏任何章节。"
                "代码修改建议必须包含修改前和修改后的对比。"
                "请用中文回答，代码部分用英文。"
            )
        },
        {"role": "user", "content": prompt}
    ])

    # Step 5: Write outputs
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
    }
    with open("analysis_context.json", "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2, ensure_ascii=False)

    print(f"Analysis complete.")
    print(f"  Report:      analysis_result.md")
    print(f"  Agent task:  .cursor/agent-task.md")
    print(f"  Context:     analysis_context.json")


if __name__ == "__main__":
    main()
