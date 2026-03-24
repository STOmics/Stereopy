"""
Batch label GitHub issues with 'auto-fix' to trigger the auto-fix pipeline.
Supports priority filtering, dry-run mode, and rate limiting.

Usage:
    export GITHUB_TOKEN=your_token
    python scripts/ci/batch_process_issues.py --priority P0 --limit 5 --dry-run
    python scripts/ci/batch_process_issues.py --priority all --limit 10
"""

import os
import time
import argparse

from github import Github


PRIORITY_KEYWORDS = {
    "P0": [
        "Error", "crash", "TypeError", "KeyError", "IndexError",
        "AttributeError", "broken", "fail", "bug", "Traceback",
    ],
    "P1": [
        "incorrect", "wrong", "unexpected", "regression", "performance",
        "slow", "warning", "deprecated",
    ],
    "P2": [
        "enhancement", "feature", "request", "support", "add",
        "improve", "documentation", "question",
    ],
}


def classify_priority(title: str, body: str) -> str:
    text = f"{title} {body or ''}".lower()
    for priority, keywords in PRIORITY_KEYWORDS.items():
        if any(kw.lower() in text for kw in keywords):
            return priority
    return "P2"


def main():
    parser = argparse.ArgumentParser(
        description="Batch label issues for auto-fix pipeline"
    )
    parser.add_argument(
        "--repo", default="STOmics/Stereopy",
        help="GitHub repo in owner/name format (default: STOmics/Stereopy)"
    )
    parser.add_argument(
        "--priority", choices=["P0", "P1", "P2", "all"], default="P0",
        help="Only process issues matching this priority (default: P0)"
    )
    parser.add_argument(
        "--limit", type=int, default=5,
        help="Max number of issues to label (default: 5)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only print what would be done, don't apply labels"
    )
    parser.add_argument(
        "--delay", type=int, default=120,
        help="Seconds between labeling issues to avoid overloading CI (default: 120)"
    )
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable is required.")
        print("  export GITHUB_TOKEN=ghp_xxxxxxxxxxxx")
        return

    gh = Github(token)
    repo = gh.get_repo(args.repo)

    print(f"Fetching open issues from {args.repo}...")
    issues = repo.get_issues(state="open", sort="created", direction="asc")

    candidates = []
    for issue in issues:
        if issue.pull_request:
            continue
        existing_labels = [label.name for label in issue.labels]
        if "auto-fix" in existing_labels:
            continue

        priority = classify_priority(issue.title, issue.body)
        if args.priority != "all" and priority != args.priority:
            continue

        candidates.append((priority, issue))

    candidates.sort(key=lambda x: x[0])
    candidates = candidates[:args.limit]

    if not candidates:
        print("No matching issues found.")
        return

    print(f"\nFound {len(candidates)} issues to process:\n")
    for priority, issue in candidates:
        labels = ", ".join(l.name for l in issue.labels)
        print(f"  [{priority}] #{issue.number}: {issue.title}")
        if labels:
            print(f"         labels: {labels}")

    if args.dry_run:
        print("\n(dry run — no labels applied)")
        return

    print()
    for i, (priority, issue) in enumerate(candidates):
        print(f"Labeling #{issue.number} with 'auto-fix'...")
        try:
            issue.add_to_labels("auto-fix")
            print(f"  ✓ Done")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue

        if i < len(candidates) - 1:
            print(f"  Waiting {args.delay}s before next...")
            time.sleep(args.delay)

    print(f"\nDone. {len(candidates)} issues labeled for auto-fix.")


if __name__ == "__main__":
    main()
