#!/bin/bash
#
# Batch-run SWE-agent on all open issues labeled 'auto-fix'.
#
# Usage:
#   export ZAI_API_KEY="your_zhipu_api_key"
#   export GITHUB_TOKEN="your_github_token"
#   bash scripts/ci/batch_swe_agent.sh [--dry-run] [--limit N] [--delay SECONDS]
#
set -euo pipefail

DRY_RUN=false
LIMIT=5
DELAY=60
REPO="STOmics/Stereopy"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)   DRY_RUN=true; shift ;;
        --limit)     LIMIT="$2"; shift 2 ;;
        --delay)     DELAY="$2"; shift 2 ;;
        --repo)      REPO="$2"; shift 2 ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "${ZAI_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Error: ZAI_API_KEY or OPENAI_API_KEY must be set"
    exit 1
fi

API_KEY="${ZAI_API_KEY:-$OPENAI_API_KEY}"

echo "=== SWE-agent Batch Runner ==="
echo "  Repo:    $REPO"
echo "  Limit:   $LIMIT"
echo "  Delay:   ${DELAY}s between issues"
echo "  Dry run: $DRY_RUN"
echo ""

ISSUES=$(gh issue list \
    --repo "$REPO" \
    --label "auto-fix" \
    --state open \
    --json number,title \
    --jq '.[:'"$LIMIT"'] | .[] | "\(.number)\t\(.title)"')

if [ -z "$ISSUES" ]; then
    echo "No open issues with 'auto-fix' label found."
    exit 0
fi

echo "Found issues:"
echo "$ISSUES" | while IFS=$'\t' read -r num title; do
    echo "  #$num: $title"
done
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "(dry run — no actions taken)"
    exit 0
fi

COUNT=0
echo "$ISSUES" | while IFS=$'\t' read -r ISSUE_NUM ISSUE_TITLE; do
    COUNT=$((COUNT + 1))
    echo "=== [$COUNT] Processing issue #$ISSUE_NUM: $ISSUE_TITLE ==="

    sweagent run \
        --config config/glm5_stereopy.yaml \
        --agent.model.api_key="$API_KEY" \
        --env.repo.github_url="https://github.com/${REPO}" \
        --problem_statement.github_url="https://github.com/${REPO}/issues/${ISSUE_NUM}" \
        --actions.apply_patch_locally \
        2>&1 | tee "swe_agent_issue_${ISSUE_NUM}.log" \
        || echo "  [WARN] SWE-agent failed on issue #$ISSUE_NUM"

    echo "  Done with #$ISSUE_NUM"
    echo "  Waiting ${DELAY}s..."
    sleep "$DELAY"
done

echo ""
echo "=== Batch processing complete ==="
