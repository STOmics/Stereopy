#!/bin/bash
#
# Batch-run mini-swe-agent on open issues labeled 'auto-fix'.
#
# Prerequisites:
#   pip install mini-swe-agent
#
# Usage:
#   export OPENAI_API_KEY="your_zhipu_api_key"   # GLM-5 via OpenAI-compatible API
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

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Error: OPENAI_API_KEY must be set (use your ZhipuAI key)"
    exit 1
fi

export MSWEA_COST_TRACKING="ignore_errors"

echo "=== mini-swe-agent Batch Runner ==="
echo "  Repo:    $REPO"
echo "  Limit:   $LIMIT"
echo "  Delay:   ${DELAY}s between issues"
echo "  Dry run: $DRY_RUN"
echo ""

ISSUES=$(gh issue list \
    --repo "$REPO" \
    --label "auto-fix" \
    --state open \
    --json number,title,body \
    --jq '.[:'"$LIMIT"'] | .[] | @base64')

if [ -z "$ISSUES" ]; then
    echo "No open issues with 'auto-fix' label found."
    exit 0
fi

COUNT=0
for ROW in $ISSUES; do
    ISSUE_NUM=$(echo "$ROW" | base64 -d | jq -r '.number')
    ISSUE_TITLE=$(echo "$ROW" | base64 -d | jq -r '.title')
    ISSUE_BODY=$(echo "$ROW" | base64 -d | jq -r '.body // "(no body)"')

    COUNT=$((COUNT + 1))
    echo "=== [$COUNT] Issue #$ISSUE_NUM: $ISSUE_TITLE ==="

    if [ "$DRY_RUN" = true ]; then
        echo "  (dry run — skipped)"
        continue
    fi

    # Write issue to temp file
    TASK_FILE="issue_${ISSUE_NUM}.md"
    echo "# Issue #${ISSUE_NUM}: ${ISSUE_TITLE}" > "$TASK_FILE"
    echo "" >> "$TASK_FILE"
    echo "$ISSUE_BODY" >> "$TASK_FILE"

    TASK_CONTENT=$(cat "$TASK_FILE")
    mini \
        --config config/glm5_stereopy.yaml \
        --task "$TASK_CONTENT" \
        --yolo \
        --exit-immediately \
        2>&1 | tee "swe_agent_issue_${ISSUE_NUM}.log" \
        || echo "  [WARN] mini-swe-agent failed on issue #$ISSUE_NUM"

    rm -f "$TASK_FILE"

    echo "  Done with #$ISSUE_NUM"
    if [ $COUNT -lt $LIMIT ]; then
        echo "  Waiting ${DELAY}s..."
        sleep "$DELAY"
    fi
done

echo ""
echo "=== Batch processing complete ==="
