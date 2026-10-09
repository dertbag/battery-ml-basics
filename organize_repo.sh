#!/usr/bin/env bash
# Tidy the battery-ml-basics folder and prepare it for git.
#
# Run it from INSIDE the battery-ml-basics folder, with README.md, gitignore.txt
# and requirements.txt sitting next to it (or already in the folder).
#
#   bash organize_repo.sh            dry run: only prints what it would do
#   bash organize_repo.sh --apply    does it
#
# It only moves and copies files. It never deletes anything and never pushes.

set -u
APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

run() {
  if [ "$APPLY" -eq 1 ]; then
    "$@"
  else
    echo "  would run: $*"
  fi
}

HERE="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f soh_common.py ] && [ ! -f scripts/soh_common.py ]; then
  echo "soh_common.py not found. cd into the battery-ml-basics folder and run again."
  exit 1
fi

echo "== 1. scripts go into scripts/ =="
run mkdir -p scripts
for f in stage0.py stage1_build_table.py soh_common.py soh_models.py \
         build_dataset.py run_experiments.py summarize.py make_plots.py; do
  if [ -f "$f" ]; then run mv "$f" scripts/; fi
done

echo "== 2. raw data goes into data/raw/ (it stays out of git) =="
run mkdir -p data/raw
if [ -d Dataset_1_NCA_battery ] && [ ! -d data/raw/Dataset_1_NCA_battery ]; then
  run mv Dataset_1_NCA_battery data/raw/
fi

echo "== 3. files from other projects or mistakes go into _misc/ (ignored by git) =="
run mkdir -p _misc
for f in parity_plot.png residuals.png rul_prediction.png image.png attachment.txt; do
  if [ -f "$f" ]; then run mv "$f" _misc/; fi
done

echo "== 4. repo files =="
for f in README.md requirements.txt; do
  if [ ! -f "$f" ] && [ -f "$HERE/$f" ]; then run cp "$HERE/$f" "$f"; fi
done
if [ ! -f .gitignore ] && [ -f "$HERE/gitignore.txt" ]; then
  run cp "$HERE/gitignore.txt" .gitignore
elif [ ! -f .gitignore ] && [ -f gitignore.txt ]; then
  run cp gitignore.txt .gitignore
fi

echo "== 5. git =="
if [ ! -d .git ]; then run git init -b main; fi

if [ "$APPLY" -eq 1 ]; then
  git add -A
  echo
  echo "Staged files (first 40):"
  git status --short | head -40
  echo
  echo "Largest staged files in KB (anything over about 50000 should not be in git):"
  git diff --cached --name-only | while read -r f; do du -k "$f"; done | sort -nr | head -5
  echo
  echo "If that looks right, finish with:"
  echo "  git commit -m 'Reproduce Zhu et al. SOH from rest voltage; add MLP variant'"
  echo "  git remote add origin <YOUR-REPO-URL>"
  echo "  git push -u origin main"
else
  echo
  echo "Dry run only. Nothing changed. Run again with --apply to do it."
fi
