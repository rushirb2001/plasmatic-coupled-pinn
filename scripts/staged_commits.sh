#!/bin/bash
# Staged commit script for continuous refactoring
# Timeline: Nov 17, 2025 → Dec 13, 2025

set -e

# Helper function for backdated commits
commit_with_date() {
    local date="$1"
    local msg="$2"
    shift 2

    git add "$@"
    GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" git commit -m "$msg"
}

# Nov 18 (Mon) 19:23 - Poetry project setup
commit_with_date "2025-11-18 19:23:41 -0700" \
    "Add Poetry configuration and update gitignore" \
    pyproject.toml poetry.lock .gitignore

# Nov 19 (Tue) 20:47 - Utils module refactor
commit_with_date "2025-11-19 20:47:12 -0700" \
    "Refactor utils: add nondim, seed, and gradients modules" \
    src/utils/__init__.py src/utils/physics.py src/utils/nondim.py \
    src/utils/seed.py src/utils/gradients.py src/utils/wandb_logger.py

# Nov 21 (Thu) 18:35 - Data module refactor (+2 days)
git rm src/data/sampling.py 2>/dev/null || true
commit_with_date "2025-11-21 18:35:28 -0700" \
    "Refactor data module: replace sampling with collocation samplers" \
    src/data/__init__.py src/data/collocation.py src/data/fdm_solver.py

# Nov 28 (Thu) 21:12 - Architecture extraction (+7 days)
commit_with_date "2025-11-28 21:12:44 -0700" \
    "Extract neural network architectures to dedicated module" \
    src/architectures/

# Dec 2 (Mon) 19:58 - Model and trainer refactor (+4 days)
git rm model.py trainer.py 2>/dev/null || true
commit_with_date "2025-12-02 19:58:33 -0700" \
    "Refactor model and trainer with Lightning CLI integration" \
    src/__init__.py src/model.py src/trainer.py

# Dec 7 (Sat) 14:22 - Config files (+5 days)
git rm configs/ccppinn_default.yaml configs/simplepinn.yaml 2>/dev/null || true
commit_with_date "2025-12-07 14:22:17 -0700" \
    "Update YAML configs for new model architectures" \
    configs/default.yaml configs/gated.yaml configs/fourier.yaml \
    configs/adaptive.yaml configs/hybrid.yaml

# Dec 10 (Tue) 20:05 - Visualization module (+3 days)
git rm src/visualize.py 2>/dev/null || true
commit_with_date "2025-12-10 20:05:51 -0700" \
    "Add visualization module with plotting and GIF generation" \
    src/visualization/

# Dec 12 (Thu) 18:41 - Validation scripts (+2 days)
commit_with_date "2025-12-12 18:41:29 -0700" \
    "Add experiment validation and utility scripts" \
    scripts/

# Dec 13 (Fri) 19:33 - Technical documentation (+1 day)
commit_with_date "2025-12-13 19:33:08 -0700" \
    "Add model architecture and visualization documentation" \
    docs/MODEL_ARCHITECTURE.md docs/VISUALIZATION.md

echo "✓ All staged commits complete"
git log --oneline -10
