# Production feature inventory

- Maintain `docs/production-feature-inventory.md` whenever production functionality is added, changed, removed or reverted, or an upstream merge changes a feature's status. Update it in the production integration commit or an accompanying documentation commit.
- Keep this installation-specific inventory and its README link on `production`; do not carry them into isolated feature branches or upstream PRs.
- Verify the actual production and upstream `main` trees and the PR's merged status. A closed PR, branch name or matching commit title alone does not establish that functionality reached upstream.
- Update the last functional review date and revisions, status table, affected feature descriptions, timeline and reproduction guidance. Preserve historical milestones and distinguish source defaults from verified live settings.
- Treat the recorded revisions and counts as the last reviewed functional snapshot. Documentation-only commits do not require a new functional snapshot or self-referencing commit hash. Keep source links portable and retain the distinction between custom functionality, upstream functionality and separate experiments.
