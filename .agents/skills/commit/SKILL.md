---
name: commit
description: Create a clean git commit for the current repository. Use when the user asks to commit, make a commit, commit staged work, or save completed changes in git.
---

# Commit

## Workflow

Use this skill when committing changes in the current repo.

1. Inspect the worktree:

```bash
git status --short
```

2. Review every candidate file before staging it:

```bash
git diff HEAD -- path/to/file
```

For untracked text files, read the file before staging it. Never use broad `git add -A` unless the user explicitly asks to include all changes.

3. Stage only files that belong to the requested commit:

```bash
git add path/to/file another/path
```

Leave unrelated modified or untracked files untouched. If the worktree already contains unrelated user changes, mention that they were left out.

4. Run relevant validation before committing when practical. Prefer the repo's existing checks and keep them proportional to the change. Examples in this repo include:

```bash
pnpm --filter @heliopause/desktop typecheck
pnpm --filter @heliopause/desktop build
```

5. Commit with exactly one Conventional Commit subject line:

```bash
git commit -m "type: Subject"
```

Allowed types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `style`, `perf`.

Subject rules:

- English only.
- Start with a capitalized verb.
- Present tense.
- No trailing period.
- No scope, no `!`, no body, no footer, no `Co-Authored-By`.
- Keep it concise, ideally under 50 characters.

Good examples:

```text
feat: Add icon chat controls
fix: Prevent blank Windows builds
chore: Add Windows exe build workflow
docs: Update inference notes
```

## After Commit

Report the commit hash and subject:

```bash
git show --stat --oneline --no-renames HEAD
```

Then show any remaining uncommitted work with:

```bash
git status --short
```

Do not clean up, revert, or stage remaining unrelated files unless the user asks.
