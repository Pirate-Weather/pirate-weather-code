# AGENTS Instructions

## Coding Standards

Follow the conventions in `/.gemini/styleguide.md` when generating or reviewing code.

## Pull Request Reviews

Prioritize feedback in this order:

1. Correctness
2. API compatibility
3. Security
4. Performance
5. Maintainability
6. Style

Avoid commenting on formatting or import ordering handled automatically by Ruff.
Prefer consistency with the surrounding code when working with legacy modules.

## Testing

- New functionality should include tests where practical.
- Bug fixes should include a regression test when feasible.
- Avoid reducing existing test coverage.
- Prefer deterministic tests over time- or network-dependent tests.

## Development

This repository uses **Ruff** for linting and formatting.

Run lint checks:

```bash
scripts/lint-fix
```

Format the code:

```bash
scripts/format
```

Run the full test suite before submitting changes:

```bash
pytest
```

When making code changes:

- Run the appropriate linting, formatting, and tests before considering the work complete.
- Add comments or docstrings for non-obvious logic where they improve readability.
- Do not add comments that simply restate what the code already does.