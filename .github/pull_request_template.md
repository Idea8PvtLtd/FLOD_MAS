## What and why

<!-- What changes, and the problem it solves. Link the issue: Closes #123 -->

## Behaviour on site

- [ ] New behaviour is behind a setting whose default keeps the current behaviour
- [ ] Older `config.json` / `machine_settings.json` files still load
- [ ] No change to what operators see or do — or it is described above

## Checks

- [ ] Tests added or updated; `python -m unittest discover -s tests -p "test_*.py"` passes
- [ ] `requirements.lock` regenerated if `requirements.txt` changed
- [ ] `packaging/model-checksums.sha256` updated if a model changed
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No secrets, customer images or recordings added
