# Contributing

This repository is an **archive of past research**, not an actively developed
project. Most of the code here answered a question some years ago and then
stopped. Contributions are welcome, but set your expectations accordingly: there
is no roadmap, and review may be slow.

## What is most useful

In rough order of value:

1. **Making a project runnable again.** Pinning a requirements file to versions
   that actually resolve, or replacing a hardcoded `C:\Users\FS-AI\...` path with
   an argument, helps anyone who comes here next.
2. **Documenting provenance.** If you can establish where a model weight file or
   a pretrained checkpoint actually came from, that fills a real gap — see
   [NOTICE](NOTICE), where several lineages are marked unverified.
3. **Removing dead weight.** Several near-duplicate scripts differ only in an
   output format or a threshold. Consolidating them, without losing behaviour
   anyone depends on, would make the repository easier to read.
4. **Bug fixes** in code that still runs.

## What to avoid

- **Do not commit datasets.** Especially not images of people. This repository
  had its history rewritten to remove facial photographs, and we are not
  repeating that. Sample data belongs in a release asset or an external store.
- **Do not commit model weights.** There are already around 200 MB of them here,
  which is why cloning is slow. New weights should go to a release or a model
  registry, not into git.
- **Do not copy in code from a GPL or AGPL project.** A copy of YOLOv7's
  `detect.py` (GPL-3.0) was removed from this repository for exactly this
  reason: it cannot be redistributed under the MIT licence here. If you need
  upstream functionality, depend on the upstream package or document the
  licence explicitly — do not paste the file in.

## Licensing of contributions

By opening a pull request you agree your contribution is MIT licensed and that
you have the right to license it that way. If your change incorporates
third-party code, say so, name its licence, and confirm it is MIT-compatible.

Permissive licences (MIT, BSD, Apache-2.0) are fine. GPL, AGPL and
non-commercial-research licences are not.

## Pull requests

1. Open an issue first for anything beyond a small fix.
2. Keep the change scoped to one project directory where possible. The projects
   are independent and are better reviewed independently.
3. Say what you ran and on what data. There are no automated tests, so your
   description is the only evidence a change works.
4. Check your diff for absolute paths, API keys, personal data and weight files
   before pushing.

## Style

Standard Python conventions for new code. Do not reformat existing files
wholesale — it makes the diff unreadable and this code is read more often than
it is run.

## Secrets

Never commit credentials. `Bengali_text_detection` currently has an `api_key`
placeholder assigned inline; if you touch that file, move it to an environment
variable rather than leaving a slot where a real key could be typed and
committed.

If you commit a secret by accident, treat it as compromised and rotate it.
Removing it in a later commit does not remove it from history.
