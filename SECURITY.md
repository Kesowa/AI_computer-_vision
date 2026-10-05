# Security Policy

## Reporting a vulnerability

Please report security and privacy issues privately rather than opening a public
issue.

- Use GitHub's [private vulnerability reporting](https://github.com/Kesowa/AI_computer-_vision/security/advisories/new)
  on this repository, or
- email **security@kesowa.com** with the details.

We will acknowledge your report and tell you whether we intend to act on it.

## Privacy reports are especially welcome

This repository previously contained facial photographs of identifiable people,
committed as sample data in 2021 and removed from the full git history in 2026.

**If you find any remaining personal data here — a face, a name, a location that
identifies someone, or a dataset that should not have been published — please
tell us.** We would rather hear it from you than leave it up. Reports of this
kind are treated with the same priority as a security vulnerability, and you do
not need to demonstrate any technical exploit.

Note that a fork of this repository exists outside Kesowa's control. Rewriting
our history does not reach it. If you find removed content still reachable
somewhere, that is useful to know.

## Scope

This repository contains **research code and model weights**. It is not a
deployed service, and `server.py` is an unfinished sketch that was never run in
production. There is no Kesowa system behind this code to attack.

What is in scope:

- **Personal data** of any kind, as above.
- **Committed credentials** anywhere in the repository or its history.
- **Malicious or tampered model weights.** A `.h5`, `.pt`, `.onnx` or `.pkl`
  file is executable input to the frameworks that load it — pickle-based formats
  especially. If a weight file here does something unexpected on load, that is a
  serious finding.
- **Code execution through a crafted input image or raster**, in a path a user
  would plausibly feed untrusted data to.
- **Vulnerable dependencies** in any requirements file.

## Out of scope

- Hardcoded absolute paths like `C:\Users\FS-AI\...`. These are known, documented
  in the README, and disclose nothing but a developer's directory layout.
- The internal `cog.kesowa.com` endpoint in `Tree_detection/fetch_from_api.py`.
  It is not publicly reachable and is not a credential.
- Scripts that do not run, or requirements files that no longer resolve. These
  are bugs, not vulnerabilities — open a normal issue.
- `api_key = "APIKEY"` in `Bengali_text_detection`. That is a placeholder, not a
  leaked key. Its poor design is noted in CONTRIBUTING.
- Findings in TensorFlow, PyTorch, DeepForest or other upstream dependencies.
  Report those to their maintainers.

## If you plan to run this code

Model weight files are untrusted input. `label_encoder.pkl`-style pickle files
and `.pt` checkpoints can execute arbitrary code when loaded. Only load weights
you obtained from a source you trust, and prefer ONNX, which is data rather than
code, where a project offers both.
