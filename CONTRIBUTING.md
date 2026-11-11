# Contributing to Switchyard

Thanks for helping. Small, focused changes are welcome.

## Setup

```sh
git clone https://github.com/dxbh0517/switchyard.git
cd switchyard
python3 -m pip install -e .
# Optional speech-to-text extras (needs ffmpeg on PATH):
# python3 -m pip install -e ".[stt]"
```

Run without Docker:

```sh
python3 -m switchyard
# or: switchyard
```

Set `SWITCHYARD_SETTINGS` if you want the settings file somewhere other than the default.

## Tests

Stdlib only — no GPU, no real vLLM/ComfyUI/OVMS required:

```sh
python3 -m unittest discover -s tests
```

CI runs the same command on every pull request.

## What to work on

- Issues labeled [`good first issue`](https://github.com/dxbh0517/switchyard/labels/good%20first%20issue) or [`help wanted`](https://github.com/dxbh0517/switchyard/labels/help%20wanted)
- Bug fixes with a failing test when practical
- Docs and examples that make a first run smoother

Open an issue before large design changes so we can agree on direction.

## Pull requests

1. Fork and branch from `main`.
2. Keep the arbiter / OVMS shim / dashboard stdlib-only unless the change is clearly under the optional STT path.
3. Add or update tests for behavior changes.
4. Run the test suite locally.
5. Fill in the pull request template.

## Code style

Match the surrounding code: plain Python 3, standard library where possible, short modules, no heavy frameworks for the core process.
