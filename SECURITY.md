# Security Policy

## Supported versions

Security fixes are applied to the latest release on `main`.

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Email or message the maintainer via GitHub ([@dxbh0517](https://github.com/dxbh0517)) with:

- a description of the issue
- steps to reproduce
- impact (e.g. unauthenticated dashboard access, SSRF via upstream URLs)

You should get an acknowledgement within a few days. Once a fix is available, we will credit you if you want that.

## Notes for operators

- The dashboard can change settings. Keep `DASHBOARD_HOST` on localhost (the default), or set `SWITCHYARD_TOKEN` if you bind it on the network. Prefer an SSH tunnel (`ssh -L`) or a reverse proxy that terminates TLS.
- Switchyard proxies to configured upstreams (vLLM, ComfyUI, OVMS). Treat those URLs as trusted configuration, not as user input.
