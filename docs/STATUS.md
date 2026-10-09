# If this file disagrees with the code, the code is right and this file is a bug.

Updated: 2026-10-07

## What this tree is

RegesARC is one launcher. Strata and Colibri are not checkouts inside this repo.
Their placement ideas are the spec. Their copyrights are in NOTICE.

## What runs today

- `python setup.py --list` scans the machine and verdicts every catalog model.
  RegesCore 397B is #1 recommendation. All MoE families present (qwen, deepseek, glm, kimi, olmoe, inkling, laya).
- Models are stored in `RegesModels/` beside the checkout, never in git.
- `python setup.py --toolchain` writes `tools/installers/install-toolchain.bat`
  and `.sh`. It does not run them. It does not bind port 8080.
- Toolchain probe works: detects gcc (MSYS2), make, python, nvcc on this machine.

## What does not run today

- No engine binary. `engines/` holds the contract, not Strata and not Colibri.
- No download, no serve, no chat.
- A 170B figure is weights on disk. It is not source, and it is not compiled.

## Next, in order

1. Install/download flow: user picks models from catalog, installer downloads to `RegesModels/`.
2. Engine source absorbed into `engines/` as family adapters (qwen38 first).
3. Serve layer: OpenAI + Anthropic on one port.
4. Chat interface.
