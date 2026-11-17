# Third-party notices

- Harbor and Terminus-KIRA are derived from the selected Harbor fork. The original
  Apache-2.0 license is retained at `terminus/LICENSE`. Source-file
  copyright notices remain in place.
- slime retains the complete THUDM/slime baseline with the documented CompactionRL
  overlay selected from the development fork. Its
  Apache-2.0 license is retained at `slime/LICENSE`. Third-party
  attribution in individual framework files remains applicable.
- The agentrollforge snapshot identified in [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) was used
  to inspect protocol behavior. Its historical Rust implementation is not
  distributed here; this repository provides the documented TiTO executable
  and a minimal SQLite buffer.
- External dependencies, model checkpoints, benchmark datasets and container base
  images retain their respective licenses. This code license does not relicense
  those assets.

The supplied review manuscript was anonymous. Public title and author metadata
in the main README are taken from the [arXiv publication record](https://arxiv.org/abs/2607.05378).
The README figures are cropped from the project manuscript; see
[figure provenance](assets/README.md).

The TiTO executable contains the project’s CPython bytecode payload and a small
ELF bootstrap, under this project’s Apache-2.0 license. Its third-party Python
libraries are runtime dependencies and are not bundled in the executable. See
`terminus/bin/README.md` and `terminus/bin/tito-linux-x86_64.json` for platform and provenance details.
