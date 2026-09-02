# Third-party project boundary

This repository does not contain source code copied from the projects below.
They informed product requirements and adapter boundaries only. Clean-room
implementations in Jieyi include strict placeholder round trips, distributed
document sampling, project-scoped fuzzy translation memory, and prompt preview;
the code, data model, naming, and tests were written independently for Jieyi.

- TranslateBooksWithLLMs (TBL): AGPL-3.0  
  <https://github.com/hydropix/TranslateBooksWithLLMs>
- Supervertaler Workbench: MIT  
  <https://github.com/Supervertaler/Supervertaler-Workbench>

Before importing or adapting source from either project, record the exact files,
copyright notices, license obligations, and distribution decision in this file.
In particular, do not add TBL source to a closed-source or network-hosted product
without an explicit AGPL compliance decision.

Jieyi original code is distributed under the MIT License; see `LICENSE`.
Third-party dependencies retain their own licenses. Python dependency versions
are recorded in `uv.lock`; web dependency versions and their available license
metadata are recorded in `web/package-lock.json`. The project license does not
relicense those dependencies or user-imported books and translation content.
