# AlpaSim Challenge technical report template

This package is for the AlpaSim E2E Closed Loop Challenge technical report. It is **not** a NeurIPS paper submission.

Use `alpasim_technical_report.tex` as the report skeleton and compile it with `pdflatex` (or upload this folder to Overleaf). The main report text is limited to **four pages**, including figures and tables. References and an optional appendix do not count toward that limit.

Reports are not anonymous. Do not include the NeurIPS paper checklist, and do not use this package to imply that a report has been submitted to, accepted by, or endorsed by NeurIPS.

## Required content

1. Method summary and system diagram.
2. Policy description.
3. Training data, annotations, pretrained weights, and external resources.
   - Award-eligible nuPlan entries must identify the public resources and their licenses.
   - PAI-AV entries must provide a high-level disclosure of private resources and confirm rights to use them.
4. Training and inference setup, including latency and memory characteristics when relevant.
5. Expected limitations, failure modes, and safety considerations.

## NeurIPS style-file attribution and license

`neurips_2026.sty` is an AlpaSim-specific layout adaptation based on the **Formatting Instructions For NeurIPS 2026** style file authored by the **NeurIPS 2026 Program Chairs**. The underlying style was not written by the AlpaSim organizers.

The upstream template is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). This package uses an adaptation only for report layout. AlpaSim-specific changes remove conference-submission presentation and provide report-specific instructions; they do not imply NeurIPS endorsement.

Upstream sources:

- [Official NeurIPS 2026 Call for Papers](https://neurips.cc/Conferences/2026/CallForPapers)
- [Official Overleaf template](https://www.overleaf.com/latex/templates/formatting-instructions-for-neurips-2026/bjdwqfdkyftc)

The included skeleton uses the style file's `preprint` mode and suppresses its generic footer so that the output is not presented as a NeurIPS submission or publication. Do not use the `final` option: it is reserved for accepted NeurIPS papers.
