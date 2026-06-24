# Camera AI Attendance Thesis Report

This folder is an Overleaf-ready LaTeX project for the Camera AI attendance report.

## How to use on Overleaf

1. Compress the whole `thesis_report_overleaf` folder as a `.zip`.
2. In Overleaf, choose **New Project** then **Upload Project**.
3. Upload the `.zip`.
4. Set the main file to `main.tex` if Overleaf does not detect it automatically.
5. Compile with **pdfLaTeX**. The project now uses lightweight BibTeX-style references to avoid Overleaf free-plan timeouts.
6. If references show as question marks on the first run, click **Recompile** one or two more times.

## Files

- `main.tex`: main document and formatting.
- `chapters/`: report chapters and appendix.
- `references.bib`: bibliography.
- `figures/`: reserved for screenshots or future figures.

## Personalization

In `main.tex`, update these fields before submission:

```tex
\newcommand{\studentname}{Student Name}
\newcommand{\studentid}{Student ID}
\newcommand{\internalsupervisor}{Internal Supervisor}
\newcommand{\externalsupervisor}{External Supervisor}
```

The report intentionally avoids invented benchmark accuracy values because the repository does not include a labeled evaluation dataset.
