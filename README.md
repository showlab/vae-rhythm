# VAE Rhythm project page

Static project page for *VAE Rhythm: Temporal Decoding Traces for AI-Generated Video Detection*.
Plain HTML, CSS and JavaScript: no build step, no external requests, no third-party code or fonts.
It works from GitHub Pages and when `index.html` is opened directly from disk.

```
index.html                 the page; the link config block is at the top of <head>
.nojekyll                  tells GitHub Pages to serve the files as they are
assets/css/style.css       styles, light and dark (follows the system setting)
assets/js/main.js          link config, copy buttons, table toggle, the three charts
assets/data/page-data.js   numbers behind the charts (see "Data" below)
assets/img/*.webp          figures exported from the paper
assets/img/social-card.jpg 1200 x 630 preview image for link sharing
```

## Deploy on GitHub Pages

Either option works; the page uses only relative paths.

**From a `gh-pages` branch**

1. Copy the contents of this folder (including `.nojekyll`) to the root of a branch named `gh-pages`.
2. In the repository, open *Settings > Pages*, choose *Deploy from a branch*, select `gh-pages` and `/ (root)`, and save.
3. The page appears at `https://showlab.github.io/vae-rhythm/` after a minute or two. It belongs to the repository `https://github.com/showlab/vae-rhythm`, and the arXiv paper links to exactly this address.

**From a `/docs` folder**

1. Copy the contents of this folder (including `.nojekyll`) into `docs/` on the default branch.
2. In *Settings > Pages*, choose *Deploy from a branch*, select the default branch and `/docs`, and save.

To preview locally, open `index.html` in a browser, or run `python3 -m http.server` in this folder and visit `http://localhost:8000`.

## Placeholders to fill before publishing

- [ ] **Links**: edit `window.PROJECT_LINKS` at the top of `index.html`. Code (https://github.com/showlab/vae-rhythm) and the live demo are set; paper (arXiv) and models / data (Hugging Face) are still `"#"` and show as "coming soon".
- [x] **Authors**: Ke Sun, Shuwei Li, Haiyang Mei, Mike Zheng Shou (corresponding author), as in the paper.
- [x] **Affiliations**: Show Lab, National University of Singapore.
- [ ] **BibTeX**: authors are filled; replace `arXiv:XXXX.XXXXX` after the arXiv submission (search for `TODO BibTeX`).
- [x] **Acknowledgements**: removed, as in the arXiv paper.
- [ ] **Run locally**: replace the placeholder install and scan commands with the released ones, and delete the "not public yet" sentence (search for `PLACEHOLDER`).
- [x] **Abstract**: now the same text as the arXiv paper.
- [x] **Social preview**: `og:image` and `og:url` point to https://showlab.github.io/vae-rhythm/.

## Data

`assets/data/page-data.js` holds the numbers the interactive charts draw:

- the mean phase-scan residual curves and their standard errors for all 42 panels of the paper's rhythm atlas (39 generator variants, one variant with a second window setting, and the two real cohorts), exactly as plotted in the paper;
- the per-variant statistics of the paper's supplementary table of real-versus-generated statistics, with the source VAE and 720p AUROC of the census table;
- the AUROC of the 19 detectors with and without the VAE Rhythm plug-in, from the paper's full plug-in table.

The main results table is written directly in `index.html`. Figures in `assets/img/` are exports of the paper's figures with no metadata.
