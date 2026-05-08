# Paper Website — Setup & Usage

## Setting up in your repo

1. **Copy two things** from this template into your paper repo:
   ```
   website/                              → your-repo/website/
   .github/workflows/publish-website.yml → your-repo/.github/workflows/
   ```

2. **Enable GitHub Pages** in your repo:
   - Go to Settings → Pages
   - Under "Build and deployment", set Source to **GitHub Actions**

3. **Push to `main`** — the site builds and deploys automatically.

Your site will be live at `https://<org>.github.io/<repo>`.

---

## Editing content

Everything lives in `website/index.qmd`. Edit the YAML frontmatter at the
top, then write your paper content below it in Markdown.

### Title, authors, links

```yaml
title: "Your Paper Title"

author:
  - name: "Jane Smith"
    affiliations:
      - name: "Warsaw University"
    url: "https://janesmith.com"   # optional
  - name: "John Doe"
    affiliations:
      - name: "MIT"

# Shown below author names. Format manually when authors share institutions.
# Remove to let Quarto auto-generate from the author list above.
paper-affiliations: "¹ Warsaw University  ·  ² MIT"

# Optional venue label
# paper-venue: "NeurIPS 2025"

# Links — remove any you don't need
link-paper: "https://arxiv.org/pdf/..."
link-code:  "https://github.com/..."
link-arxiv: "https://arxiv.org/abs/..."
# link-demo:    ""
# link-video:   ""
# link-poster:  ""
# link-slides:  ""
# link-dataset: ""
# link-models:  ""
```

### Abstract

Use the `.abstract` div right after the frontmatter:

```markdown
::: {.abstract}
Your abstract text. Math and citations work here.
:::
```

### Math (KaTeX)

Inline: `$x^2 + y^2 = r^2$`

Display:
```
$$
\mathcal{L}(\theta) = \sum_{i=1}^{N} \log p_\theta(y_i \mid x_i)
$$
```

### Figures

Drop images into `website/figures/` and reference them:

```markdown
![Caption text.](figures/my_figure.png){width=80% fig-align="center"}
```

### Tables

Standard Markdown tables work out of the box:

```markdown
| Column A | Column B |
|----------|----------|
| value 1  | value 2  |
```

### Citations

Add BibTeX entries to `website/references.bib`, then cite with:

```markdown
[@key]              single citation
[@key1; @key2]      multiple citations
```

The reference list is auto-generated at the `## References` section
(the `{#refs}` div at the bottom of `index.qmd`).

---

## Local preview (optional)

Install [Quarto](https://quarto.org/docs/get-started/), then:

```bash
cd website
quarto preview
```

This opens a live-reloading preview in your browser. Pushing to `main`
is enough if you prefer not to install Quarto locally.
