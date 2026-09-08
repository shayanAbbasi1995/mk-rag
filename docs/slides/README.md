# M731 Slide Decks (Fall 2026)

One revealjs deck per session, each self-contained in its own
`Session N/` folder, plus a shared `_quarto.yml` at the project root
for theme and code settings (Quarto projects apply root-level config
to every `.qmd` in the tree, however deeply nested).

## Structure

- `_quarto.yml` — shared theme (McMaster maroon + gold accents), footer, code settings
- `README.md` — this file
- `data-raw/generate_datasets.R` — single reproducible script (fixed
  seed) that generates every simulated dataset used across all 13
  sessions; writes to `data-raw/output/`, from which copies are placed
  into each `Session N/data/` folder
- `Session 1/` … `Session 13/` — one folder per week, each containing:
  - `session-NN.qmd` (+ rendered `session-NN.html`) — the in-class deck
  - `solutions-NN.qmd` (+ rendered `.html`) — a full worked walkthrough
    of that week's case-study analysis (code, real output,
    interpretation), referenced from a pointer slide in the deck.
    Not present for Session 13 (presentation day; no case-study dataset).
  - `data/` — that session's dataset(s) as local files, matching the
    relative `read_csv("data/...")` paths used in the deck's code chunks

Assignment content (due dates, rubrics, "what to submit" briefs) has
been deliberately removed from every deck — assignments are built and
maintained separately by the instructor. Test dates and group-project
deliverable dates remain, since those are schedule facts, not
assignment content.

## Speaker notes

Each deck uses Quarto's `::: {.notes} ... :::` divs for instructor
speaker notes. These are invisible in the main slide view and visible
only when pressing **`S`** in the presentation to open the speaker
view.

Where useful, `<!-- HTML comments -->` in the .qmd source add
structural cues visible only to whoever is editing the file.

## Rendering

From this folder (renders every `.qmd` in every `Session N/` subfolder):

```powershell
& "$env:LOCALAPPDATA\Programs\Quarto\bin\quarto.exe" render .
```

Or per-file:

```powershell
& "$env:LOCALAPPDATA\Programs\Quarto\bin\quarto.exe" render "Session 1/session-01.qmd"
& "$env:LOCALAPPDATA\Programs\Quarto\bin\quarto.exe" render "Session 1/solutions-01.qmd"
```

Rendered `.html` files land beside each `.qmd`. Each is
self-contained (`embed-resources: true`) — send a single HTML file
and it opens anywhere. Solutions files execute their R code for real
(`execute: eval: true` in their own front matter, overriding the
project default of `eval: false` used by the slide decks) — regenerate
the datasets first (see below) if you haven't already.

## Regenerating the datasets

All datasets are instructor-simulated and fully reproducible from one
script:

```powershell
cd data-raw
Rscript generate_datasets.R
```

This writes every CSV/TXT to `data-raw/output/`. Copy the relevant
file(s) into each `Session N/data/` folder per the dataset map below.

## Presenting

Open the HTML in a modern browser. Keyboard:

- **`space` / arrows** — navigate
- **`S`** — speaker notes view (opens a second window with notes + timer)
- **`F`** — fullscreen
- **`?`** — full keyboard help
- **`ESC`** — slide overview

## Session map

| Folder     | Wk | Date       | Topic                                         | Dataset(s) in `data/`                              | Solutions file        |
|------------|----|------------|------------------------------------------------|-----------------------------------------------------|------------------------|
| Session 1  | 1  | Sept. 9    | Research process + R setup                    | `maritime_transactions.csv`                          | `solutions-01.qmd`    |
| Session 2  | 2  | Sept. 16   | Secondary data + research designs             | `statscan_coffee.csv`                                 | `solutions-02.qmd`    |
| Session 3  | 3  | Sept. 23   | Measurement, data collection, sampling        | `maritime_survey.csv`                                 | `solutions-03.qmd`    |
| Session 4  | 4  | Sept. 30   | Survey design                                 | `kaggle_survey_raw.csv` (simulated stand-in)          | `solutions-04.qmd`    |
| Session 5  | 5  | Oct. 7     | Qualitative research + Test 1                 | `maritime_focus.txt`                                  | `solutions-05.qmd`    |
| Session 6  | 6  | Oct. 14    | Data analysis + hypothesis testing            | `maritime_survey.csv`                                 | `solutions-06.qmd`    |
| Session 7  | 7  | Oct. 21    | Data analysis + hypothesis testing (cont.)    | `maritime_survey.csv`                                 | `solutions-07.qmd`    |
| Session 8  | 8  | Oct. 28    | Marketing scales                              | `maritime_survey.csv`                                 | `solutions-08.qmd`    |
| Session 9  | 9  | Nov. 4     | Factor analysis                               | `maritime_survey.csv`                                 | `solutions-09.qmd`    |
| Session 10 | 10 | Nov. 11    | Experimental research                         | `maritime_abtest.csv`                                 | `solutions-10.qmd`    |
| Session 11 | 11 | Nov. 18    | Regression + Test 2                           | `maritime_survey.csv`, `maritime_abtest.csv`          | `solutions-11.qmd`    |
| Session 12 | 12 | Nov. 25    | Reports + presentations + guest speaker       | `maritime_survey.csv`                                 | `solutions-12.qmd`    |
| Session 13 | 13 | Dec. 2     | Team presentations                            | *(none — presentation day, no case-study component)* | *(none)*              |

`maritime_survey.csv` and `maritime_abtest.csv` are copied identically
into every session folder that uses them, so each `Session N/` stays
fully self-contained — there is no cross-folder dependency when
opening or sharing a single session.
