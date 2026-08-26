# Teaching calendar — ML I & ML II, PEU-CD 2026

Builds the ML I / ML II schedule and the hour split between the instructor and
the teaching assistant, starting from the two official calendars.

## The scheduling constraint

The ENEI program runs Machine Learning on **Monday, Wednesday and Friday
evenings, 19:00–22:00**. During the same weeks the instructor teaches Python and
Intermediate Python at the **PUCP Diploma**, on **Wednesday evenings**, and that
calendar is fixed and cannot be moved.

Every Wednesday of the ENEI term therefore clashes, and is covered by the
teaching assistant as a lab session.

## Assignment rules

1. Each course is 30 hours: **18 h for the instructor** (6 sessions) and
   **12 h for the teaching assistant** (4 sessions), matching the 2025 edition.
   Every session is 3 hours long.
2. Every session that clashes with the PUCP calendar goes to the teaching assistant.
3. If the clashes do not add up to the assistant's 12 hours, the last sessions of
   the course are reassigned until they do.

For ML I the four Wednesdays land exactly on the assistant's 12 hours. For ML II
they only add up to 9 h, so the closing session (Monday, October 26) is
reassigned as a lab.

## Resulting split

| Course | Instructor — Alexander Quispe | Teaching assistant — Rodrigo Grijalba |
|---|---|---|
| **ML I** | Sep 7, 11, 14, 18, 21, 28 (18 h) | Sep 9, 16, 23, 30 (12 h) |
| **ML II** | Oct 2, 9, 12, 16, 19, 23 (18 h) | Oct 7, 14, 21, 26 (12 h) |

## Layout

```
data/raw/         Official calendars, as received (not edited)
data/processed/   Tidy CSVs produced by the parser
src/              Parsing and schedule-building code
output/           The Excel file shared with the ENEI and the teaching assistant
```

### Source files (`data/raw/`)

| File | What it is |
|---|---|
| `enei_peu_cd_2026_program_calendar.xlsx` | ENEI calendar for PEU-CD 2026: which course runs on each date, plus the 18:00–19:00 INEI talks |
| `pucp_diploma_2026_official_calendar.pdf` | PUCP Diploma 2026 official calendar — the fixed constraint |
| `enei_peu_cd_2025_ml_schedule_reference.xlsx` | 2025 edition, used as the template for the 18/12 hour split and the layout |

## Reproducing

```bash
pip install -r requirements.txt

cd schedule
python src/parse_calendars.py   # raw calendars -> data/processed/*.csv
python src/build_schedule.py    # assignment rules -> output/*.xlsx
```

`build_schedule.py` asserts that each course ends up at 18 h + 12 h and that no
session left on the instructor collides with the PUCP calendar, so a change in
either source calendar fails loudly instead of producing a wrong schedule.

## Output

`output/horario_peu_cd_2026_ml1_ml2.xlsx` — in Spanish, three sheets:

- **RESUMEN** — courses, who teaches what, hours, start/end dates, notes on the constraint
- **HORARIO DETALLE ML-I ML-II** — week-by-week calendar, with the rest of the program in grey for context
- **DETALLE SESIONES** — all 20 sessions listed, with the clash flagged on each Wednesday and an hour check
