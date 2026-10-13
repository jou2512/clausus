# Clausus eval set: `cases.json`

24 short meeting texts with hand-marked confidential spans. Used to measure how many marked spans
Clausus fails to replace with a placeholder (leak rate).

## Provenance

All content is fictional and was written for this repository. Names of people, small companies,
projects, phone numbers, e-mail addresses, IBANs and AHV numbers are invented; any resemblance to
real persons or firms is coincidental. A few large, well-known Swiss organisations (a cantonal bank,
Swisscom, UBS, SBB) appear as ORG in at most four cases, because real meetings mention them.
IBANs and AHV numbers do not have valid check digits.

## License

[CDLA-Permissive-2.0](https://cdla.dev/permissive-2-0/)

## Styles

| `style` | Meaning |
|---|---|
| `hochdeutsch_transkript` | Speech-to-text output of Swiss German rendered as Standard German (fillers, repetitions) |
| `mundart` | Written Swiss German (Zurich, Bern, Basel, Lucerne, Valais), articles before names (`d'Anna`, `em Peter`) |
| `folie` | Presentation-slide text with bullet lines |
| `gemischt_en` | Standard German with English business terms (Action Item, Deadline, Steering Committee) |

Six cases per style.

## Schema

```json
{
  "version": 1,
  "cases": [
    {
      "id": "c01",
      "style": "hochdeutsch_transkript | mundart | folie | gemischt_en",
      "text": "<meeting text>",
      "secrets": [
        {"text": "<exact substring of text>", "type": "PERSON|ORG|PROJECT|PLACE|AMOUNT|DATE|PHONE|EMAIL|IBAN|AHV"}
      ]
    }
  ]
}
```

Rules:

- `secrets[].text` is an exact, case-sensitive substring of `text`.
- A secret is listed once per spelling. Different spellings of the same entity
  (e.g. `Nathalie Rossier` and later `Rossier`) are listed separately.
- A secret may occur inside a longer secret (e.g. `Odermatt` inside
  `Schmid & Odermatt Engineering GmbH`). Every occurrence of every listed string counts as must-redact.
- Generic words (customer, team, bank, budget, offer) and relative dates ("next Tuesday", "Friday")
  are not secrets.

## Held-out set: `holdout.json`

Same schema and styles, 24 cases, 191 marked spans. Written by a separate agent after the
detection fixes were made against `cases.json`, and never used to tune anything. **Only this set's
number is reported.** One known bias: the brief for this set named a few date forms ("anfangs
Dezember", "Mitte Januar") that the rules had just learned. All content is fictional; whether the
IBANs carry valid check digits was not checked.

Result 02.10.2026 (`apertus-v1.5-70b`, narrow person pass `apertus-v1.5-8b`), 3 runs, identical:
**leak rate 2.6 % (5 of 191), 3 fully and 2 partly visible.** By style: mundart 5.7 %,
gemischt_en 2.2 %, hochdeutsch_transkript 2.0 %, folie 0 %. The five: an amount written in words,
a surname that is also part of a company name (counted twice), a vague Swiss German date, and one
large public organisation.

Run: `python -m clausus.eval --cases data/eval/holdout.json --runs 3 --out <dir>`

## Baseline: Microsoft Presidio on the same held-out set (06.10.2026)

Presidio 2.2.364 with `de_core_news_lg` 3.8.0, all predefined recognizers for German, scored with
the same `score_case` (`eval_baselines/`).

| | all 191 spans | types Presidio covers (person, org, place, e-mail, phone: 122) | Swiss German (53) | masked although not secret |
|---|---:|---:|---:|---:|
| Presidio | 51.8 % leaked | 32.8 % | 52.8 % | 12 |
| Clausus | 2.6 % leaked | 2.5 % | 5.7 % | 2 |

Read with care: Presidio has no recognizer for amounts or project names, and the IBAN/AHV numbers in
this set have invalid check digits on purpose, which Presidio's checksum rejects. That is why the
second column exists. Many of Presidio's person leaks are a surname mentioned again later ("Pedrini"
after "Marco Pedrini"): it looks at each mention alone, while Clausus' vault masks every later
mention of a name it already found. Presidio runs locally in about a second for the whole set;
Clausus needs a model call per pass.

## Long texts: `long.json` and `--concat` (07.10.2026)

Every case above is under 400 characters. A real chat message or a pasted planning document is
longer, and on one such text (2755 characters, live test in the playground) the broad 70B pass
returned only three names: amounts, quarters and a person reached the answering model in clear.
Two measurements make this visible:

- `--concat 6` joins six held-out cases into one text (four texts of about 1.6k characters). Same
  191 spans, only the length changes.
- `long.json`: four new invented texts, 1.0 to 1.4k characters, 89 marked spans, in the forms such
  texts take: a planning document with a table, a meeting transcript, an e-mail thread with English
  terms, a Swiss German conversation. Same provenance and license as above; written after the
  length problem was found, and used to choose the section size below (so it is not held out).

Since then detection runs per section of at most 400 characters, cut at line ends, the sections in
parallel. A text under 400 characters is one section, so the short-case numbers are unaffected.

| Leak rate | one call per text | 700-char sections | 400-char sections |
|---|---:|---:|---:|
| held-out joined (`--concat 6`, 191 spans) | 9.4 % (2 runs) | 4.7 % (2 runs) | **2.1 %** |
| `long.json` (89 spans) | 6.7 % (2 runs) | 1.1 % (2 runs) | **0.0 %** |
| held-out as short cases (191 spans) | 1.6 % | | 1.6 % |

400-char sections: one run each. The cost is more calls per text (three per section) and a few more
spans masked that are not secret (5 on `long.json`, mostly table labels). What still leaks on the
joined held-out set is the hard part of the short set too: amounts written as words and vague
Swiss German dates.

Run: `python -m clausus.eval --cases data/eval/holdout.json --concat 6` ·
`python -m clausus.eval --cases data/eval/long.json`

## Over-masking set: `traps.json`

8 cases, 113 marked spans, same schema, provenance and license as above. Written by a separate
agent on 08.10.2026 without seeing the detection code.

Purpose: measure extra masking (the `extra_masked` count), not leaks. Each text (600 to 1200
characters) is full of words that must stay in clear: roles and bodies without a name
(Geschäftsleitung, Verwaltigsrat, Gmeind, Vorstand, Einkauf, HR), weekdays and times (Mäntig,
Fritig, um nüüni, 14 Uhr), relative dates (nächscht Wuche, Ende Monat), business and IT terms
(Offerte, Rechnig, Schnittstelle, Rollout, Deadline, Onboarding, NDA, Go-live, Sprint, Release),
document names (Protokoll, Agenda, Pendenzenliste, Adresstabelle), public software used generically
(Excel, Teams, SharePoint, Abacus, SAP, Salesforce), small counts and percentages, and Swiss German
function words that look like names (`wil`, `Ins`, `au`), some at sentence start. Real secrets of
every type are mixed in, so leaks are measured too.

Styles: 3 `mundart` (Zurich, Bern, Basel with a Lucerne speaker), 2 `hochdeutsch_transkript`,
2 `gemischt_en`, 1 `folie`.

This set targets known failure classes on purpose, so it is a targeted test, not a blind held-out
set.

Run: `python -m clausus.eval --cases data/eval/traps.json`

## State at submission (13.10.2026)

Commit `4c8a69b`, one run per set, without → with the judge (the shipped default is with):
held-out 0.5 % → 0.5 % (1 of 191), held-out joined 0.5 % → 1.1 %, `long.json` 0.0 % → 0.0 %,
`traps.json` 0.9 % → 0.9 %, `cases.json` 0.6 % → 0.6 %. Spans masked although not secret:
2 → 1, 3 → 1, 5 → 3, 11 → 5, 1 → 1. The held-out set has not been blind since the leaks of
02.10. were seen, so 2.6 % stays the first-contact number. Details: `technical_report.md` § 5.
