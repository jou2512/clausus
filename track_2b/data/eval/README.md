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
