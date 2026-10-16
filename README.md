# Clausus

A masking layer, built with Apertus for [Hack Apertus](https://hackapertus.ch/)
Track 2B, that keeps confidential content out of every model call.

Clausus takes text, slides and Swiss German audio, finds what is confidential
(people, companies, projects, amounts, dates, contact data) and replaces it with
placeholders before any model sees it. The model
only ever sees `[PERSON_1]`, `[ORG_2]`, `[AMOUNT_1]`. The originals stay in a
vault on your side and are put back into the answer at the end. Every run
leaves a receipt that lists what left Clausus, and whether it was masked or
original.

Clausus runs on its own (playground, chat, API) or as a guardrail in a LiteLLM
proxy, in front of any model behind an OpenAI-compatible API. If Clausus is down,
the proxy lets nothing through.

- **Demo video (2 min):** <https://youtu.be/OSoTQuUp_lg>
- **Report (PDF):** [`track_2b/Clausus_Report.pdf`](track_2b/Clausus_Report.pdf),
  source [`track_2b/technical_report.md`](track_2b/technical_report.md)
- **Code:** everything lives in [`track_2b/`](track_2b/)

## Why

Swiss companies want to use language models on their notes, documents, slides and
meetings, which are mostly in Swiss German. At the same time they have to stay
compliant: client names, amounts and projects must not end up with a US cloud
provider. Today they have to choose between a model that understands Swiss German
and one they are allowed to use. Clausus removes that choice, built on Apertus
because it is open, made in Switzerland and runs where the company decides.

## Status and where it is going

This is the hackathon prototype. The playground is built to make every step
visible, not as the final interface. The direction is a package a company installs
once, with its own trained models plugged in: every model Clausus uses already sits
behind a configurable OpenAI-compatible endpoint. Next come model packages per
sector (SMEs, trust companies, medical practices, public offices), training on a
company's own names and dialect, and a small local detector trained on Apertus's
decisions. Details in section 8 of the report.

## Run it

You need Docker and an API key for an OpenAI-compatible endpoint serving
Apertus 1.5 (the Hack Apertus endpoint is preset).

```bash
cd track_2b
cp .env.example .env      # then set LLM_API_KEY in .env
make run                  # with an NVIDIA GPU for speech: make run-gpu
```

Open <http://localhost:8080>. The playground shows the three stages side by
side: the original, what the model sees, and the answer with the names put
back. It also has a chat whose vault is stored encrypted.

- **Port 8080 taken?** Set `CLAUSUS_PORT=8180` (or any free port) in `.env`.
- **No `make` (Windows)?** Run `docker compose up --build` inside `track_2b/`.
- **Speed on CPU:** speech recognition runs locally. On my machine a 50 second
  recording plus one slide took about 3.5 minutes end to end on CPU.
- **Health:** `curl localhost:8080/health` reports the model endpoint and the
  speech service.

## API

| Endpoint | What it does |
| --- | --- |
| `POST /redact` | Text in, masked text and placeholder list out |
| `POST /meeting` | Transcript in, protocol with real names out |
| `POST /meeting/upload` | Audio, slides and text in one call |
| `GET /receipts/{id}` | What left Clausus in that run |
| `POST /beta/litellm_basic_guardrail_api` | Use Clausus as a LiteLLM guardrail ([docs](track_2b/docs/litellm.md)) |

## License

All Hack Apertus projects are open source. See the
[Terms & Conditions](https://hackapertus.ch/terms-and-conditions), section 6.
