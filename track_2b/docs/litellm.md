# Clausus behind LiteLLM

Clausus implements LiteLLM's Generic Guardrail API (`POST /beta/litellm_basic_guardrail_api`).
LiteLLM calls it before every model call (`pre_call`) and after it (`post_call`):

- **pre_call:** Clausus finds the confidential parts of the prompt, replaces them with placeholders
  (`[PERSON_1]`, `[ORG_1]`, …) and keeps the mapping in memory under LiteLLM's `litellm_call_id`.
  The model only sees placeholders.
- **post_call:** Clausus puts the originals back into the answer and deletes the mapping at once.
  Unused mappings expire after 10 minutes. Nothing is written to disk except a receipt without originals.
- If Clausus cannot check the content (Apertus unreachable, rate limited), it answers `BLOCKED`.
  Images are blocked for now.

## LiteLLM config

```yaml
guardrails:
  - guardrail_name: clausus
    litellm_params:
      guardrail: generic_guardrail_api
      mode: [pre_call, post_call]
      api_base: http://clausus:8080
      api_key: os.environ/CLAUSUS_GUARDRAIL_KEY   # sent as the x-api-key header
      unreachable_fallback: fail_closed
      default_on: true
```

Set the same `CLAUSUS_GUARDRAIL_KEY` in Clausus' `.env`. Without it, Clausus accepts every caller,
so only leave it empty when Clausus is not reachable from outside.

## Verified end to end (02.10.2026, LiteLLM 1.83.9)

With a LiteLLM proxy in front of a stand-in model that logs what it receives:

- The model received `[PERSON_1] von der [ORG_1] will das [OTHER_1] bis [DATE_1] abschliessen,
  Budget [AMOUNT_1]. Rückruf an [PHONE_1].`
- The caller got the original names back in the answer.
- With Clausus stopped, LiteLLM answered HTTP 500 and the model received nothing.

Note: `guardrails:` sits at the top level of the config. Under `litellm_settings:` LiteLLM reads
it in its legacy format and refuses to start.
