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
litellm_settings:
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
