import pytest

from clausus.config import DEFAULT_MODEL, ConfigError, Settings

VALID = {
    "LLM_BASE_URL": "https://example.test/v1/",
    "LLM_API_KEY": "sk-secret-123",
    "LLM_NAME": "apertus-v1.5-8b",
}


def test_reads_all_three_variables_and_strips_trailing_slash():
    s = Settings.from_env(VALID)
    assert s.llm_base_url == "https://example.test/v1"
    assert s.llm_api_key == "sk-secret-123"
    assert s.llm_name == "apertus-v1.5-8b"


def test_model_defaults_to_apertus_70b_when_not_set():
    env = {k: v for k, v in VALID.items() if k != "LLM_NAME"}
    assert Settings.from_env(env).llm_name == DEFAULT_MODEL == "apertus-v1.5-70b"


@pytest.mark.parametrize("missing", ["LLM_BASE_URL", "LLM_API_KEY"])
def test_missing_required_variable_names_it_and_points_to_env_example(missing):
    env = dict(VALID)
    env[missing] = "   "
    with pytest.raises(ConfigError) as e:
        Settings.from_env(env)
    assert missing in str(e.value)
    assert ".env.example" in str(e.value)


def test_both_missing_are_reported_together():
    with pytest.raises(ConfigError) as e:
        Settings.from_env({})
    assert "LLM_BASE_URL" in str(e.value) and "LLM_API_KEY" in str(e.value)


def test_base_url_without_scheme_is_rejected():
    with pytest.raises(ConfigError) as e:
        Settings.from_env({**VALID, "LLM_BASE_URL": "example.test/v1"})
    assert "http" in str(e.value)


def test_api_key_never_appears_in_repr_or_str():
    s = Settings.from_env(VALID)
    assert "sk-secret-123" not in repr(s)
    assert "sk-secret-123" not in str(s)
