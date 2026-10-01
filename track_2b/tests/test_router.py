from clausus.router import Routes, route


def test_default_routes_keep_everything_local():
    r = Routes.default()
    assert route("protocol", r, external_configured=True).destination == "local"


def test_external_needs_both_a_rule_and_a_configured_target():
    r = Routes(tasks={"protocol": "external"})
    assert route("protocol", r, external_configured=True).destination == "external"
    d = route("protocol", r, external_configured=False)
    assert d.destination == "local"
    assert "not configured" in d.reason


def test_unknown_task_stays_local():
    assert route("translate", Routes(tasks={"protocol": "external"}),
                 external_configured=True).destination == "local"


def test_routes_load_from_yaml(tmp_path):
    p = tmp_path / "routes.yaml"
    p.write_text("tasks:\n  protocol: external\n", encoding="utf-8")
    assert Routes.load(p).tasks == {"protocol": "external"}


def test_invalid_destination_in_yaml_is_rejected(tmp_path):
    p = tmp_path / "routes.yaml"
    p.write_text("tasks:\n  protocol: everywhere\n", encoding="utf-8")
    try:
        Routes.load(p)
    except ValueError as e:
        assert "everywhere" in str(e)
    else:
        raise AssertionError("invalid destination accepted")
