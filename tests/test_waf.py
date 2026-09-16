from http_attack_agent.waf.audit import event_concepts, parse_alert_line


def test_parse_modsecurity_alert():
    line = (
        'ModSecurity: Warning. Matched "Operator" against variable `ARGS:user\' '
        '[id "942100"] [msg "SQL Injection Attack"] '
        '[data "Matched Data: union select"] [severity "CRITICAL"] '
        '[tag "attack-sqli"] [tag "OWASP_CRS/ATTACK-SQLI"] '
        '[uri "/login"] [unique_id "abc123"]'
    )
    event = parse_alert_line(line)
    assert event is not None
    assert event.rule_id == "942100"
    assert event.request_id is None
    assert event.transaction_id == "abc123"
    assert event.matched_variable == "ARGS:user"
    matchers = {
        "waf_sqli": {
            "tags": {"attack-sqli"},
            "rule_ids": set(),
            "variable_prefixes": set(),
        },
        "waf_args": {
            "tags": set(),
            "rule_ids": set(),
            "variable_prefixes": {"ARGS"},
        },
    }
    assert event_concepts(event, matchers) == ["waf_args", "waf_sqli"]
