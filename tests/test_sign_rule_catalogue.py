from trafficsign.safety.catalogue import EventType, RuleCategory, SeverityLevel, SignRuleCatalogue


def test_sign_rule_catalogue():
    catalogue = SignRuleCatalogue()
    assert catalogue.total_registered_rules() >= 66

    # Test Stop sign (class 59)
    rule_stop = catalogue.get_rule(59)
    assert rule_stop.event_type == EventType.STOP_MANDATE
    assert rule_stop.severity == SeverityLevel.CRITICAL
    assert rule_stop.is_regulatory()

    # Test Speed limit 40 (class 51)
    rule_speed40 = catalogue.get_rule(51)
    assert rule_speed40.is_speed_limit()
    assert rule_speed40.numerical_value == 40.0
    assert "40 km/h" in rule_speed40.driver_advisory
