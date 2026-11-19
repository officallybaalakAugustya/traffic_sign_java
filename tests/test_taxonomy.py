from trafficsign.data.taxonomy import IndianTrafficSignTaxonomy


def test_taxonomy_loading_and_retrieval():
    tax = IndianTrafficSignTaxonomy()
    assert tax.get_total_classes() >= 66

    # Test class 0: All Motor Vehicles Prohibited
    c0 = tax.get_class_info(0)
    assert c0 is not None
    assert "Motor" in c0.canonical_name

    # Test speed limits
    c_speed40 = tax.get_class_by_label("SPEED_LIMIT_40")
    assert c_speed40 is not None
    assert c_speed40.speed_limit_kmh == 40
