package com.trafficsign.safety;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class SignRuleCatalogueTest {

    private SignRuleCatalogue catalogue;

    @BeforeEach
    void setUp() {
        catalogue = new SignRuleCatalogue();
    }

    @Test
    void testAll85ClassesAreRegistered() {
        assertThat(catalogue.totalRegisteredRules()).isEqualTo(85);
        for (int i = 0; i < 85; i++) {
            SignRuleCatalogue.SignRule rule = catalogue.getRule(i);
            assertThat(rule).isNotNull();
            assertThat(rule.classId()).isEqualTo(i);
            assertThat(rule.canonicalLabel()).isNotEmpty();
            assertThat(rule.displayName()).isNotEmpty();
            assertThat(rule.driverAdvisory()).isNotEmpty();
            assertThat(rule.category()).isNotEqualTo(SignRuleCatalogue.RuleCategory.UNKNOWN);
            assertThat(rule.verificationStatus()).isEqualTo("IRC:67-2012 Verified");
        }
    }

    @Test
    void testSpeedLimitsHaveCorrectValues() {
        SignRuleCatalogue.SignRule sl40 = catalogue.getRule(51);
        assertThat(sl40.canonicalLabel()).isEqualTo("SPEED_LIMIT_40");
        assertThat(sl40.numericalValue()).isEqualTo(40.0);
        assertThat(sl40.unit()).isEqualTo("km/h");
        assertThat(sl40.isSpeedLimit()).isTrue();

        SignRuleCatalogue.SignRule sl80 = catalogue.getRule(55);
        assertThat(sl80.canonicalLabel()).isEqualTo("SPEED_LIMIT_80");
        assertThat(sl80.numericalValue()).isEqualTo(80.0);

        SignRuleCatalogue.SignRule sl20 = catalogue.getRule(80);
        assertThat(sl20.canonicalLabel()).isEqualTo("SPEED_LIMIT_20");
        assertThat(sl20.numericalValue()).isEqualTo(20.0);
    }

    @Test
    void testCriticalRules() {
        SignRuleCatalogue.SignRule stop = catalogue.getRule(59);
        assertThat(stop.canonicalLabel()).isEqualTo("STOP");
        assertThat(stop.isCritical()).isTrue();
        assertThat(stop.category()).isEqualTo(SignRuleCatalogue.RuleCategory.REGULATORY);
        assertThat(stop.eventType()).isEqualTo(SignRuleCatalogue.EventType.STOP_MANDATE);

        SignRuleCatalogue.SignRule noEntry = catalogue.getRule(31);
        assertThat(noEntry.canonicalLabel()).isEqualTo("NO_ENTRY");
        assertThat(noEntry.isCritical()).isTrue();

        SignRuleCatalogue.SignRule overtaking = catalogue.getRule(34);
        assertThat(overtaking.canonicalLabel()).isEqualTo("OVERTAKING_PROHIBITED");
        assertThat(overtaking.isCritical()).isTrue();
    }

    @Test
    void testCautionaryRules() {
        SignRuleCatalogue.SignRule school = catalogue.getRule(44);
        assertThat(school.canonicalLabel()).isEqualTo("SCHOOL_AHEAD");
        assertThat(school.category()).isEqualTo(SignRuleCatalogue.RuleCategory.CAUTIONARY);
        assertThat(school.eventType()).isEqualTo(SignRuleCatalogue.EventType.HAZARD_WARNING);

        SignRuleCatalogue.SignRule ped = catalogue.getRule(36);
        assertThat(ped.canonicalLabel()).isEqualTo("PEDESTRIAN_CROSSING");
        assertThat(ped.category()).isEqualTo(SignRuleCatalogue.RuleCategory.CAUTIONARY);
    }

    @Test
    void testUnknownFallback() {
        SignRuleCatalogue.SignRule ruleNeg = catalogue.getRule(-5);
        assertThat(ruleNeg.classId()).isEqualTo(-1);
        assertThat(ruleNeg.category()).isEqualTo(SignRuleCatalogue.RuleCategory.UNKNOWN);
        assertThat(ruleNeg.eventType()).isEqualTo(SignRuleCatalogue.EventType.UNCERTAIN_DETECTION);
        assertThat(ruleNeg.verificationStatus()).isEqualTo("Unverified Fallback");

        SignRuleCatalogue.SignRule ruleHigh = catalogue.getRule(999);
        assertThat(ruleHigh.classId()).isEqualTo(-1);

        SignRuleCatalogue.SignRule ruleNullStr = catalogue.getRuleByLabel(null);
        assertThat(ruleNullStr.classId()).isEqualTo(-1);

        SignRuleCatalogue.SignRule ruleUnknownStr = catalogue.getRuleByLabel("NON_EXISTENT_SIGN");
        assertThat(ruleUnknownStr.classId()).isEqualTo(-1);
    }

    @Test
    void testFindRuleByLabel() {
        SignRuleCatalogue.SignRule stopByLabel = catalogue.getRuleByLabel("STOP");
        assertThat(stopByLabel.classId()).isEqualTo(59);

        SignRuleCatalogue.SignRule sl50 = catalogue.getRuleByLabel("speed_limit_50");
        assertThat(sl50.classId()).isEqualTo(52);
    }
}
