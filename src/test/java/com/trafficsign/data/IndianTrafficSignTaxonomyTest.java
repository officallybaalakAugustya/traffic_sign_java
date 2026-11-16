package com.trafficsign.data;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;

class IndianTrafficSignTaxonomyTest {

    private IndianTrafficSignTaxonomy taxonomy;

    @BeforeEach
    void setUp() {
        taxonomy = new IndianTrafficSignTaxonomy();
    }

    @Test
    @DisplayName("Should contain at least 66 defined IRC:67 classes with valid metadata")
    void shouldHaveValidIndianClasses() {
        assertThat(taxonomy.totalClasses()).isGreaterThanOrEqualTo(66);

        // Verify key Indian IRC:67 specific signs
        Optional<Integer> cattleId = taxonomy.findClassIdByLabel("CATTLE");
        assertThat(cattleId).isPresent();
        assertThat(taxonomy.getCategory(cattleId.get())).isEqualTo("CAUTIONARY");

        Optional<Integer> speedBreakerId = taxonomy.findClassIdByLabel("HUMP_OR_ROUGH_ROAD");
        assertThat(speedBreakerId).isPresent();
        assertThat(taxonomy.getCategory(speedBreakerId.get())).isEqualTo("CAUTIONARY");

        Optional<Integer> hornProhibitedId = taxonomy.findClassIdByLabel("HORN_PROHIBITED");
        assertThat(hornProhibitedId).isPresent();
        assertThat(taxonomy.getCategory(hornProhibitedId.get())).isEqualTo("MANDATORY");

        Optional<Integer> compHornId = taxonomy.findClassIdByLabel("COMPULSARY_SOUND_HORN");
        assertThat(compHornId).isPresent();
        assertThat(taxonomy.getCategory(compHornId.get())).isEqualTo("MANDATORY");
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "ALL_MOTOR_VEHICLE_PROHIBITED",
            "0ALL_MOTOR_VEHICLE_PROHIBITED",
            "all_motor_vehicle_prohibited",
            "All Motor Vehicles Prohibited"
    })
    @DisplayName("Should normalize and resolve various raw label formats to the same class ID")
    void shouldResolveVariousLabelFormats(String label) {
        Optional<Integer> id = taxonomy.findClassIdByLabel(label);
        assertThat(id).isPresent().contains(0);
    }

    @Test
    @DisplayName("Should safely return empty optional for unmapped or unknown sign labels")
    void shouldHandleUnknownLabelsGracefully() {
        assertThat(taxonomy.findClassIdByLabel("NON_EXISTENT_INDIAN_SIGN_XYZ")).isEmpty();
        assertThat(taxonomy.findClassIdByLabel("")).isEmpty();
        assertThat(taxonomy.findClassIdByLabel(null)).isEmpty();
    }

    @Test
    @DisplayName("Should correctly extract speed limit values for Indian regulatory speed signs")
    void shouldExtractSpeedLimits() {
        Optional<Integer> limit40Id = taxonomy.findClassIdByLabel("SPEED_LIMIT_40");
        assertThat(limit40Id).isPresent();
        assertThat(taxonomy.getSpeedLimitKmH(limit40Id.get())).contains(40);

        Optional<Integer> stopId = taxonomy.findClassIdByLabel("STOP");
        assertThat(stopId).isPresent();
        assertThat(taxonomy.getSpeedLimitKmH(stopId.get())).isEmpty();
    }

    @Test
    @DisplayName("Should provide actionable road safety alerts for recognized Indian signs")
    void shouldProvideSafetyAlerts() {
        Optional<Integer> stopId = taxonomy.findClassIdByLabel("STOP");
        assertThat(stopId).isPresent();
        Optional<String> alert = taxonomy.getSafetyAlert(stopId.get());
        assertThat(alert).isPresent();
        assertThat(alert.get().toLowerCase()).containsAnyOf("stop", "halt");
    }
}
