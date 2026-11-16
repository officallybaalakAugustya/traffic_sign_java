package com.trafficsign.data;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.HashSet;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class TrafficSignClassTest {

    @Test
    @DisplayName("Should verify that exactly 43 standard GTSRB classes are defined with unique IDs 0 to 42")
    void shouldHave43ContinuousUniqueClasses() {
        TrafficSignClass[] classes = TrafficSignClass.values();
        assertThat(classes).hasSize(43);

        Set<Integer> seenIds = new HashSet<>();
        for (int i = 0; i < 43; i++) {
            TrafficSignClass signClass = TrafficSignClass.requireById(i);
            assertThat(signClass.getId()).isEqualTo(i);
            assertThat(signClass.getLabelName()).isNotBlank();
            assertThat(signClass.getCategory()).isNotNull();
            seenIds.add(signClass.getId());
        }
        assertThat(seenIds).hasSize(43);
    }

    @ParameterizedTest
    @ValueSource(ints = {-1, 43, 99, -100})
    @DisplayName("Should throw IllegalArgumentException when class ID is out of range")
    void shouldThrowOnInvalidId(int invalidId) {
        assertThatThrownBy(() -> TrafficSignClass.requireById(invalidId))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Valid range is 0 to 42");
    }

    @Test
    @DisplayName("Should correctly identify regulatory speed limits and categories")
    void shouldVerifySpeedLimitsAndCategories() {
        TrafficSignClass speed50 = TrafficSignClass.requireById(2);
        assertThat(speed50.getLabelName()).isEqualTo("Speed limit (50km/h)");
        assertThat(speed50.getCategory()).isEqualTo(TrafficSignClass.SignCategory.PROHIBITORY);
        assertThat(speed50.getSpeedLimitKmH()).contains(50);

        TrafficSignClass stop = TrafficSignClass.requireById(14);
        assertThat(stop.getLabelName()).isEqualTo("Stop");
        assertThat(stop.getCategory()).isEqualTo(TrafficSignClass.SignCategory.PROHIBITORY);
        assertThat(stop.getSpeedLimitKmH()).isEmpty();

        TrafficSignClass yield = TrafficSignClass.requireById(13);
        assertThat(yield.getCategory()).isEqualTo(TrafficSignClass.SignCategory.DANGER);
    }
}
