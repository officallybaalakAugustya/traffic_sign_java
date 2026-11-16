package com.trafficsign;

import com.trafficsign.app.Main;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class AppTest {

    @Test
    @DisplayName("Stage 0 Smoke Test: Verify application metadata and Java version")
    void shouldVerifyApplicationMetadata() {
        assertThat(Main.APP_NAME).isEqualTo("Traffic Sign Recognition and Road Safety Monitor");
        assertThat(Main.APP_VERSION).isEqualTo("1.0.0-SNAPSHOT");

        int javaMajorVersion = Runtime.version().feature();
        assertThat(javaMajorVersion)
                .as("Runtime Java major version must be at least Java 21")
                .isGreaterThanOrEqualTo(21);
    }
}
