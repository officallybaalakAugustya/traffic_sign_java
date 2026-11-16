package com.trafficsign.data;

import java.util.Optional;

/**
 * Universal interface for traffic sign taxonomies across different national standards.
 * Decouples model architectures and data pipelines from any single country's class definition.
 */
public interface TrafficSignTaxonomy {

    /**
     * Total number of distinct sign classes defined in this taxonomy.
     */
    int totalClasses();

    /**
     * Standard human-readable canonical name for a given class ID.
     */
    String getCanonicalName(int classId);

    /**
     * Functional category (MANDATORY, CAUTIONARY, INFORMATORY, OTHER).
     */
    String getCategory(int classId);

    /**
     * Legal speed limit in km/h if this class is a speed limit sign, or empty.
     */
    Optional<Integer> getSpeedLimitKmH(int classId);

    /**
     * Maps an external or raw dataset label string to a stable integer class ID.
     * Returns empty if the label is unmapped/unknown.
     */
    Optional<Integer> findClassIdByLabel(String label);

    /**
     * Checks if the given class ID is valid within this taxonomy.
     */
    default boolean isValidClassId(int classId) {
        return classId >= 0 && classId < totalClasses();
    }
}
