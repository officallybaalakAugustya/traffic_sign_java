package com.trafficsign.ui;

import com.trafficsign.safety.SafetyEvent;
import com.trafficsign.safety.SignRuleCatalogue;

import javax.swing.table.AbstractTableModel;
import java.util.ArrayList;
import java.util.List;

public class SafetyEventTableModel extends AbstractTableModel {

    private static final String[] COLUMN_NAMES = {
            "Event ID", "Frame", "Time (s)", "Severity", "Category", "Sign Class", "Confidence", "Driver Advisory", "Review"
    };

    private final List<SafetyEvent> allEvents = new ArrayList<>();
    private final List<SafetyEvent> displayedEvents = new ArrayList<>();
    private SignRuleCatalogue.SeverityLevel severityFilter = null;
    private SignRuleCatalogue.RuleCategory categoryFilter = null;

    @Override
    public int getRowCount() {
        return displayedEvents.size();
    }

    @Override
    public int getColumnCount() {
        return COLUMN_NAMES.length;
    }

    @Override
    public String getColumnName(int column) {
        return COLUMN_NAMES[column];
    }

    @Override
    public Object getValueAt(int rowIndex, int columnIndex) {
        if (rowIndex < 0 || rowIndex >= displayedEvents.size()) {
            return null;
        }
        SafetyEvent evt = displayedEvents.get(rowIndex);
        return switch (columnIndex) {
            case 0 -> evt.eventId();
            case 1 -> evt.frameIndex();
            case 2 -> String.format("%.2f", evt.timestampMs() / 1000.0);
            case 3 -> evt.severity() != null ? evt.severity().name() : "";
            case 4 -> evt.category() != null ? evt.category().name() : "";
            case 5 -> evt.displayName();
            case 6 -> String.format("%.1f%%", evt.classificationConfidence() * 100.0f);
            case 7 -> evt.driverAdvisory();
            case 8 -> evt.operatorStatus() != null ? evt.operatorStatus().name() : "";
            default -> "";
        };
    }

    public synchronized void addEvent(SafetyEvent event) {
        allEvents.add(event);
        if (matchesFilter(event)) {
            displayedEvents.add(event);
            int row = displayedEvents.size() - 1;
            fireTableRowsInserted(row, row);
        }
    }

    public synchronized void setEvents(List<SafetyEvent> events) {
        allEvents.clear();
        if (events != null) {
            allEvents.addAll(events);
        }
        applyFilters();
    }

    public synchronized SafetyEvent getEventAt(int row) {
        if (row >= 0 && row < displayedEvents.size()) {
            return displayedEvents.get(row);
        }
        return null;
    }

    public synchronized void setSeverityFilter(SignRuleCatalogue.SeverityLevel filter) {
        this.severityFilter = filter;
        applyFilters();
    }

    public synchronized void setCategoryFilter(SignRuleCatalogue.RuleCategory filter) {
        this.categoryFilter = filter;
        applyFilters();
    }

    public synchronized void clear() {
        allEvents.clear();
        displayedEvents.clear();
        fireTableDataChanged();
    }

    public synchronized void updateEvent(SafetyEvent updated) {
        for (int i = 0; i < allEvents.size(); i++) {
            if (allEvents.get(i).eventId().equals(updated.eventId())) {
                allEvents.set(i, updated);
                break;
            }
        }
        for (int i = 0; i < displayedEvents.size(); i++) {
            if (displayedEvents.get(i).eventId().equals(updated.eventId())) {
                displayedEvents.set(i, updated);
                fireTableRowsUpdated(i, i);
                break;
            }
        }
    }

    private boolean matchesFilter(SafetyEvent evt) {
        if (severityFilter != null && evt.severity() != severityFilter) {
            return false;
        }
        if (categoryFilter != null && evt.category() != categoryFilter) {
            return false;
        }
        return true;
    }

    private synchronized void applyFilters() {
        displayedEvents.clear();
        for (SafetyEvent evt : allEvents) {
            if (matchesFilter(evt)) {
                displayedEvents.add(evt);
            }
        }
        fireTableDataChanged();
    }
}
