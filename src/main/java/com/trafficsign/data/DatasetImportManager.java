package com.trafficsign.data;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import javax.imageio.ImageIO;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

public final class DatasetImportManager {
    public static final long MAX_ARCHIVE_BYTES = 2_000_000_000L;
    public static final long MAX_UNCOMPRESSED_BYTES = 10_000_000_000L;
    public static final long MAX_ENTRY_BYTES = 512_000_000L;
    public static final long MAX_ENTRIES = 100_000L;
    public static final long MAX_COMPRESSION_RATIO = 100L;
    private static final Set<String> IMAGE_EXTENSIONS = Set.of(".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".gif");
    private static final ObjectMapper MAPPER = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

    public record Entry(String archive, String path, long size, String sha256, String type, String action, String reason) {
    }

    public record Archive(String path, long size, String modified, String sha256, String classification,
                          boolean readable, boolean reviewRequired, String reason, int entries,
                          long uncompressedBytes, List<Entry> files) {
    }

    public record Report(String generatedAt, String workspace, String rawDirectory, List<Archive> archives,
                         long proposedImports, long exactDuplicates, long bytesProposed, String markdown) {
    }

    public record ImportResult(String generatedAt, long imported, long skipped, List<String> failures, String markdown) {
    }

    private record Limits(long archiveBytes, long totalBytes, long entryBytes, long entries, long ratio) {
    }

    private DatasetImportManager() {
    }

    public static Report audit(Path workspace, Path input) throws IOException {
        Path root = workspace.toAbsolutePath().normalize();
        Path raw = root.resolve("data").resolve("raw").normalize();
        Map<String, List<String>> existing = indexExisting(raw);
        List<Path> archives = discover(input);
        List<Archive> results = new ArrayList<>();
        Set<String> seenIncoming = new HashSet<>();
        for (Path archive : archives) {
            results.add(inspectArchive(archive, raw, existing, seenIncoming));
        }
        long proposed = results.stream().flatMap(a -> a.files().stream()).filter(e -> "IMPORT".equals(e.action())).count();
        long duplicates = results.stream().flatMap(a -> a.files().stream()).filter(e -> "SKIP_DUPLICATE".equals(e.action())).count();
        long bytes = results.stream().flatMap(a -> a.files().stream()).filter(e -> "IMPORT".equals(e.action())).mapToLong(Entry::size).sum();
        String markdown = markdown(root, raw, results, proposed, duplicates, bytes);
        return new Report(Instant.now().toString(), root.toString(), raw.toString(), List.copyOf(results), proposed, duplicates, bytes, markdown);
    }

    public static void writeReports(Path workspace, Report report, boolean includePlan) throws IOException {
        Path dir = workspace.toAbsolutePath().normalize().resolve("reports").resolve("dataset_imports");
        Files.createDirectories(dir);
        String stamp = DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss'Z'").withZone(java.time.ZoneOffset.UTC).format(Instant.now());
        Files.writeString(dir.resolve("dataset_import_" + stamp + ".md"), report.markdown(), StandardCharsets.UTF_8);
        Files.writeString(dir.resolve("dataset_import_" + stamp + ".json"), MAPPER.writeValueAsString(report), StandardCharsets.UTF_8);
        if (includePlan) {
            Files.writeString(dir.resolve("dataset_import_plan.json"), MAPPER.writeValueAsString(report), StandardCharsets.UTF_8);
        }
        System.out.println("Reports written under " + dir);
    }

    public static ImportResult importPlan(Path workspace, Path planPath) throws IOException {
        Path root = workspace.toAbsolutePath().normalize();
        Report report = MAPPER.readValue(planPath.toFile(), Report.class);
        Path raw = root.resolve("data").resolve("raw").normalize();
        Path staging = Files.createTempDirectory(root, ".dataset-import-");
        long imported = 0;
        long skipped = 0;
        List<String> failures = new ArrayList<>();
        try {
            for (Archive archive : report.archives()) {
                for (Entry entry : archive.files()) {
                    if (!"IMPORT".equals(entry.action())) {
                        skipped++;
                        continue;
                    }
                    Path source = Path.of(archive.path());
                    Path staged = stagedPath(staging, archive, entry);
                    try (ZipFile zip = new ZipFile(source.toFile())) {
                        ZipEntry ze = zip.getEntry(entry.path());
                        if (ze == null) throw new IOException("Entry missing from archive: " + entry.path());
                        Files.createDirectories(staged.getParent());
                        try (InputStream in = zip.getInputStream(ze)) {
                            Files.copy(in, staged, StandardCopyOption.REPLACE_EXISTING);
                        }
                        if (!entry.sha256().equals(DatasetIntegrityChecker.computeSha256(staged))) {
                            throw new IOException("Staged hash mismatch: " + entry.path());
                        }
                    }
                }
            }
            for (Archive archive : report.archives()) {
                for (Entry entry : archive.files()) {
                    if (!"IMPORT".equals(entry.action())) continue;
                    Path destination = safeResolve(raw.resolve("imported").resolve(Path.of(archive.path()).getFileName().toString()), entry.path());
                    Path staged = stagedPath(staging, archive, entry);
                    if (Files.exists(destination)) {
                        if (entry.sha256().equals(DatasetIntegrityChecker.computeSha256(destination))) skipped++;
                        else failures.add("Collision preserved: " + destination);
                        continue;
                    }
                    Files.createDirectories(destination.getParent());
                    try {
                        Files.move(staged, destination, StandardCopyOption.ATOMIC_MOVE);
                    } catch (java.nio.file.AtomicMoveNotSupportedException e) {
                        Files.move(staged, destination);
                    }
                    if (!entry.sha256().equals(DatasetIntegrityChecker.computeSha256(destination))) {
                        throw new IOException("Promoted hash mismatch: " + destination);
                    }
                    imported++;
                }
            }
        } catch (Exception e) {
            failures.add(e.getMessage());
        } finally {
            deleteTree(staging);
        }
        String text = "# Dataset Import Result\n\nImported: " + imported + "\nSkipped: " + skipped + "\nFailures: " + failures.size() + "\n" +
                String.join("\n", failures.stream().map(f -> "- " + f).toList());
        return new ImportResult(Instant.now().toString(), imported, skipped, List.copyOf(failures), text);
    }

    private static Archive inspectArchive(Path path, Path raw, Map<String, List<String>> existing, Set<String> seenIncoming) {
        String absolute = path.toAbsolutePath().normalize().toString();
        try {
            long size = Files.size(path);
            String hash = DatasetIntegrityChecker.computeSha256(path);
            if (!path.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".zip") || path.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".opdownload")) {
                return new Archive(absolute, size, Files.getLastModifiedTime(path).toString(), hash, "REJECTED", false, false, "Incomplete download or unsupported archive suffix", 0, 0, List.of());
            }
            if (size > MAX_ARCHIVE_BYTES) return rejected(path, size, hash, "Archive exceeds size limit");
            List<Entry> entries = new ArrayList<>();
            Set<String> names = new HashSet<>();
            long total = 0;
            boolean review = false;
            try (ZipFile zip = new ZipFile(path.toFile())) {
                if (zip.size() > MAX_ENTRIES) return rejected(path, size, hash, "Entry count exceeds limit");
                Enumeration<? extends ZipEntry> en = zip.entries();
                while (en.hasMoreElements()) {
                    ZipEntry ze = en.nextElement();
                    String normalized = normalize(ze.getName());
                    if (!names.add(normalized)) throw new IOException("Duplicate normalized entry path: " + normalized);
                    if (ze.isDirectory()) continue;
                    if (ze.getSize() < 0 || ze.getSize() > MAX_ENTRY_BYTES) throw new IOException("Entry exceeds size limit: " + normalized);
                    total = Math.addExact(total, ze.getSize());
                    if (total > MAX_UNCOMPRESSED_BYTES) throw new IOException("Archive exceeds total expansion limit");
                    if (ze.getCompressedSize() > 0 && ze.getSize() / ze.getCompressedSize() > MAX_COMPRESSION_RATIO) throw new IOException("Compression ratio exceeds limit: " + normalized);
                    byte[] bytes = read(zip.getInputStream(ze), ze.getSize());
                    String entryHash = sha256(bytes);
                    String type = type(normalized);
                    String action = "IMPORT";
                    String reason = "Validated";
                    if (!seenIncoming.add(entryHash)) { action = "SKIP_DUPLICATE"; reason = "Duplicate content in selected archives"; }
                    else if (existing.containsKey(entryHash)) { action = "SKIP_DUPLICATE"; reason = "Exact duplicate already in data/raw: " + existing.get(entryHash).get(0); }
                    if ("IMAGE".equals(type)) {
                        try (ByteArrayInputStream in = new ByteArrayInputStream(bytes)) { if (ImageIO.read(in) == null) throw new IOException("Unreadable image"); }
                    }
                    if ("ANNOTATION".equals(type)) { validateAnnotation(bytes); review = true; reason = "Unknown class mapping/license; review required"; }
                    entries.add(new Entry(absolute, normalized, ze.getSize(), entryHash, type, action, reason));
                }
            }
            String lower = path.getFileName().toString().toLowerCase(Locale.ROOT);
            if (review) {
                entries = entries.stream().map(e -> "IMPORT".equals(e.action())
                        ? new Entry(e.archive(), e.path(), e.size(), e.sha256(), e.type(), "REVIEW", "Archive requires license/class-mapping review")
                        : e).toList();
            }
            String classification = review ? "NEEDS_REVIEW" : "APPROVED";
            if (entries.stream().anyMatch(e -> "SKIP_DUPLICATE".equals(e.action()))) classification = "APPROVED_WITH_EXCLUSIONS";
            return new Archive(absolute, size, Files.getLastModifiedTime(path).toString(), hash, classification, true, review, review ? "License and class mapping require review" : "", entries.size(), total, List.copyOf(entries));
        } catch (Exception e) {
            try { return rejected(path, Files.size(path), DatasetIntegrityChecker.computeSha256(path), e.getMessage()); }
            catch (IOException ignored) { return new Archive(absolute, 0, "", "", "REJECTED", false, false, e.getMessage(), 0, 0, List.of()); }
        }
    }

    private static Archive rejected(Path p, long size, String hash, String reason) throws IOException {
        return new Archive(p.toAbsolutePath().normalize().toString(), size, Files.getLastModifiedTime(p).toString(), hash, "REJECTED", false, false, reason, 0, 0, List.of());
    }

    private static Map<String, List<String>> indexExisting(Path raw) throws IOException {
        Map<String, List<String>> map = new HashMap<>();
        if (!Files.isDirectory(raw)) return map;
        try (var stream = Files.walk(raw)) {
            for (Path file : stream.filter(Files::isRegularFile).sorted().toList()) map.computeIfAbsent(DatasetIntegrityChecker.computeSha256(file), k -> new ArrayList<>()).add(file.toString());
        }
        return map;
    }

    private static List<Path> discover(Path input) throws IOException {
        if (Files.isRegularFile(input)) return List.of(input.toAbsolutePath().normalize());
        if (!Files.isDirectory(input)) throw new IOException("Input path does not exist: " + input);
        try (var stream = Files.walk(input)) {
            return stream.filter(Files::isRegularFile).filter(p -> p.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".zip") || p.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".zip.opdownload")).sorted().toList();
        }
    }

    private static String normalize(String name) throws IOException {
        String n = name.replace('\\', '/');
        if (n.startsWith("/") || n.matches("^[A-Za-z]:.*") || n.contains("//")) throw new IOException("Unsafe ZIP path: " + name);
        Path p = Path.of(n).normalize();
        if (p.isAbsolute() || p.startsWith("..") || n.matches("(^|/)\\.\\.(/|$)")) throw new IOException("Path traversal in ZIP: " + name);
        for (String part : n.split("/")) if (part.matches("(?i)CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]")) throw new IOException("Reserved device name in ZIP path: " + name);
        return p.toString().replace('\\', '/');
    }

    private static String type(String name) {
        String lower = name.toLowerCase(Locale.ROOT);
        if (IMAGE_EXTENSIONS.stream().anyMatch(lower::endsWith)) return "IMAGE";
        if (lower.endsWith(".txt") || lower.endsWith(".csv") || lower.endsWith(".json") || lower.endsWith(".xml")) return "ANNOTATION";
        return "OTHER";
    }

    private static byte[] read(InputStream in, long expected) throws IOException {
        try (in) {
            byte[] data = in.readAllBytes();
            if (data.length != expected) throw new IOException("Entry size mismatch");
            return data;
        }
    }

    private static void validateAnnotation(byte[] bytes) throws IOException {
        String text = new String(bytes, StandardCharsets.UTF_8);
        for (String line : text.split("\\R")) {
            if (line.isBlank()) continue;
            String[] parts = line.trim().split("\\s+");
            if (parts.length != 5 || !parts[0].matches("\\d+")) throw new IOException("Invalid annotation row");
            for (int i = 1; i < 5; i++) { double v = Double.parseDouble(parts[i]); if (!(v > 0 && v <= 1)) throw new IOException("Invalid normalized annotation coordinate"); }
        }
    }

    private static String sha256(byte[] data) throws IOException {
        try { byte[] digest = MessageDigest.getInstance("SHA-256").digest(data); StringBuilder s = new StringBuilder(); for (byte b : digest) s.append(String.format("%02x", b)); return s.toString(); }
        catch (Exception e) { throw new IOException(e); }
    }

    private static Path safeResolve(Path root, String relative) throws IOException {
        Path result = root.resolve(relative).normalize();
        if (!result.startsWith(root.toAbsolutePath().normalize())) throw new IOException("Resolved path escapes root: " + relative);
        return result;
    }

    private static Path stagedPath(Path staging, Archive archive, Entry entry) throws IOException {
        String archiveKey = sha256(archive.path().getBytes(StandardCharsets.UTF_8));
        return safeResolve(staging.resolve(archiveKey), entry.path());
    }

    private static void deleteTree(Path root) throws IOException {
        if (root == null || !Files.exists(root)) return;
        try (var stream = Files.walk(root)) { for (Path p : stream.sorted(Comparator.reverseOrder()).toList()) Files.deleteIfExists(p); }
    }

    private static String markdown(Path root, Path raw, List<Archive> archives, long proposed, long duplicates, long bytes) {
        StringBuilder s = new StringBuilder("# Dataset Import Dry-Run\n\n");
        s.append("Workspace: `").append(root).append("`\n\nRaw directory: `").append(raw).append("`\n\n");
        s.append("Proposed imports: ").append(proposed).append("\n\nExact duplicates skipped: ").append(duplicates).append("\n\nBytes proposed: ").append(bytes).append("\n\n");
        for (Archive a : archives) { s.append("## ").append(a.path()).append("\n\nClassification: `").append(a.classification()).append("`\n\n"); if (!a.reason().isBlank()) s.append(a.reason()).append("\n\n"); s.append("Entries: ").append(a.entries()).append("; uncompressed bytes: ").append(a.uncompressedBytes()).append("\n\n"); }
        s.append("No training splits, held-out test data, manifests, checkpoints, or existing files are modified by audit/plan.\n");
        return s.toString();
    }
}
