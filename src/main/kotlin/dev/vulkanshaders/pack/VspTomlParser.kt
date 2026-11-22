package dev.vulkanshaders.pack

/**
 * Full TOML parser for `.vsp` `pack.toml` manifests (`VSP Graph v2`).
 * Supports:
 * - `[section]` and `[section.sub]` tables (including quoted segments)
 * - `[[passes]]` array-of-tables (indexed as `passes.0`, `passes.1`, ... and `[passes.0.bindings]`)
 * - Dotted keys (`a.b = 1`) at table scope and inside inline tables
 * - Inline tables `{ k = "v", ... }` (`Value.Tbl`)
 * - Single-line (`"..."` / `'...'`) and multiline (`"""..."""` / `'''...'''`) strings with full escape decoding
 * - Single-line and multiline arrays (`[...]`) with comments and trailing commas
 * - Booleans, integers, floats with `_` digit separators, and `0x`/`0o`/`0b` integer literals.
 */
object VspTomlParser {
    sealed interface Value {
        data class Str(val value: String) : Value
        data class Bool(val value: Boolean) : Value
        data class Num(val value: Double, val isFloat: Boolean) : Value
        data class Arr(val items: List<Value>) : Value
        data class Tbl(val entries: Map<String, Value>) : Value
    }

    fun parse(text: String): Map<String, Map<String, Value>> {
        val root = linkedMapOf<String, MutableMap<String, Value>>()
        var currentSection = ""
        root[currentSection] = linkedMapOf()
        val arrayTableCounters = mutableMapOf<String, Int>()
        var lastPassIndex = -1

        val lines = text.replace("\r\n", "\n").replace("\r", "\n").split('\n')
        var lineIdx = 0
        while (lineIdx < lines.size) {
            val rawLine = lines[lineIdx]
            lineIdx++
            val trimmed = stripLineComment(rawLine).trim()
            if (trimmed.isEmpty()) continue

            if (trimmed.startsWith("[[") && trimmed.endsWith("]]") && trimmed.length > 4) {
                val arrayName = parseKeyPath(trimmed.substring(2, trimmed.length - 2)).joinToString(".")
                if (arrayName.isEmpty()) continue
                val idx = arrayTableCounters.getOrDefault(arrayName, 0)
                arrayTableCounters[arrayName] = idx + 1
                currentSection = "$arrayName.$idx"
                if (arrayName == "passes") {
                    lastPassIndex = idx
                }
                root.getOrPut(currentSection) { linkedMapOf() }
                continue
            }

            if (trimmed.startsWith("[") && trimmed.endsWith("]") && trimmed.length > 2) {
                val rawSec = parseKeyPath(trimmed.substring(1, trimmed.length - 1)).joinToString(".")
                if (rawSec.isEmpty()) continue
                currentSection = if (rawSec == "passes.bindings" && lastPassIndex >= 0) {
                    "passes.$lastPassIndex.bindings"
                } else {
                    rawSec
                }
                root.getOrPut(currentSection) { linkedMapOf() }
                continue
            }

            val eqIdx = findTopLevelChar(rawLine, '=')
            if (eqIdx <= 0) continue

            val keySegments = parseKeyPath(rawLine.substring(0, eqIdx))
            if (keySegments.isEmpty()) continue

            var rawValPart = rawLine.substring(eqIdx + 1)
            val combinedVal = StringBuilder(rawValPart)

            // Accumulate continuation lines while inside multiline string, array, or inline table.
            while (!isValueComplete(combinedVal.toString()) && lineIdx < lines.size) {
                combinedVal.append('\n').append(lines[lineIdx])
                lineIdx++
            }

            val cleanedVal = stripTrailingCommentFromValue(combinedVal.toString()).trim()
            val parsed = parseValue(cleanedVal) ?: continue

            if (keySegments.size == 1) {
                root.getOrPut(currentSection) { linkedMapOf() }[keySegments[0]] = parsed
            } else {
                val prefix = keySegments.dropLast(1).joinToString(".")
                val targetSection = if (currentSection.isEmpty()) prefix else "$currentSection.$prefix"
                val leafKey = keySegments.last()
                root.getOrPut(targetSection) { linkedMapOf() }[leafKey] = parsed
            }
        }
        return root
    }

    private fun parseKeyPath(raw: String): List<String> {
        val segments = mutableListOf<String>()
        val s = raw.trim()
        var i = 0
        val n = s.length
        while (i < n) {
            while (i < n && (s[i] == ' ' || s[i] == '\t')) i++
            if (i >= n) break
            when (s[i]) {
                '"' -> {
                    val (str, nextIdx) = readBasicString(s, i)
                    segments.add(str)
                    i = nextIdx
                }
                '\'' -> {
                    val (str, nextIdx) = readLiteralString(s, i)
                    segments.add(str)
                    i = nextIdx
                }
                else -> {
                    val start = i
                    while (i < n && s[i] != '.' && s[i] != ' ' && s[i] != '\t') i++
                    val bare = s.substring(start, i)
                    if (bare.isNotEmpty()) segments.add(bare)
                }
            }
            while (i < n && (s[i] == ' ' || s[i] == '\t')) i++
            if (i < n && s[i] == '.') i++
        }
        return segments
    }

    private fun findTopLevelChar(s: String, target: Char): Int {
        var inDouble = false
        var inSingle = false
        var escape = false
        for (i in s.indices) {
            val c = s[i]
            if (escape) {
                escape = false
                continue
            }
            if (c == '\\' && inDouble) {
                escape = true
                continue
            }
            if (c == '"' && !inSingle) inDouble = !inDouble
            else if (c == '\'' && !inDouble) inSingle = !inSingle
            else if (!inDouble && !inSingle) {
                if (c == '#') return -1
                if (c == target) return i
            }
        }
        return -1
    }

    private fun stripLineComment(line: String): String {
        var inDouble = false
        var inSingle = false
        var escape = false
        for (i in line.indices) {
            val c = line[i]
            if (escape) {
                escape = false
                continue
            }
            if (c == '\\' && inDouble) {
                escape = true
                continue
            }
            if (c == '"' && !inSingle) inDouble = !inDouble
            else if (c == '\'' && !inDouble) inSingle = !inSingle
            else if (c == '#' && !inDouble && !inSingle) return line.substring(0, i)
        }
        return line
    }

    /**
     * Checks whether a value expression has closed all multiline strings, `[...]` arrays,
     * and `{...}` inline tables (ignoring comments outside strings).
     */
    private fun isValueComplete(raw: String): Boolean {
        var i = 0
        val n = raw.length
        var inTripleDouble = false
        var inTripleSingle = false
        var inDouble = false
        var inSingle = false
        var bracketDepth = 0
        var braceDepth = 0
        var seenToken = false

        while (i < n) {
            val c = raw[i]
            when {
                inTripleDouble -> {
                    if (c == '\\' && i + 1 < n) {
                        i += 2
                        continue
                    }
                    if (c == '"' && i + 2 < n && raw[i + 1] == '"' && raw[i + 2] == '"') {
                        // Consume up to 2 extra quotes allowed at the end of a multiline basic string
                        i += 3
                        while (i < n && raw[i] == '"') i++
                        inTripleDouble = false
                        continue
                    }
                    i++
                }
                inTripleSingle -> {
                    if (c == '\'' && i + 2 < n && raw[i + 1] == '\'' && raw[i + 2] == '\'') {
                        i += 3
                        while (i < n && raw[i] == '\'') i++
                        inTripleSingle = false
                        continue
                    }
                    i++
                }
                inDouble -> {
                    if (c == '\\' && i + 1 < n) {
                        i += 2
                        continue
                    }
                    if (c == '"') inDouble = false
                    i++
                }
                inSingle -> {
                    if (c == '\'') inSingle = false
                    i++
                }
                c == '#' -> {
                    while (i < n && raw[i] != '\n') i++
                }
                c == '"' -> {
                    seenToken = true
                    if (i + 2 < n && raw[i + 1] == '"' && raw[i + 2] == '"') {
                        inTripleDouble = true
                        i += 3
                    } else {
                        inDouble = true
                        i++
                    }
                }
                c == '\'' -> {
                    seenToken = true
                    if (i + 2 < n && raw[i + 1] == '\'' && raw[i + 2] == '\'') {
                        inTripleSingle = true
                        i += 3
                    } else {
                        inSingle = true
                        i++
                    }
                }
                c == '[' -> {
                    seenToken = true
                    bracketDepth++
                    i++
                }
                c == ']' -> {
                    bracketDepth = (bracketDepth - 1).coerceAtLeast(0)
                    i++
                }
                c == '{' -> {
                    seenToken = true
                    braceDepth++
                    i++
                }
                c == '}' -> {
                    braceDepth = (braceDepth - 1).coerceAtLeast(0)
                    i++
                }
                else -> {
                    if (!c.isWhitespace()) seenToken = true
                    i++
                }
            }
        }
        return seenToken && !inTripleDouble && !inTripleSingle && !inDouble && !inSingle &&
            bracketDepth == 0 && braceDepth == 0
    }

    /** Strips `# ...` comments outside strings from a (possibly multiline) value string. */
    private fun stripTrailingCommentFromValue(raw: String): String {
        val sb = StringBuilder(raw.length)
        var i = 0
        val n = raw.length
        var inTripleDouble = false
        var inTripleSingle = false
        var inDouble = false
        var inSingle = false

        while (i < n) {
            val c = raw[i]
            when {
                inTripleDouble -> {
                    if (c == '\\' && i + 1 < n) {
                        sb.append(c).append(raw[i + 1])
                        i += 2
                        continue
                    }
                    if (c == '"' && i + 2 < n && raw[i + 1] == '"' && raw[i + 2] == '"') {
                        sb.append("\"\"\"")
                        i += 3
                        while (i < n && raw[i] == '"') {
                            sb.append('"')
                            i++
                        }
                        inTripleDouble = false
                        continue
                    }
                    sb.append(c)
                    i++
                }
                inTripleSingle -> {
                    if (c == '\'' && i + 2 < n && raw[i + 1] == '\'' && raw[i + 2] == '\'') {
                        sb.append("'''")
                        i += 3
                        while (i < n && raw[i] == '\'') {
                            sb.append('\'')
                            i++
                        }
                        inTripleSingle = false
                        continue
                    }
                    sb.append(c)
                    i++
                }
                inDouble -> {
                    if (c == '\\' && i + 1 < n) {
                        sb.append(c).append(raw[i + 1])
                        i += 2
                        continue
                    }
                    if (c == '"') inDouble = false
                    sb.append(c)
                    i++
                }
                inSingle -> {
                    if (c == '\'') inSingle = false
                    sb.append(c)
                    i++
                }
                c == '#' -> {
                    while (i < n && raw[i] != '\n') i++
                }
                c == '"' -> {
                    if (i + 2 < n && raw[i + 1] == '"' && raw[i + 2] == '"') {
                        inTripleDouble = true
                        sb.append("\"\"\"")
                        i += 3
                    } else {
                        inDouble = true
                        sb.append(c)
                        i++
                    }
                }
                c == '\'' -> {
                    if (i + 2 < n && raw[i + 1] == '\'' && raw[i + 2] == '\'') {
                        inTripleSingle = true
                        sb.append("'''")
                        i += 3
                    } else {
                        inSingle = true
                        sb.append(c)
                        i++
                    }
                }
                else -> {
                    sb.append(c)
                    i++
                }
            }
        }
        return sb.toString()
    }

    private fun parseValue(rawInput: String): Value? {
        val raw = rawInput.trim()
        if (raw.isEmpty()) return null

        // Multiline basic string """..."""
        if (raw.startsWith("\"\"\"") && raw.endsWith("\"\"\"") && raw.length >= 6) {
            var body = raw.substring(3, raw.length - 3)
            if (body.startsWith("\n")) body = body.substring(1)
            return Value.Str(unescapeBasicString(body, multiline = true))
        }
        // Multiline literal string '''...'''
        if (raw.startsWith("'''") && raw.endsWith("'''") && raw.length >= 6) {
            var body = raw.substring(3, raw.length - 3)
            if (body.startsWith("\n")) body = body.substring(1)
            return Value.Str(body)
        }
        // Basic string "..."
        if (raw.startsWith('"') && raw.endsWith('"') && raw.length >= 2) {
            return Value.Str(unescapeBasicString(raw.substring(1, raw.length - 1), multiline = false))
        }
        // Literal string '...'
        if (raw.startsWith('\'') && raw.endsWith('\'') && raw.length >= 2) {
            return Value.Str(raw.substring(1, raw.length - 1))
        }

        if (raw == "true") return Value.Bool(true)
        if (raw == "false") return Value.Bool(false)

        if (raw.startsWith("[") && raw.endsWith("]")) {
            val inner = raw.substring(1, raw.length - 1).trim()
            if (inner.isEmpty()) return Value.Arr(emptyList())
            val items = splitTopLevelItems(inner, ',')
                .map { it.trim() }
                .filter { it.isNotEmpty() }
                .mapNotNull { parseValue(it) }
            return Value.Arr(items)
        }

        if (raw.startsWith("{") && raw.endsWith("}")) {
            val inner = raw.substring(1, raw.length - 1).trim()
            if (inner.isEmpty()) return Value.Tbl(emptyMap())
            val map = linkedMapOf<String, Value>()
            for (entry in splitTopLevelItems(inner, ',')) {
                val trimmedEntry = entry.trim()
                if (trimmedEntry.isEmpty()) continue
                val eq = findTopLevelChar(trimmedEntry, '=')
                if (eq <= 0) continue
                val keySegments = parseKeyPath(trimmedEntry.substring(0, eq))
                if (keySegments.isEmpty()) continue
                val v = parseValue(trimmedEntry.substring(eq + 1)) ?: continue
                insertNestedTableValue(map, keySegments, v)
            }
            return Value.Tbl(map)
        }

        return parseNumber(raw)
    }

    private fun insertNestedTableValue(
        target: MutableMap<String, Value>,
        keySegments: List<String>,
        value: Value,
    ) {
        if (keySegments.size == 1) {
            target[keySegments[0]] = value
            return
        }
        val head = keySegments[0]
        val existing = (target[head] as? Value.Tbl)?.entries?.toMutableMap() ?: linkedMapOf()
        insertNestedTableValue(existing, keySegments.drop(1), value)
        target[head] = Value.Tbl(existing)
    }

    private fun parseNumber(raw: String): Value.Num? {
        val cleaned = raw.replace("_", "")
        if (cleaned.isEmpty()) return null
        if (cleaned.startsWith("0x", ignoreCase = true)) {
            val parsed = cleaned.substring(2).toLongOrNull(16) ?: return null
            return Value.Num(parsed.toDouble(), isFloat = false)
        }
        if (cleaned.startsWith("0o", ignoreCase = true)) {
            val parsed = cleaned.substring(2).toLongOrNull(8) ?: return null
            return Value.Num(parsed.toDouble(), isFloat = false)
        }
        if (cleaned.startsWith("0b", ignoreCase = true)) {
            val parsed = cleaned.substring(2).toLongOrNull(2) ?: return null
            return Value.Num(parsed.toDouble(), isFloat = false)
        }
        val num = cleaned.toDoubleOrNull() ?: return null
        val isFloat = cleaned.contains('.') || cleaned.contains('e', ignoreCase = true)
        return Value.Num(num, isFloat)
    }

    private fun readBasicString(s: String, startQuote: Int): Pair<String, Int> {
        val sb = StringBuilder()
        var i = startQuote + 1
        val n = s.length
        while (i < n) {
            val c = s[i]
            if (c == '\\' && i + 1 < n) {
                val (decoded, nextIdx) = decodeEscape(s, i + 1, multiline = false)
                sb.append(decoded)
                i = nextIdx
                continue
            }
            if (c == '"') {
                return sb.toString() to (i + 1)
            }
            sb.append(c)
            i++
        }
        return sb.toString() to n
    }

    private fun readLiteralString(s: String, startQuote: Int): Pair<String, Int> {
        val end = s.indexOf('\'', startQuote + 1)
        return if (end < 0) {
            s.substring(startQuote + 1) to s.length
        } else {
            s.substring(startQuote + 1, end) to (end + 1)
        }
    }

    private fun unescapeBasicString(body: String, multiline: Boolean): String {
        val sb = StringBuilder(body.length)
        var i = 0
        val n = body.length
        while (i < n) {
            val c = body[i]
            if (c == '\\' && i + 1 < n) {
                val (decoded, nextIdx) = decodeEscape(body, i + 1, multiline)
                sb.append(decoded)
                i = nextIdx
            } else {
                sb.append(c)
                i++
            }
        }
        return sb.toString()
    }

    private fun decodeEscape(s: String, escIdx: Int, multiline: Boolean): Pair<String, Int> {
        val n = s.length
        if (escIdx >= n) return "\\" to n
        val c = s[escIdx]
        if (multiline && (c == '\n' || c == ' ' || c == '\t')) {
            // Line-ending backslash trims whitespace and newlines until next non-whitespace char
            var j = escIdx
            var sawNewline = false
            while (j < n && (s[j] == ' ' || s[j] == '\t' || s[j] == '\n' || s[j] == '\r')) {
                if (s[j] == '\n') sawNewline = true
                j++
            }
            if (sawNewline) return "" to j
        }
        return when (c) {
            '"' -> "\"" to (escIdx + 1)
            '\\' -> "\\" to (escIdx + 1)
            'n' -> "\n" to (escIdx + 1)
            't' -> "\t" to (escIdx + 1)
            'r' -> "\r" to (escIdx + 1)
            'b' -> "\b" to (escIdx + 1)
            'f' -> "\u000C" to (escIdx + 1)
            'u' -> {
                if (escIdx + 5 <= n) {
                    val code = s.substring(escIdx + 1, escIdx + 5).toIntOrNull(16)
                    if (code != null) return code.toChar().toString() to (escIdx + 5)
                }
                "\\u" to (escIdx + 1)
            }
            'U' -> {
                if (escIdx + 9 <= n) {
                    val code = s.substring(escIdx + 1, escIdx + 9).toIntOrNull(16)
                    if (code != null && Character.isValidCodePoint(code)) {
                        return String(Character.toChars(code)) to (escIdx + 9)
                    }
                }
                "\\U" to (escIdx + 1)
            }
            else -> c.toString() to (escIdx + 1)
        }
    }

    private fun splitTopLevelItems(inner: String, delimiter: Char): List<String> {
        val out = mutableListOf<String>()
        val cur = StringBuilder()
        var inTripleDouble = false
        var inTripleSingle = false
        var inDouble = false
        var inSingle = false
        var escape = false
        var bracketDepth = 0
        var braceDepth = 0
        var i = 0
        val n = inner.length

        while (i < n) {
            val c = inner[i]
            if (escape) {
                cur.append(c)
                escape = false
                i++
                continue
            }
            when {
                inTripleDouble -> {
                    if (c == '\\') {
                        escape = true
                        cur.append(c)
                        i++
                    } else if (c == '"' && i + 2 < n && inner[i + 1] == '"' && inner[i + 2] == '"') {
                        cur.append("\"\"\"")
                        i += 3
                        while (i < n && inner[i] == '"') {
                            cur.append('"')
                            i++
                        }
                        inTripleDouble = false
                    } else {
                        cur.append(c)
                        i++
                    }
                }
                inTripleSingle -> {
                    if (c == '\'' && i + 2 < n && inner[i + 1] == '\'' && inner[i + 2] == '\'') {
                        cur.append("'''")
                        i += 3
                        while (i < n && inner[i] == '\'') {
                            cur.append('\'')
                            i++
                        }
                        inTripleSingle = false
                    } else {
                        cur.append(c)
                        i++
                    }
                }
                inDouble -> {
                    if (c == '\\') {
                        escape = true
                    } else if (c == '"') {
                        inDouble = false
                    }
                    cur.append(c)
                    i++
                }
                inSingle -> {
                    if (c == '\'') {
                        inSingle = false
                    }
                    cur.append(c)
                    i++
                }
                c == '#' -> {
                    while (i < n && inner[i] != '\n') i++
                }
                c == '"' -> {
                    if (i + 2 < n && inner[i + 1] == '"' && inner[i + 2] == '"') {
                        inTripleDouble = true
                        cur.append("\"\"\"")
                        i += 3
                    } else {
                        inDouble = true
                        cur.append(c)
                        i++
                    }
                }
                c == '\'' -> {
                    if (i + 2 < n && inner[i + 1] == '\'' && inner[i + 2] == '\'') {
                        inTripleSingle = true
                        cur.append("'''")
                        i += 3
                    } else {
                        inSingle = true
                        cur.append(c)
                        i++
                    }
                }
                c == '[' -> {
                    bracketDepth++
                    cur.append(c)
                    i++
                }
                c == ']' -> {
                    bracketDepth = (bracketDepth - 1).coerceAtLeast(0)
                    cur.append(c)
                    i++
                }
                c == '{' -> {
                    braceDepth++
                    cur.append(c)
                    i++
                }
                c == '}' -> {
                    braceDepth = (braceDepth - 1).coerceAtLeast(0)
                    cur.append(c)
                    i++
                }
                c == delimiter && bracketDepth == 0 && braceDepth == 0 -> {
                    out.add(cur.toString())
                    cur.setLength(0)
                    i++
                }
                else -> {
                    cur.append(c)
                    i++
                }
            }
        }
        if (cur.isNotEmpty()) out.add(cur.toString())
        return out
    }
}
