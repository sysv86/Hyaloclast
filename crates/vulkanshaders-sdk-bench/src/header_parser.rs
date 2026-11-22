use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParsedHeader {
    pub source_files: Vec<String>,
    pub defines_str: BTreeMap<String, String>, // #define NAME "value"
    pub defines_int: BTreeMap<String, i128>,   // #define NAME 0x1234 or (1 << 3)
    pub enum_values: BTreeMap<String, i128>, // EnumeratorName -> value (also scoped EnumName::Enumerator)
    pub functions: BTreeMap<String, usize>,  // C function name -> parameter count
    pub struct_field_counts: BTreeMap<String, usize>, // struct name -> top-level field count
}

impl ParsedHeader {
    pub fn merge(&mut self, other: ParsedHeader) {
        for f in other.source_files {
            if !self.source_files.contains(&f) {
                self.source_files.push(f);
            }
        }
        self.defines_str.extend(other.defines_str);
        self.defines_int.extend(other.defines_int);
        self.enum_values.extend(other.enum_values);
        self.functions.extend(other.functions);
        self.struct_field_counts.extend(other.struct_field_counts);
    }
}

pub fn parse_c_header(file_label: &str, source: &str) -> ParsedHeader {
    let mut result = ParsedHeader {
        source_files: vec![file_label.to_string()],
        ..Default::default()
    };

    let spliced = join_backslash_continuations(source);
    let stripped = strip_comments_preserving_strings(&spliced);

    parse_defines(&stripped, &mut result);

    let code_only = strip_preprocessor_lines(&stripped);
    parse_enums(&code_only, &mut result);
    parse_structs(&code_only, &mut result);
    parse_functions(&code_only, &mut result);

    result
}

pub fn parse_c_header_file(path: &Path) -> Result<ParsedHeader, String> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read header '{}': {e}", path.display()))?;
    Ok(parse_c_header(&path.display().to_string(), &source))
}

// ============================================================================
// Preprocessing: line continuations, comment stripping, directive filtering
// ============================================================================

fn join_backslash_continuations(source: &str) -> String {
    let normalized = source.replace("\r\n", "\n");
    let mut out = String::with_capacity(normalized.len());
    let bytes = normalized.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
            i += 2;
            continue;
        }
        let ch = normalized[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn strip_comments_preserving_strings(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let chars: Vec<char> = source.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let c = chars[i];
        if c == '"' {
            out.push(c);
            i += 1;
            while i < len {
                let sc = chars[i];
                out.push(sc);
                i += 1;
                if sc == '\\' && i < len {
                    out.push(chars[i]);
                    i += 1;
                } else if sc == '"' {
                    break;
                }
            }
        } else if c == '\'' {
            out.push(c);
            i += 1;
            while i < len {
                let sc = chars[i];
                out.push(sc);
                i += 1;
                if sc == '\\' && i < len {
                    out.push(chars[i]);
                    i += 1;
                } else if sc == '\'' {
                    break;
                }
            }
        } else if c == '/' && i + 1 < len && chars[i + 1] == '/' {
            i += 2;
            while i < len && chars[i] != '\n' {
                i += 1;
            }
        } else if c == '/' && i + 1 < len && chars[i + 1] == '*' {
            i += 2;
            out.push(' ');
            while i + 1 < len && !(chars[i] == '*' && chars[i + 1] == '/') {
                if chars[i] == '\n' {
                    out.push('\n');
                }
                i += 1;
            }
            if i + 1 < len {
                i += 2;
            } else {
                i = len;
            }
        } else {
            out.push(c);
            i += 1;
        }
    }

    out
}

fn strip_preprocessor_lines(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    for line in source.lines() {
        if line.trim_start().starts_with('#') {
            out.push('\n');
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

// ============================================================================
// #define parsing
// ============================================================================

fn parse_defines(source: &str, out: &mut ParsedHeader) {
    for line in source.lines() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix('#') else {
            continue;
        };
        let rest = rest.trim_start();
        let Some(after_define) = rest.strip_prefix("define") else {
            continue;
        };
        if !after_define
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_whitespace())
        {
            continue;
        }
        let rest = after_define.trim_start();
        let ident_len = rest
            .bytes()
            .take_while(|&b| b.is_ascii_alphanumeric() || b == b'_')
            .count();
        if ident_len == 0 {
            continue;
        }
        let name = &rest[..ident_len];
        if !is_ident_start(name.as_bytes()[0]) {
            continue;
        }
        let after_name = &rest[ident_len..];
        // Function-like macro `#define FOO(x) ...` has no whitespace before `(`
        if after_name.starts_with('(') {
            continue;
        }
        let val_str = after_name.trim();
        if val_str.is_empty() {
            continue;
        }

        if let Some(s) = parse_c_string_literal(val_str) {
            out.defines_str.insert(name.to_string(), s);
            continue;
        }

        if let Some(v) = eval_int_expr(val_str, &out.defines_int, &out.enum_values) {
            out.defines_int.insert(name.to_string(), v);
        }
    }
}

fn parse_c_string_literal(input: &str) -> Option<String> {
    let mut s = input.trim();
    // Unwrap outer parentheses if present, e.g. ("foo")
    while s.starts_with('(') && s.ends_with(')') {
        let inner = s[1..s.len() - 1].trim();
        if inner.starts_with('"') && inner.ends_with('"') {
            s = inner;
        } else {
            break;
        }
    }
    if !s.starts_with('"') {
        return None;
    }

    let mut result = String::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut parsed_any = false;

    while i < chars.len() {
        while i < chars.len() && chars[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= chars.len() {
            break;
        }
        if chars[i] != '"' {
            return None;
        }
        parsed_any = true;
        i += 1;
        let mut closed = false;
        while i < chars.len() {
            let c = chars[i];
            i += 1;
            if c == '\\' && i < chars.len() {
                let esc = chars[i];
                i += 1;
                match esc {
                    'n' => result.push('\n'),
                    'r' => result.push('\r'),
                    't' => result.push('\t'),
                    '\\' => result.push('\\'),
                    '"' => result.push('"'),
                    '0' => result.push('\0'),
                    other => result.push(other),
                }
            } else if c == '"' {
                closed = true;
                break;
            } else {
                result.push(c);
            }
        }
        if !closed {
            return None;
        }
    }

    if parsed_any { Some(result) } else { None }
}

// ============================================================================
// Integer Constant Expression Evaluator
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
enum ExprTok {
    Int(i128),
    Ident(String),
    LParen,
    RParen,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Shl,
    Shr,
    BitOr,
    BitAnd,
    BitXor,
    BitNot,
    Bang,
    EqEq,
    NotEq,
    Lt,
    Le,
    Gt,
    Ge,
    AndAnd,
    OrOr,
}

fn tokenize_expr(input: &str) -> Option<Vec<ExprTok>> {
    let mut toks = Vec::new();
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        let b = bytes[i];
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // Character literal e.g. 'A'
        if b == b'\'' {
            i += 1;
            if i >= len {
                return None;
            }
            let val = if bytes[i] == b'\\' && i + 1 < len {
                i += 1;
                let esc = match bytes[i] {
                    b'n' => b'\n',
                    b'r' => b'\r',
                    b't' => b'\t',
                    b'0' => 0,
                    other => other,
                };
                i += 1;
                esc as i128
            } else {
                let c = bytes[i] as i128;
                i += 1;
                c
            };
            if i < len && bytes[i] == b'\'' {
                i += 1;
                toks.push(ExprTok::Int(val));
                continue;
            }
            return None;
        }

        // Integer literal (hex, binary, octal, decimal + u/l/ull suffixes)
        if b.is_ascii_digit() {
            let start = i;
            if b == b'0' && i + 1 < len && (bytes[i + 1] == b'x' || bytes[i + 1] == b'X') {
                i += 2;
                let hex_start = i;
                while i < len && (bytes[i].is_ascii_hexdigit() || bytes[i] == b'\'') {
                    i += 1;
                }
                let digits: String = input[hex_start..i].chars().filter(|&c| c != '\'').collect();
                if digits.is_empty() {
                    return None;
                }
                let v = i128::from_str_radix(&digits, 16).ok()?;
                skip_int_suffixes(bytes, &mut i);
                toks.push(ExprTok::Int(v));
                continue;
            } else if b == b'0' && i + 1 < len && (bytes[i + 1] == b'b' || bytes[i + 1] == b'B') {
                i += 2;
                let bin_start = i;
                while i < len && (bytes[i] == b'0' || bytes[i] == b'1' || bytes[i] == b'\'') {
                    i += 1;
                }
                let digits: String = input[bin_start..i].chars().filter(|&c| c != '\'').collect();
                if digits.is_empty() {
                    return None;
                }
                let v = i128::from_str_radix(&digits, 2).ok()?;
                skip_int_suffixes(bytes, &mut i);
                toks.push(ExprTok::Int(v));
                continue;
            } else {
                while i < len && (bytes[i].is_ascii_digit() || bytes[i] == b'\'') {
                    i += 1;
                }
                // Reject floats like 1.0f or 1e5
                if i < len && (bytes[i] == b'.' || bytes[i] == b'e' || bytes[i] == b'E') {
                    return None;
                }
                let digits: String = input[start..i].chars().filter(|&c| c != '\'').collect();
                let v = if digits.len() > 1 && digits.starts_with('0') {
                    i128::from_str_radix(&digits, 8).ok()?
                } else {
                    digits.parse::<i128>().ok()?
                };
                skip_int_suffixes(bytes, &mut i);
                toks.push(ExprTok::Int(v));
                continue;
            }
        }

        // Identifier (including C++ scoped Enum::Value)
        if is_ident_start(b) {
            let start = i;
            while i < len && (is_ident_continue(bytes[i]) || bytes[i] == b':') {
                if bytes[i] == b':' {
                    if i + 1 < len && bytes[i + 1] == b':' {
                        i += 2;
                    } else {
                        break;
                    }
                } else {
                    i += 1;
                }
            }
            toks.push(ExprTok::Ident(input[start..i].to_string()));
            continue;
        }

        // Two-char operators
        if i + 1 < len {
            let two = &bytes[i..i + 2];
            let matched = match two {
                b"<<" => Some(ExprTok::Shl),
                b">>" => Some(ExprTok::Shr),
                b"==" => Some(ExprTok::EqEq),
                b"!=" => Some(ExprTok::NotEq),
                b"<=" => Some(ExprTok::Le),
                b">=" => Some(ExprTok::Ge),
                b"&&" => Some(ExprTok::AndAnd),
                b"||" => Some(ExprTok::OrOr),
                _ => None,
            };
            if let Some(tok) = matched {
                toks.push(tok);
                i += 2;
                continue;
            }
        }

        // Single-char operators
        let single = match b {
            b'(' => ExprTok::LParen,
            b')' => ExprTok::RParen,
            b'+' => ExprTok::Plus,
            b'-' => ExprTok::Minus,
            b'*' => ExprTok::Star,
            b'/' => ExprTok::Slash,
            b'%' => ExprTok::Percent,
            b'|' => ExprTok::BitOr,
            b'&' => ExprTok::BitAnd,
            b'^' => ExprTok::BitXor,
            b'~' => ExprTok::BitNot,
            b'!' => ExprTok::Bang,
            b'<' => ExprTok::Lt,
            b'>' => ExprTok::Gt,
            _ => return None,
        };
        toks.push(single);
        i += 1;
    }

    Some(toks)
}

fn skip_int_suffixes(bytes: &[u8], i: &mut usize) {
    while *i < bytes.len() && matches!(bytes[*i], b'u' | b'U' | b'l' | b'L') {
        *i += 1;
    }
}

struct ExprParser<'a> {
    toks: &'a [ExprTok],
    pos: usize,
    defines: &'a BTreeMap<String, i128>,
    enums: &'a BTreeMap<String, i128>,
}

impl<'a> ExprParser<'a> {
    fn peek(&self) -> Option<&ExprTok> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Option<&ExprTok> {
        let t = self.toks.get(self.pos)?;
        self.pos += 1;
        Some(t)
    }

    fn parse_or_or(&mut self) -> Option<i128> {
        let mut lhs = self.parse_and_and()?;
        while self.peek() == Some(&ExprTok::OrOr) {
            self.next();
            let rhs = self.parse_and_and()?;
            lhs = i128::from(lhs != 0 || rhs != 0);
        }
        Some(lhs)
    }

    fn parse_and_and(&mut self) -> Option<i128> {
        let mut lhs = self.parse_bit_or()?;
        while self.peek() == Some(&ExprTok::AndAnd) {
            self.next();
            let rhs = self.parse_bit_or()?;
            lhs = i128::from(lhs != 0 && rhs != 0);
        }
        Some(lhs)
    }

    fn parse_bit_or(&mut self) -> Option<i128> {
        let mut lhs = self.parse_bit_xor()?;
        while self.peek() == Some(&ExprTok::BitOr) {
            self.next();
            let rhs = self.parse_bit_xor()?;
            lhs |= rhs;
        }
        Some(lhs)
    }

    fn parse_bit_xor(&mut self) -> Option<i128> {
        let mut lhs = self.parse_bit_and()?;
        while self.peek() == Some(&ExprTok::BitXor) {
            self.next();
            let rhs = self.parse_bit_and()?;
            lhs ^= rhs;
        }
        Some(lhs)
    }

    fn parse_bit_and(&mut self) -> Option<i128> {
        let mut lhs = self.parse_eq()?;
        while self.peek() == Some(&ExprTok::BitAnd) {
            self.next();
            let rhs = self.parse_eq()?;
            lhs &= rhs;
        }
        Some(lhs)
    }

    fn parse_eq(&mut self) -> Option<i128> {
        let mut lhs = self.parse_rel()?;
        loop {
            match self.peek() {
                Some(ExprTok::EqEq) => {
                    self.next();
                    let rhs = self.parse_rel()?;
                    lhs = i128::from(lhs == rhs);
                }
                Some(ExprTok::NotEq) => {
                    self.next();
                    let rhs = self.parse_rel()?;
                    lhs = i128::from(lhs != rhs);
                }
                _ => break,
            }
        }
        Some(lhs)
    }

    fn parse_rel(&mut self) -> Option<i128> {
        let mut lhs = self.parse_shift()?;
        loop {
            match self.peek() {
                Some(ExprTok::Lt) => {
                    self.next();
                    lhs = i128::from(lhs < self.parse_shift()?);
                }
                Some(ExprTok::Le) => {
                    self.next();
                    lhs = i128::from(lhs <= self.parse_shift()?);
                }
                Some(ExprTok::Gt) => {
                    self.next();
                    lhs = i128::from(lhs > self.parse_shift()?);
                }
                Some(ExprTok::Ge) => {
                    self.next();
                    lhs = i128::from(lhs >= self.parse_shift()?);
                }
                _ => break,
            }
        }
        Some(lhs)
    }

    fn parse_shift(&mut self) -> Option<i128> {
        let mut lhs = self.parse_add()?;
        loop {
            match self.peek() {
                Some(ExprTok::Shl) => {
                    self.next();
                    let rhs = self.parse_add()?;
                    if !(0..=127).contains(&rhs) {
                        return None;
                    }
                    lhs = lhs.wrapping_shl(rhs as u32);
                }
                Some(ExprTok::Shr) => {
                    self.next();
                    let rhs = self.parse_add()?;
                    if !(0..=127).contains(&rhs) {
                        return None;
                    }
                    lhs = lhs.wrapping_shr(rhs as u32);
                }
                _ => break,
            }
        }
        Some(lhs)
    }

    fn parse_add(&mut self) -> Option<i128> {
        let mut lhs = self.parse_mul()?;
        loop {
            match self.peek() {
                Some(ExprTok::Plus) => {
                    self.next();
                    lhs = lhs.wrapping_add(self.parse_mul()?);
                }
                Some(ExprTok::Minus) => {
                    self.next();
                    lhs = lhs.wrapping_sub(self.parse_mul()?);
                }
                _ => break,
            }
        }
        Some(lhs)
    }

    fn parse_mul(&mut self) -> Option<i128> {
        let mut lhs = self.parse_unary()?;
        loop {
            match self.peek() {
                Some(ExprTok::Star) => {
                    self.next();
                    lhs = lhs.wrapping_mul(self.parse_unary()?);
                }
                Some(ExprTok::Slash) => {
                    self.next();
                    let rhs = self.parse_unary()?;
                    if rhs == 0 {
                        return None;
                    }
                    lhs = lhs.wrapping_div(rhs);
                }
                Some(ExprTok::Percent) => {
                    self.next();
                    let rhs = self.parse_unary()?;
                    if rhs == 0 {
                        return None;
                    }
                    lhs = lhs.wrapping_rem(rhs);
                }
                _ => break,
            }
        }
        Some(lhs)
    }

    fn parse_unary(&mut self) -> Option<i128> {
        match self.peek() {
            Some(ExprTok::Plus) => {
                self.next();
                self.parse_unary()
            }
            Some(ExprTok::Minus) => {
                self.next();
                Some(self.parse_unary()?.wrapping_neg())
            }
            Some(ExprTok::BitNot) => {
                self.next();
                Some(!self.parse_unary()?)
            }
            Some(ExprTok::Bang) => {
                self.next();
                Some(i128::from(self.parse_unary()? == 0))
            }
            Some(ExprTok::LParen) => {
                // Check for C-style cast like `(uint32_t)0x1234` or `(unsigned int)1`
                if let Some(after_cast) = self.try_skip_c_cast() {
                    let saved = self.pos;
                    self.pos = after_cast;
                    if let Some(val) = self.parse_unary() {
                        return Some(val);
                    }
                    self.pos = saved;
                }
                self.parse_primary()
            }
            _ => self.parse_primary(),
        }
    }

    fn try_skip_c_cast(&self) -> Option<usize> {
        if self.toks.get(self.pos) != Some(&ExprTok::LParen) {
            return None;
        }
        let mut i = self.pos + 1;
        let mut saw_ident = false;
        while i < self.toks.len() {
            match &self.toks[i] {
                ExprTok::Ident(name) => {
                    // If this identifier is a known constant, it's probably `(CONST)`, not a cast
                    if !saw_ident
                        && (self.defines.contains_key(name) || self.enums.contains_key(name))
                    {
                        return None;
                    }
                    saw_ident = true;
                    i += 1;
                }
                ExprTok::Star => {
                    i += 1;
                }
                ExprTok::RParen => {
                    if saw_ident && i + 1 < self.toks.len() {
                        return Some(i + 1);
                    }
                    return None;
                }
                _ => return None,
            }
        }
        None
    }

    fn parse_primary(&mut self) -> Option<i128> {
        match self.next()?.clone() {
            ExprTok::Int(v) => Some(v),
            ExprTok::Ident(name) => {
                if let Some(&v) = self.enums.get(&name) {
                    Some(v)
                } else if let Some(&v) = self.defines.get(&name) {
                    Some(v)
                } else if let Some((_, short)) = name.rsplit_once("::") {
                    self.enums
                        .get(short)
                        .or_else(|| self.defines.get(short))
                        .copied()
                } else {
                    None
                }
            }
            ExprTok::LParen => {
                let v = self.parse_or_or()?;
                if self.next() != Some(&ExprTok::RParen) {
                    return None;
                }
                Some(v)
            }
            _ => None,
        }
    }
}

fn eval_int_expr(
    input: &str,
    defines: &BTreeMap<String, i128>,
    enums: &BTreeMap<String, i128>,
) -> Option<i128> {
    let toks = tokenize_expr(input)?;
    if toks.is_empty() {
        return None;
    }
    let mut parser = ExprParser {
        toks: &toks,
        pos: 0,
        defines,
        enums,
    };
    let val = parser.parse_or_or()?;
    if parser.pos == toks.len() {
        Some(val)
    } else {
        None
    }
}

// ============================================================================
// Enum Parser
// ============================================================================

fn parse_enums(code: &str, out: &mut ParsedHeader) {
    let bytes = code.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        // Skip string literals
        if bytes[i] == b'"' || bytes[i] == b'\'' {
            i = skip_quoted(bytes, i);
            continue;
        }

        if is_keyword_at(bytes, i, b"enum") {
            let is_typedef = preceded_by_keyword(code, i, "typedef");
            let mut cur = i + 4;
            skip_ws(bytes, &mut cur);

            // Optional `class` or `struct` (C++ scoped enum)
            if is_keyword_at(bytes, cur, b"class") {
                cur += 5;
                skip_ws(bytes, &mut cur);
            } else if is_keyword_at(bytes, cur, b"struct") {
                cur += 6;
                skip_ws(bytes, &mut cur);
            }

            // Optional enum tag name
            let tag_name = read_ident_at(code, &mut cur);
            skip_ws(bytes, &mut cur);

            // Optional underlying type `: uint32_t`
            if cur < len && bytes[cur] == b':' {
                cur += 1;
                while cur < len && bytes[cur] != b'{' && bytes[cur] != b';' {
                    cur += 1;
                }
            }

            if cur < len && bytes[cur] == b'{' {
                let body_start = cur + 1;
                if let Some(body_end) = find_matching_brace(bytes, cur) {
                    let body = &code[body_start..body_end];
                    let mut after = body_end + 1;
                    skip_ws(bytes, &mut after);
                    let typedef_name = if is_typedef {
                        read_ident_at(code, &mut after)
                    } else {
                        None
                    };

                    let mut scopes = Vec::new();
                    if let Some(ref t) = tag_name {
                        scopes.push(t.as_str());
                    }
                    if let Some(ref td) = typedef_name {
                        if !scopes.contains(&td.as_str()) {
                            scopes.push(td.as_str());
                        }
                    }

                    parse_enum_body(body, &scopes, out);
                    i = body_end + 1;
                    continue;
                }
            }
        }

        i += 1;
    }
}

fn parse_enum_body(body: &str, scopes: &[&str], out: &mut ParsedHeader) {
    let mut next_val: i128 = 0;

    for entry in split_top_level(body, ',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let (name_part, expr_part) = match entry.split_once('=') {
            Some((n, e)) => (n.trim(), Some(e.trim())),
            None => (entry, None),
        };

        // Take first valid identifier from name_part (ignores trailing attributes)
        let mut pos = 0;
        let Some(ident) = read_ident_at(name_part, &mut pos) else {
            continue;
        };

        let val = if let Some(expr) = expr_part {
            match eval_int_expr(expr, &out.defines_int, &out.enum_values) {
                Some(v) => {
                    next_val = v.wrapping_add(1);
                    v
                }
                None => {
                    next_val = next_val.wrapping_add(1);
                    continue;
                }
            }
        } else {
            let v = next_val;
            next_val = next_val.wrapping_add(1);
            v
        };

        out.enum_values.insert(ident.clone(), val);
        for scope in scopes {
            out.enum_values.insert(format!("{scope}::{ident}"), val);
        }
    }
}

// ============================================================================
// Struct Parser
// ============================================================================

fn parse_structs(code: &str, out: &mut ParsedHeader) {
    let bytes = code.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        if bytes[i] == b'"' || bytes[i] == b'\'' {
            i = skip_quoted(bytes, i);
            continue;
        }

        if is_keyword_at(bytes, i, b"struct") {
            // Skip `enum struct`
            if preceded_by_keyword(code, i, "enum") {
                i += 6;
                continue;
            }
            let is_typedef = preceded_by_keyword(code, i, "typedef");
            let mut cur = i + 6;
            skip_ws(bytes, &mut cur);

            // Read identifiers between `struct` and `{` (handles API export macros like `struct NRD_API Foo`)
            let mut idents = Vec::new();
            while cur < len {
                skip_ws(bytes, &mut cur);
                if cur >= len || bytes[cur] == b'{' || bytes[cur] == b';' || bytes[cur] == b':' {
                    break;
                }
                if let Some(id) = read_ident_at(code, &mut cur) {
                    idents.push(id);
                } else {
                    break;
                }
            }

            // Optional C++ inheritance `: public Base`
            if cur < len && bytes[cur] == b':' {
                cur += 1;
                while cur < len && bytes[cur] != b'{' && bytes[cur] != b';' {
                    cur += 1;
                }
            }

            if cur < len && bytes[cur] == b'{' {
                let body_start = cur + 1;
                if let Some(body_end) = find_matching_brace(bytes, cur) {
                    let body = &code[body_start..body_end];
                    let mut after = body_end + 1;
                    skip_ws(bytes, &mut after);
                    let typedef_name = if is_typedef {
                        read_ident_at(code, &mut after)
                    } else {
                        None
                    };

                    let tag_name = idents.last().cloned();
                    let field_count = count_struct_fields(body);

                    if let Some(ref td) = typedef_name {
                        out.struct_field_counts.insert(td.clone(), field_count);
                    }
                    if let Some(ref tag) = tag_name {
                        out.struct_field_counts.insert(tag.clone(), field_count);
                    }

                    i = body_end + 1;
                    continue;
                }
            }
        }

        i += 1;
    }
}

fn count_struct_fields(body: &str) -> usize {
    let bytes = body.as_bytes();
    let len = bytes.len();
    let mut brace_depth = 0usize;
    let mut paren_depth = 0usize;
    let mut stmt_start = 0usize;
    let mut count = 0usize;
    let mut i = 0;

    while i < len {
        let b = bytes[i];
        if b == b'"' || b == b'\'' {
            i = skip_quoted(bytes, i);
            continue;
        }
        match b {
            b'{' => brace_depth += 1,
            b'}' => {
                if brace_depth > 0 {
                    brace_depth -= 1;
                    if brace_depth == 0 {
                        // Check if this closes an inline method body (e.g. `void foo() { ... }`)
                        let prefix = body[stmt_start..=i].trim();
                        if looks_like_inline_method(prefix) {
                            stmt_start = i + 1;
                        }
                    }
                }
            }
            b'(' => paren_depth += 1,
            b')' => paren_depth = paren_depth.saturating_sub(1),
            b';' if brace_depth == 0 && paren_depth == 0 => {
                let stmt = body[stmt_start..i].trim();
                let stmt = strip_access_specifier(stmt);
                if is_struct_field_statement(stmt) {
                    count += 1;
                }
                stmt_start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }

    count
}

fn strip_access_specifier(mut stmt: &str) -> &str {
    loop {
        let trimmed = stmt.trim_start();
        let next = trimmed
            .strip_prefix("public:")
            .or_else(|| trimmed.strip_prefix("private:"))
            .or_else(|| trimmed.strip_prefix("protected:"));
        if let Some(rest) = next {
            stmt = rest.trim_start();
        } else {
            return trimmed;
        }
    }
}

fn looks_like_inline_method(stmt: &str) -> bool {
    let stmt = strip_access_specifier(stmt);
    // An inline method has `(` ... `)` before its `{` body, whereas a nested struct/union does not.
    if let Some(brace_pos) = stmt.find('{') {
        let before_brace = &stmt[..brace_pos];
        if before_brace.contains('(') && before_brace.contains(')') {
            return true;
        }
    }
    false
}

fn is_struct_field_statement(stmt: &str) -> bool {
    if stmt.is_empty() {
        return false;
    }
    // Skip nested type declarations / C++ directives that are not fields
    for prefix in [
        "typedef ",
        "using ",
        "friend ",
        "static_assert",
        "static ",
        "virtual ",
        "inline ",
        "constexpr ",
        "explicit ",
    ] {
        if stmt.starts_with(prefix) {
            return false;
        }
    }
    // Skip nested `enum ... { ... }` type definitions without a field declarator
    if stmt.starts_with("enum ") && stmt.ends_with('}') {
        return false;
    }
    // Skip nested `struct Name { ... }` type-only declarations (no variable name after `}`),
    // while keeping anonymous `union { ... }` / `struct { ... }` and named `struct { ... } field`.
    if stmt.starts_with("struct ") && stmt.ends_with('}') {
        if let Some(brace_pos) = stmt.find('{') {
            let header = stmt[6..brace_pos].trim();
            if !header.is_empty() {
                return false;
            }
        }
    }
    // Skip non-inline C++ method declarations like `void Reset()` (has `(` but not function pointer `(*`)
    if let Some(paren_pos) = stmt.find('(') {
        // If there is an `=` before `(`, it's a default member initializer like `int x = foo()` -> field!
        let before_paren = &stmt[..paren_pos];
        if !before_paren.contains('=') {
            let after_paren = stmt[paren_pos + 1..].trim_start();
            // Function pointer fields look like `void (*pfn)(...)` or `void (API *pfn)(...)`
            let is_fn_ptr = after_paren.starts_with('*')
                || after_paren
                    .split_once(')')
                    .is_some_and(|(inside, _)| inside.contains('*'));
            if !is_fn_ptr {
                return false;
            }
        }
    }
    true
}

// ============================================================================
// C Function Declaration Parser
// ============================================================================

fn parse_functions(code: &str, out: &mut ParsedHeader) {
    // Walk top-level statements, recursively descending into `extern "C" { ... }` and `namespace ... { ... }`
    parse_functions_in_scope(code, out);
}

fn parse_functions_in_scope(code: &str, out: &mut ParsedHeader) {
    let bytes = code.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    let mut stmt_start = 0;

    while i < len {
        let b = bytes[i];
        if b == b'"' || b == b'\'' {
            i = skip_quoted(bytes, i);
            continue;
        }

        if b == b'{' {
            let prefix = code[stmt_start..i].trim();
            if let Some(block_end) = find_matching_brace(bytes, i) {
                let inner = &code[i + 1..block_end];
                if is_extern_c_or_namespace(prefix) {
                    parse_functions_in_scope(inner, out);
                } else {
                    // Could be an inline function definition like `static inline int foo(int a) { ... }`
                    try_parse_function_signature(prefix, out);
                }
                i = block_end + 1;
                stmt_start = i;
                continue;
            }
        } else if b == b';' {
            let stmt = code[stmt_start..i].trim();
            try_parse_function_signature(stmt, out);
            stmt_start = i + 1;
        }

        i += 1;
    }
}

fn is_extern_c_or_namespace(prefix: &str) -> bool {
    let p = prefix.trim();
    if p.starts_with("extern") && p.contains("\"C\"") && !p.contains('(') {
        return true;
    }
    if p.starts_with("namespace") && !p.contains('(') {
        return true;
    }
    false
}

fn try_parse_function_signature(stmt: &str, out: &mut ParsedHeader) {
    let stmt = stmt.trim();
    if stmt.is_empty() {
        return;
    }

    // Skip typedefs, structs, enums, unions, classes, templates, using/static_assert
    for skip in [
        "typedef ",
        "typedef\t",
        "typedef\n",
        "struct ",
        "enum ",
        "union ",
        "class ",
        "using ",
        "static_assert",
        "return ",
    ] {
        if stmt.starts_with(skip) {
            return;
        }
    }

    // Must contain `(` and `)`
    let Some(open_paren) = find_param_list_open_paren(stmt) else {
        return;
    };
    let before_paren = stmt[..open_paren].trim();
    // Skip variable initializers like `const int x = foo(a, b)`
    if before_paren.contains('=') {
        return;
    }

    let Some(close_paren) = find_matching_paren(stmt.as_bytes(), open_paren) else {
        return;
    };

    // After closing paren, only allow empty or trailing qualifiers/attributes (no `=` or extra code)
    let after_paren = stmt[close_paren + 1..].trim();
    if after_paren.contains('=') {
        return;
    }

    // Extract tokens before `(`: must have at least a return type and a function name
    let tokens = collect_prefix_tokens(before_paren);
    if tokens.len() < 2 {
        return;
    }
    let func_name = tokens.last().unwrap();
    if !is_valid_c_ident(func_name) || is_c_control_keyword(func_name) {
        return;
    }

    // Ensure there is a real return-type token (not just `extern "C"` or macros alone)
    let has_return_type = tokens[..tokens.len() - 1]
        .iter()
        .any(|t| t != "extern" && t != "\"C\"");
    if !has_return_type {
        return;
    }

    let params_str = stmt[open_paren + 1..close_paren].trim();
    let param_count = count_parameters(params_str);
    out.functions.insert(func_name.clone(), param_count);
}

fn find_param_list_open_paren(stmt: &str) -> Option<usize> {
    // Skip `__attribute__((...))` or `__declspec(...)` before the function name
    let bytes = stmt.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'"' || bytes[i] == b'\'' {
            i = skip_quoted(bytes, i);
            continue;
        }
        if is_keyword_at(bytes, i, b"__attribute__") || is_keyword_at(bytes, i, b"__declspec") {
            while i < bytes.len() && bytes[i] != b'(' {
                i += 1;
            }
            if i < bytes.len() {
                if let Some(close) = find_matching_paren(bytes, i) {
                    i = close + 1;
                    continue;
                }
            }
        }
        if bytes[i] == b'(' {
            // Skip function-pointer declarations `ret (*name)(args)`
            let after = stmt[i + 1..].trim_start();
            if after.starts_with('*') {
                return None;
            }
            return Some(i);
        }
        i += 1;
    }
    None
}

fn collect_prefix_tokens(prefix: &str) -> Vec<String> {
    let mut toks = Vec::new();
    let bytes = prefix.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        skip_ws(bytes, &mut i);
        if i >= len {
            break;
        }
        if bytes[i] == b'"' {
            let start = i;
            i = skip_quoted(bytes, i);
            toks.push(prefix[start..i].to_string());
            continue;
        }
        if is_keyword_at(bytes, i, b"__attribute__") || is_keyword_at(bytes, i, b"__declspec") {
            while i < len && bytes[i] != b'(' {
                i += 1;
            }
            if i < len {
                if let Some(close) = find_matching_paren(bytes, i) {
                    i = close + 1;
                    continue;
                }
            }
        }
        if is_ident_start(bytes[i]) {
            let start = i;
            while i < len && is_ident_continue(bytes[i]) {
                i += 1;
            }
            toks.push(prefix[start..i].to_string());
        } else if bytes[i] == b'*' || bytes[i] == b'&' {
            toks.push((bytes[i] as char).to_string());
            i += 1;
        } else {
            i += 1;
        }
    }
    toks
}

fn count_parameters(params: &str) -> usize {
    let trimmed = params.trim();
    if trimmed.is_empty() || trimmed == "void" {
        return 0;
    }
    split_top_level(trimmed, ',')
        .into_iter()
        .filter(|p| !p.trim().is_empty())
        .count()
}

// ============================================================================
// Scanning utilities
// ============================================================================

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn is_valid_c_ident(s: &str) -> bool {
    let bytes = s.as_bytes();
    !bytes.is_empty() && is_ident_start(bytes[0]) && bytes.iter().all(|&b| is_ident_continue(b))
}

fn is_c_control_keyword(s: &str) -> bool {
    matches!(
        s,
        "if" | "else"
            | "for"
            | "while"
            | "switch"
            | "return"
            | "sizeof"
            | "typeof"
            | "alignof"
            | "decltype"
            | "defined"
    )
}

fn skip_ws(bytes: &[u8], i: &mut usize) {
    while *i < bytes.len() && bytes[*i].is_ascii_whitespace() {
        *i += 1;
    }
}

fn skip_quoted(bytes: &[u8], start: usize) -> usize {
    let quote = bytes[start];
    let mut i = start + 1;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            i += 2;
        } else if bytes[i] == quote {
            return i + 1;
        } else {
            i += 1;
        }
    }
    bytes.len()
}

fn is_keyword_at(bytes: &[u8], pos: usize, kw: &[u8]) -> bool {
    if pos + kw.len() > bytes.len() || &bytes[pos..pos + kw.len()] != kw {
        return false;
    }
    let before_ok = pos == 0 || !is_ident_continue(bytes[pos - 1]);
    let after_ok = pos + kw.len() == bytes.len() || !is_ident_continue(bytes[pos + kw.len()]);
    before_ok && after_ok
}

fn preceded_by_keyword(code: &str, pos: usize, kw: &str) -> bool {
    let before = code[..pos].trim_end();
    if let Some(rest) = before.strip_suffix(kw) {
        rest.is_empty() || !is_ident_continue(rest.as_bytes()[rest.len() - 1])
    } else {
        false
    }
}

fn read_ident_at(code: &str, cur: &mut usize) -> Option<String> {
    let bytes = code.as_bytes();
    skip_ws(bytes, cur);
    if *cur >= bytes.len() || !is_ident_start(bytes[*cur]) {
        return None;
    }
    let start = *cur;
    while *cur < bytes.len() && is_ident_continue(bytes[*cur]) {
        *cur += 1;
    }
    Some(code[start..*cur].to_string())
}

fn find_matching_brace(bytes: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut i = open;
    while i < bytes.len() {
        if bytes[i] == b'"' || bytes[i] == b'\'' {
            i = skip_quoted(bytes, i);
            continue;
        }
        if bytes[i] == b'{' {
            depth += 1;
        } else if bytes[i] == b'}' {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

fn find_matching_paren(bytes: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut i = open;
    while i < bytes.len() {
        if bytes[i] == b'"' || bytes[i] == b'\'' {
            i = skip_quoted(bytes, i);
            continue;
        }
        if bytes[i] == b'(' {
            depth += 1;
        } else if bytes[i] == b')' {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

fn split_top_level(input: &str, delim: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let bytes = input.as_bytes();
    let delim_b = delim as u8;
    let mut paren = 0usize;
    let mut brace = 0usize;
    let mut bracket = 0usize;
    let mut start = 0usize;
    let mut i = 0;

    while i < bytes.len() {
        let b = bytes[i];
        if b == b'"' || b == b'\'' {
            i = skip_quoted(bytes, i);
            continue;
        }
        match b {
            b'(' => paren += 1,
            b')' => paren = paren.saturating_sub(1),
            b'{' => brace += 1,
            b'}' => brace = brace.saturating_sub(1),
            b'[' => bracket += 1,
            b']' => bracket = bracket.saturating_sub(1),
            _ if b == delim_b && paren == 0 && brace == 0 && bracket == 0 => {
                parts.push(&input[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    parts.push(&input[start..]);
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_defines_enums_structs_functions() {
        let src = r#"
/* Block comment with // inside and #define FAKE 1 */
#define NGX_PARAM_WIDTH "Width"
#define BASE_FLAG (1u << 3)
#define COMBINED_FLAG (BASE_FLAG | 0x04UL)
#define NEG_CODE (-1)
#define CONTINUED_VAL \
    (COMBINED_FLAG + 2)

extern "C" {

typedef enum NVSDK_NGX_Result {
    NVSDK_NGX_Result_Success = 0x1,
    NVSDK_NGX_Result_Fail = 0xBAD00000,
    NVSDK_NGX_Result_Next,
} NVSDK_NGX_Result;

enum class Denoiser : uint32_t {
    REBLUR_DIFFUSE = 0,
    REBLUR_SPECULAR,
    MAX_NUM = (REBLUR_SPECULAR + 1)
};

typedef struct FfxApiHeader {
    uint64_t type;
    struct FfxApiHeader* pNext;
    union {
        uint32_t a;
        float b;
    };
} FfxApiHeader;

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_VULKAN_Init(
    unsigned long long InApplicationId,
    const wchar_t* InApplicationDataPath,
    void* InInstance,
    void* InPD,
    void* InDevice
);

FFX_API int ffxNoArgs(void);
XESS_EXPORT int xessEmpty();

}
"#;
        let parsed = parse_c_header("sample.h", src);
        assert_eq!(
            parsed
                .defines_str
                .get("NGX_PARAM_WIDTH")
                .map(String::as_str),
            Some("Width")
        );
        assert_eq!(parsed.defines_int.get("BASE_FLAG"), Some(&8));
        assert_eq!(parsed.defines_int.get("COMBINED_FLAG"), Some(&12));
        assert_eq!(parsed.defines_int.get("NEG_CODE"), Some(&-1));
        assert_eq!(parsed.defines_int.get("CONTINUED_VAL"), Some(&14));
        assert!(!parsed.defines_int.contains_key("FAKE"));

        assert_eq!(parsed.enum_values.get("NVSDK_NGX_Result_Success"), Some(&1));
        assert_eq!(
            parsed.enum_values.get("NVSDK_NGX_Result_Fail"),
            Some(&0xBAD0_0000)
        );
        assert_eq!(
            parsed.enum_values.get("NVSDK_NGX_Result_Next"),
            Some(&0xBAD0_0001)
        );
        assert_eq!(
            parsed.enum_values.get("Denoiser::REBLUR_SPECULAR"),
            Some(&1)
        );
        assert_eq!(parsed.enum_values.get("Denoiser::MAX_NUM"), Some(&2));

        assert_eq!(parsed.struct_field_counts.get("FfxApiHeader"), Some(&3));
        assert_eq!(parsed.functions.get("NVSDK_NGX_VULKAN_Init"), Some(&5));
        assert_eq!(parsed.functions.get("ffxNoArgs"), Some(&0));
        assert_eq!(parsed.functions.get("xessEmpty"), Some(&0));
    }
}
