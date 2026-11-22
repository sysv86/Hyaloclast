use std::collections::BTreeSet;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryFormat {
    Elf64,
    Pe32Plus,
    UnixAr,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedBinary {
    pub path: String,
    pub format: BinaryFormat,
    pub architecture: String, // e.g. "x86_64", "aarch64", "unknown"
    pub blake3_hex: String,
    pub size_bytes: u64,
    pub soname_or_dll_name: Option<String>,
    pub dependencies: Vec<String>, // DT_NEEDED for ELF, Import DLLs for PE
    pub exported_symbols: BTreeSet<String>,
    pub host_dlopen_ok: Option<bool>,
    pub host_dlopen_error: Option<String>,
}

pub fn parse_binary_bytes(path_label: &str, bytes: &[u8]) -> Result<ParsedBinary, String> {
    let blake3_hex = blake3::hash(bytes).to_hex().to_string();
    let size_bytes = bytes.len() as u64;

    if bytes.len() >= 4 && &bytes[0..4] == b"\x7fELF" {
        return parse_elf64(path_label, bytes, blake3_hex, size_bytes);
    }

    if bytes.len() >= 64 && &bytes[0..2] == b"MZ" {
        return parse_pe32_plus(path_label, bytes, blake3_hex, size_bytes);
    }

    if bytes.len() >= 8 && &bytes[0..8] == b"!<arch>\n" {
        return parse_unix_ar(path_label, bytes, blake3_hex, size_bytes);
    }

    Ok(ParsedBinary {
        path: path_label.to_string(),
        format: BinaryFormat::Unknown,
        architecture: "unknown".to_string(),
        blake3_hex,
        size_bytes,
        soname_or_dll_name: None,
        dependencies: Vec::new(),
        exported_symbols: BTreeSet::new(),
        host_dlopen_ok: None,
        host_dlopen_error: None,
    })
}

pub fn parse_binary_file(path: &Path, try_host_dlopen: bool) -> Result<ParsedBinary, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("Failed to read binary '{}': {e}", path.display()))?;
    let label = path.display().to_string();
    let mut parsed = parse_binary_bytes(&label, &bytes)?;

    if try_host_dlopen && should_try_host_dlopen(path, &parsed.format) {
        // SAFETY: We explicitly test OS dynamic loader resolution on a user/SDK library path
        // and immediately drop the handle without invoking any symbols.
        match unsafe { libloading::Library::new(path) } {
            Ok(lib) => {
                drop(lib);
                parsed.host_dlopen_ok = Some(true);
                parsed.host_dlopen_error = None;
            }
            Err(err) => {
                parsed.host_dlopen_ok = Some(false);
                parsed.host_dlopen_error = Some(err.to_string());
            }
        }
    }

    Ok(parsed)
}

fn should_try_host_dlopen(path: &Path, format: &BinaryFormat) -> bool {
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    if cfg!(target_os = "linux") {
        *format == BinaryFormat::Elf64 && (file_name.ends_with(".so") || file_name.contains(".so."))
    } else if cfg!(target_os = "windows") {
        *format == BinaryFormat::Pe32Plus && file_name.ends_with(".dll")
    } else {
        false
    }
}

// ============================================================================
// Low-level byte helpers
// ============================================================================

fn read_u16_le(bytes: &[u8], off: usize) -> Option<u16> {
    let slice = bytes.get(off..off.checked_add(2)?)?;
    Some(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32_le(bytes: &[u8], off: usize) -> Option<u32> {
    let slice = bytes.get(off..off.checked_add(4)?)?;
    Some(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_u32_be(bytes: &[u8], off: usize) -> Option<u32> {
    let slice = bytes.get(off..off.checked_add(4)?)?;
    Some(u32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_u64_le(bytes: &[u8], off: usize) -> Option<u64> {
    let slice = bytes.get(off..off.checked_add(8)?)?;
    Some(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn read_i64_le(bytes: &[u8], off: usize) -> Option<i64> {
    let slice = bytes.get(off..off.checked_add(8)?)?;
    Some(i64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn read_u64_be(bytes: &[u8], off: usize) -> Option<u64> {
    let slice = bytes.get(off..off.checked_add(8)?)?;
    Some(u64::from_be_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn read_c_string(bytes: &[u8], off: usize) -> Option<String> {
    if off >= bytes.len() {
        return None;
    }
    let tail = &bytes[off..];
    let end = tail.iter().position(|&b| b == 0).unwrap_or(tail.len());
    Some(String::from_utf8_lossy(&tail[..end]).into_owned())
}

fn push_unique(vec: &mut Vec<String>, item: String) {
    if !item.is_empty() && !vec.iter().any(|existing| existing == &item) {
        vec.push(item);
    }
}

// ============================================================================
// ELF64 Parser
// ============================================================================

const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;

const SHT_SYMTAB: u32 = 2;
const SHT_DYNAMIC: u32 = 6;
const SHT_DYNSYM: u32 = 11;

const STB_GLOBAL: u8 = 1;
const STB_WEAK: u8 = 2;

const DT_NULL: i64 = 0;
const DT_NEEDED: i64 = 1;
const DT_SONAME: i64 = 14;

#[derive(Debug, Clone, Copy)]
struct Elf64SectionHeader {
    sh_type: u32,
    sh_offset: u64,
    sh_size: u64,
    sh_link: u32,
    sh_entsize: u64,
}

fn elf_machine_name(e_machine: u16) -> String {
    match e_machine {
        62 => "x86_64".to_string(),
        183 => "aarch64".to_string(),
        3 => "x86".to_string(),
        40 => "arm".to_string(),
        243 => "riscv".to_string(),
        other => format!("elf_machine_{other}"),
    }
}

fn parse_elf64(
    path_label: &str,
    bytes: &[u8],
    blake3_hex: String,
    size_bytes: u64,
) -> Result<ParsedBinary, String> {
    if bytes.len() < 64 {
        return Err(format!(
            "ELF file '{path_label}' is truncated ({} bytes < 64)",
            bytes.len()
        ));
    }

    let class = bytes[4];
    if class != ELFCLASS64 {
        return Err(format!(
            "ELF file '{path_label}' has unsupported class {class} (expected 2 / ELF64)"
        ));
    }

    let data_encoding = bytes[5];
    if data_encoding != ELFDATA2LSB {
        return Err(format!(
            "ELF file '{path_label}' has unsupported endianness {data_encoding} (expected 1 / little-endian)"
        ));
    }

    let e_machine = read_u16_le(bytes, 18)
        .ok_or_else(|| format!("ELF file '{path_label}' missing e_machine"))?;
    let architecture = elf_machine_name(e_machine);

    let e_shoff = read_u64_le(bytes, 40)
        .ok_or_else(|| format!("ELF file '{path_label}' missing e_shoff"))?
        as usize;
    let e_shentsize = read_u16_le(bytes, 58)
        .ok_or_else(|| format!("ELF file '{path_label}' missing e_shentsize"))?
        as usize;
    let mut e_shnum = read_u16_le(bytes, 60)
        .ok_or_else(|| format!("ELF file '{path_label}' missing e_shnum"))?
        as usize;

    let mut sections = Vec::new();
    if e_shoff != 0 && e_shentsize >= 64 {
        if e_shnum == 0 {
            // Extended section numbering: sh_size of section 0 holds the actual count.
            if let Some(ext_shnum) = read_u64_le(bytes, e_shoff.saturating_add(32)) {
                e_shnum = ext_shnum as usize;
            }
        }

        for idx in 0..e_shnum {
            let off = match idx
                .checked_mul(e_shentsize)
                .and_then(|d| e_shoff.checked_add(d))
            {
                Some(o) if o.saturating_add(64) <= bytes.len() => o,
                _ => break,
            };
            let sh_type = read_u32_le(bytes, off + 4).unwrap_or(0);
            let sh_offset = read_u64_le(bytes, off + 24).unwrap_or(0);
            let sh_size = read_u64_le(bytes, off + 32).unwrap_or(0);
            let sh_link = read_u32_le(bytes, off + 40).unwrap_or(0);
            let sh_entsize = read_u64_le(bytes, off + 56).unwrap_or(0);
            sections.push(Elf64SectionHeader {
                sh_type,
                sh_offset,
                sh_size,
                sh_link,
                sh_entsize,
            });
        }
    }

    let mut exported_symbols = BTreeSet::new();
    let mut dependencies = Vec::new();
    let mut soname_or_dll_name = None;

    for sec in &sections {
        match sec.sh_type {
            SHT_DYNSYM | SHT_SYMTAB => {
                let Some(strtab_sec) = sections.get(sec.sh_link as usize) else {
                    continue;
                };
                let Some(strtab) = section_slice(bytes, *strtab_sec) else {
                    continue;
                };
                let Some(sym_bytes) = section_slice(bytes, *sec) else {
                    continue;
                };
                let entsize = if sec.sh_entsize >= 24 {
                    sec.sh_entsize as usize
                } else {
                    24
                };
                let mut pos = 0usize;
                while pos.saturating_add(24) <= sym_bytes.len() {
                    let st_name = read_u32_le(sym_bytes, pos).unwrap_or(0) as usize;
                    let st_info = sym_bytes[pos + 4];
                    let bind = st_info >> 4;
                    let st_shndx = read_u16_le(sym_bytes, pos + 6).unwrap_or(0);

                    if st_shndx != 0 && (bind == STB_GLOBAL || bind == STB_WEAK) {
                        if let Some(name) = read_c_string(strtab, st_name) {
                            if !name.is_empty() {
                                exported_symbols.insert(name);
                            }
                        }
                    }

                    pos = pos.saturating_add(entsize);
                }
            }
            SHT_DYNAMIC => {
                let Some(strtab_sec) = sections.get(sec.sh_link as usize) else {
                    continue;
                };
                let Some(strtab) = section_slice(bytes, *strtab_sec) else {
                    continue;
                };
                let Some(dyn_bytes) = section_slice(bytes, *sec) else {
                    continue;
                };
                let entsize = if sec.sh_entsize >= 16 {
                    sec.sh_entsize as usize
                } else {
                    16
                };
                let mut pos = 0usize;
                while pos.saturating_add(16) <= dyn_bytes.len() {
                    let d_tag = read_i64_le(dyn_bytes, pos).unwrap_or(DT_NULL);
                    let d_val = read_u64_le(dyn_bytes, pos + 8).unwrap_or(0);
                    if d_tag == DT_NULL {
                        break;
                    }
                    if d_tag == DT_NEEDED {
                        if let Some(dep) = read_c_string(strtab, d_val as usize) {
                            push_unique(&mut dependencies, dep);
                        }
                    } else if d_tag == DT_SONAME {
                        if let Some(soname) = read_c_string(strtab, d_val as usize) {
                            if !soname.is_empty() {
                                soname_or_dll_name = Some(soname);
                            }
                        }
                    }
                    pos = pos.saturating_add(entsize);
                }
            }
            _ => {}
        }
    }

    Ok(ParsedBinary {
        path: path_label.to_string(),
        format: BinaryFormat::Elf64,
        architecture,
        blake3_hex,
        size_bytes,
        soname_or_dll_name,
        dependencies,
        exported_symbols,
        host_dlopen_ok: None,
        host_dlopen_error: None,
    })
}

fn section_slice(bytes: &[u8], sec: Elf64SectionHeader) -> Option<&[u8]> {
    let start = usize::try_from(sec.sh_offset).ok()?;
    let size = usize::try_from(sec.sh_size).ok()?;
    let end = start.checked_add(size)?;
    bytes.get(start..end)
}

// ============================================================================
// PE32+ Parser
// ============================================================================

const PE32_PLUS_MAGIC: u16 = 0x020B;
const PE_MACHINE_AMD64: u16 = 0x8664;
const PE_MACHINE_ARM64: u16 = 0xAA64;

#[derive(Debug, Clone, Copy)]
struct PeSection {
    virtual_size: u32,
    virtual_address: u32,
    size_of_raw_data: u32,
    pointer_to_raw_data: u32,
}

fn pe_rva_to_offset(rva: u32, sections: &[PeSection], file_len: usize) -> Option<usize> {
    for sec in sections {
        let span = sec.virtual_size.max(sec.size_of_raw_data);
        if rva >= sec.virtual_address && rva < sec.virtual_address.saturating_add(span) {
            let delta = rva - sec.virtual_address;
            if delta < sec.size_of_raw_data {
                let off = (sec.pointer_to_raw_data as usize).checked_add(delta as usize)?;
                if off < file_len {
                    return Some(off);
                }
            }
            return None;
        }
    }
    // Fallback for RVAs inside the PE header itself (before first section VA).
    let first_va = sections
        .iter()
        .map(|s| s.virtual_address)
        .min()
        .unwrap_or(0);
    if rva < first_va && (rva as usize) < file_len {
        return Some(rva as usize);
    }
    None
}

fn parse_pe32_plus(
    path_label: &str,
    bytes: &[u8],
    blake3_hex: String,
    size_bytes: u64,
) -> Result<ParsedBinary, String> {
    let e_lfanew = read_u32_le(bytes, 0x3C)
        .ok_or_else(|| format!("PE file '{path_label}' missing e_lfanew"))?
        as usize;

    if e_lfanew.saturating_add(24) > bytes.len() || &bytes[e_lfanew..e_lfanew + 4] != b"PE\0\0" {
        return Err(format!(
            "PE file '{path_label}' does not have valid PE\\0\\0 signature at offset 0x{e_lfanew:x}"
        ));
    }

    let coff_off = e_lfanew + 4;
    let machine = read_u16_le(bytes, coff_off)
        .ok_or_else(|| format!("PE file '{path_label}' missing COFF Machine"))?;
    let num_sections = read_u16_le(bytes, coff_off + 2)
        .ok_or_else(|| format!("PE file '{path_label}' missing NumberOfSections"))?
        as usize;
    let size_of_opt_header = read_u16_le(bytes, coff_off + 16)
        .ok_or_else(|| format!("PE file '{path_label}' missing SizeOfOptionalHeader"))?
        as usize;

    let architecture = match machine {
        PE_MACHINE_AMD64 => "x86_64".to_string(),
        PE_MACHINE_ARM64 => "aarch64".to_string(),
        0x014C => "x86".to_string(),
        other => format!("pe_machine_0x{other:04x}"),
    };

    let opt_off = coff_off + 20;
    if size_of_opt_header < 112 || opt_off.saturating_add(size_of_opt_header) > bytes.len() {
        return Err(format!(
            "PE file '{path_label}' has invalid Optional Header size {size_of_opt_header}"
        ));
    }

    let opt_magic = read_u16_le(bytes, opt_off)
        .ok_or_else(|| format!("PE file '{path_label}' missing Optional Header magic"))?;
    if opt_magic != PE32_PLUS_MAGIC {
        return Err(format!(
            "PE file '{path_label}' has Optional Header magic 0x{opt_magic:04x} (expected PE32+ 0x020B)"
        ));
    }

    let num_rva_and_sizes = read_u32_le(bytes, opt_off + 108).unwrap_or(0) as usize;
    let data_dir_off = opt_off + 112;

    // Parse Section Table
    let sec_table_off = opt_off + size_of_opt_header;
    let mut sections = Vec::with_capacity(num_sections);
    for idx in 0..num_sections {
        let off = sec_table_off.saturating_add(idx * 40);
        if off.saturating_add(40) > bytes.len() {
            break;
        }
        sections.push(PeSection {
            virtual_size: read_u32_le(bytes, off + 8).unwrap_or(0),
            virtual_address: read_u32_le(bytes, off + 12).unwrap_or(0),
            size_of_raw_data: read_u32_le(bytes, off + 16).unwrap_or(0),
            pointer_to_raw_data: read_u32_le(bytes, off + 20).unwrap_or(0),
        });
    }

    let mut soname_or_dll_name = None;
    let mut exported_symbols = BTreeSet::new();
    let mut dependencies = Vec::new();

    // DataDirectory[0]: Export Directory
    if num_rva_and_sizes > 0 && data_dir_off + 8 <= opt_off + size_of_opt_header {
        let export_rva = read_u32_le(bytes, data_dir_off).unwrap_or(0);
        let export_size = read_u32_le(bytes, data_dir_off + 4).unwrap_or(0);
        if export_rva != 0 && export_size != 0 {
            if let Some(exp_off) = pe_rva_to_offset(export_rva, &sections, bytes.len()) {
                if exp_off.saturating_add(40) <= bytes.len() {
                    let name_rva = read_u32_le(bytes, exp_off + 12).unwrap_or(0);
                    if name_rva != 0 {
                        if let Some(name_off) = pe_rva_to_offset(name_rva, &sections, bytes.len()) {
                            if let Some(dll_name) = read_c_string(bytes, name_off) {
                                if !dll_name.is_empty() {
                                    soname_or_dll_name = Some(dll_name);
                                }
                            }
                        }
                    }

                    let number_of_names = read_u32_le(bytes, exp_off + 24).unwrap_or(0) as usize;
                    let addr_of_names_rva = read_u32_le(bytes, exp_off + 32).unwrap_or(0);
                    if number_of_names > 0 && addr_of_names_rva != 0 {
                        if let Some(names_off) =
                            pe_rva_to_offset(addr_of_names_rva, &sections, bytes.len())
                        {
                            for i in 0..number_of_names {
                                let entry_off =
                                    match i.checked_mul(4).and_then(|d| names_off.checked_add(d)) {
                                        Some(o) if o + 4 <= bytes.len() => o,
                                        _ => break,
                                    };
                                let sym_rva = read_u32_le(bytes, entry_off).unwrap_or(0);
                                if sym_rva != 0 {
                                    if let Some(sym_off) =
                                        pe_rva_to_offset(sym_rva, &sections, bytes.len())
                                    {
                                        if let Some(sym_name) = read_c_string(bytes, sym_off) {
                                            if !sym_name.is_empty() {
                                                exported_symbols.insert(sym_name);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // DataDirectory[1]: Import Directory
    if num_rva_and_sizes > 1 && data_dir_off + 16 <= opt_off + size_of_opt_header {
        let import_rva = read_u32_le(bytes, data_dir_off + 8).unwrap_or(0);
        let import_size = read_u32_le(bytes, data_dir_off + 12).unwrap_or(0);
        if import_rva != 0 && import_size != 0 {
            if let Some(mut imp_off) = pe_rva_to_offset(import_rva, &sections, bytes.len()) {
                while imp_off.saturating_add(20) <= bytes.len() {
                    let orig_first_thunk = read_u32_le(bytes, imp_off).unwrap_or(0);
                    let time_date_stamp = read_u32_le(bytes, imp_off + 4).unwrap_or(0);
                    let forwarder_chain = read_u32_le(bytes, imp_off + 8).unwrap_or(0);
                    let name_rva = read_u32_le(bytes, imp_off + 12).unwrap_or(0);
                    let first_thunk = read_u32_le(bytes, imp_off + 16).unwrap_or(0);

                    if orig_first_thunk == 0
                        && time_date_stamp == 0
                        && forwarder_chain == 0
                        && name_rva == 0
                        && first_thunk == 0
                    {
                        break;
                    }

                    if name_rva != 0 {
                        if let Some(name_off) = pe_rva_to_offset(name_rva, &sections, bytes.len()) {
                            if let Some(dll_name) = read_c_string(bytes, name_off) {
                                push_unique(&mut dependencies, dll_name);
                            }
                        }
                    }

                    imp_off = imp_off.saturating_add(20);
                }
            }
        }
    }

    Ok(ParsedBinary {
        path: path_label.to_string(),
        format: BinaryFormat::Pe32Plus,
        architecture,
        blake3_hex,
        size_bytes,
        soname_or_dll_name,
        dependencies,
        exported_symbols,
        host_dlopen_ok: None,
        host_dlopen_error: None,
    })
}

// ============================================================================
// Unix `ar` Archive Parser
// ============================================================================

fn parse_unix_ar(
    path_label: &str,
    bytes: &[u8],
    blake3_hex: String,
    size_bytes: u64,
) -> Result<ParsedBinary, String> {
    if bytes.len() < 8 || &bytes[0..8] != b"!<arch>\n" {
        return Err(format!(
            "File '{path_label}' is not a valid Unix ar archive"
        ));
    }

    let mut exported_symbols = BTreeSet::new();
    let mut architecture = "unknown".to_string();
    let mut saw_symtab = false;

    let mut pos = 8usize;
    while pos.saturating_add(60) <= bytes.len() {
        let header = &bytes[pos..pos + 60];
        if &header[58..60] != b"`\n" {
            // Some archives have trailing padding or non-standard endings; stop cleanly.
            break;
        }

        let raw_name = String::from_utf8_lossy(&header[0..16]);
        let name_trimmed = raw_name.trim();
        let size_str = String::from_utf8_lossy(&header[48..58]);
        let member_size = match size_str.trim().parse::<usize>() {
            Ok(s) => s,
            Err(_) => break,
        };

        let data_start = pos + 60;
        let Some(data_end) = data_start.checked_add(member_size) else {
            break;
        };
        if data_end > bytes.len() {
            break;
        }
        let member_data = &bytes[data_start..data_end];

        // Resolve BSD extended name (#1/<len>) if present.
        let (effective_name, effective_data) =
            if let Some(len_str) = name_trimmed.strip_prefix("#1/") {
                if let Ok(name_len) = len_str.parse::<usize>() {
                    if name_len <= member_data.len() {
                        let ext_name = String::from_utf8_lossy(&member_data[..name_len])
                            .trim_end_matches('\0')
                            .to_string();
                        (ext_name, &member_data[name_len..])
                    } else {
                        (name_trimmed.to_string(), member_data)
                    }
                } else {
                    (name_trimmed.to_string(), member_data)
                }
            } else {
                (name_trimmed.to_string(), member_data)
            };

        if effective_name == "/" {
            // SysV / GNU 32-bit symbol table
            saw_symtab = true;
            parse_ar_sysv_symtab_32(effective_data, &mut exported_symbols);
        } else if effective_name == "/SYM64/" {
            // SysV / GNU 64-bit symbol table
            saw_symtab = true;
            parse_ar_sysv_symtab_64(effective_data, &mut exported_symbols);
        } else if effective_name.starts_with("__.SYMDEF") {
            // BSD symbol table
            saw_symtab = true;
            parse_ar_bsd_symdef(effective_data, &mut exported_symbols);
        } else if effective_data.len() >= 64
            && &effective_data[0..4] == b"\x7fELF"
            && effective_data[4] == ELFCLASS64
            && effective_data[5] == ELFDATA2LSB
        {
            if architecture == "unknown" {
                if let Some(e_machine) = read_u16_le(effective_data, 18) {
                    architecture = elf_machine_name(e_machine);
                }
            }
            if !saw_symtab {
                // Fallback: extract defined global/weak symbols directly from member object
                if let Ok(member_parsed) = parse_elf64(
                    &effective_name,
                    effective_data,
                    String::new(),
                    effective_data.len() as u64,
                ) {
                    exported_symbols.extend(member_parsed.exported_symbols);
                }
            }
        }

        // Members are 2-byte aligned
        pos = data_end + (member_size % 2);
    }

    Ok(ParsedBinary {
        path: path_label.to_string(),
        format: BinaryFormat::UnixAr,
        architecture,
        blake3_hex,
        size_bytes,
        soname_or_dll_name: None,
        dependencies: Vec::new(),
        exported_symbols,
        host_dlopen_ok: None,
        host_dlopen_error: None,
    })
}

fn parse_ar_sysv_symtab_32(data: &[u8], out: &mut BTreeSet<String>) {
    let Some(count) = read_u32_be(data, 0) else {
        return;
    };
    let count = count as usize;
    let Some(str_start) = count.checked_mul(4).and_then(|n| n.checked_add(4)) else {
        return;
    };
    if str_start >= data.len() {
        return;
    }
    let mut cursor = str_start;
    for _ in 0..count {
        if cursor >= data.len() {
            break;
        }
        let end = data[cursor..]
            .iter()
            .position(|&b| b == 0)
            .map(|rel| cursor + rel)
            .unwrap_or(data.len());
        let sym = String::from_utf8_lossy(&data[cursor..end]).into_owned();
        if !sym.is_empty() {
            out.insert(sym);
        }
        cursor = end.saturating_add(1);
    }
}

fn parse_ar_sysv_symtab_64(data: &[u8], out: &mut BTreeSet<String>) {
    let Some(count) = read_u64_be(data, 0) else {
        return;
    };
    let Ok(count) = usize::try_from(count) else {
        return;
    };
    let Some(str_start) = count.checked_mul(8).and_then(|n| n.checked_add(8)) else {
        return;
    };
    if str_start >= data.len() {
        return;
    }
    let mut cursor = str_start;
    for _ in 0..count {
        if cursor >= data.len() {
            break;
        }
        let end = data[cursor..]
            .iter()
            .position(|&b| b == 0)
            .map(|rel| cursor + rel)
            .unwrap_or(data.len());
        let sym = String::from_utf8_lossy(&data[cursor..end]).into_owned();
        if !sym.is_empty() {
            out.insert(sym);
        }
        cursor = end.saturating_add(1);
    }
}

fn parse_ar_bsd_symdef(data: &[u8], out: &mut BTreeSet<String>) {
    // BSD __.SYMDEF format (32-bit little-endian):
    // [u32 ranlib_byte_len] [ranlib entries: (u32 strx, u32 off) * N] [u32 strtab_len] [strtab bytes]
    let Some(ranlib_bytes) = read_u32_le(data, 0) else {
        return;
    };
    let ranlib_bytes = ranlib_bytes as usize;
    let Some(strtab_len_off) = 4usize.checked_add(ranlib_bytes) else {
        return;
    };
    let Some(strtab_len) = read_u32_le(data, strtab_len_off) else {
        return;
    };
    let strtab_start = strtab_len_off + 4;
    let Some(strtab_end) = strtab_start.checked_add(strtab_len as usize) else {
        return;
    };
    if strtab_end > data.len() {
        return;
    }
    let strtab = &data[strtab_start..strtab_end];
    let num_entries = ranlib_bytes / 8;
    for i in 0..num_entries {
        let entry_off = 4 + i * 8;
        if let Some(strx) = read_u32_le(data, entry_off) {
            if let Some(sym) = read_c_string(strtab, strx as usize) {
                if !sym.is_empty() {
                    out.insert(sym);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_minimal_elf64_so() -> Vec<u8> {
        // Layout:
        // 0x000..0x040: ELF64 header (64 bytes)
        // 0x040..0x080: .dynstr (64 bytes)
        // 0x080..0x0E0: .dynsym (4 entries * 24 = 96 bytes)
        // 0x0E0..0x120: .dynamic (4 entries * 16 = 64 bytes)
        // 0x120..0x220: Section headers (4 sections * 64 = 256 bytes)
        let mut buf = vec![0u8; 0x220];

        // ELF64 header
        buf[0..4].copy_from_slice(b"\x7fELF");
        buf[4] = ELFCLASS64;
        buf[5] = ELFDATA2LSB;
        buf[6] = 1; // EV_CURRENT
        buf[16..18].copy_from_slice(&3u16.to_le_bytes()); // ET_DYN
        buf[18..20].copy_from_slice(&62u16.to_le_bytes()); // EM_X86_64
        buf[40..48].copy_from_slice(&0x120u64.to_le_bytes()); // e_shoff
        buf[58..60].copy_from_slice(&64u16.to_le_bytes()); // e_shentsize
        buf[60..62].copy_from_slice(&4u16.to_le_bytes()); // e_shnum

        // .dynstr at 0x040
        // 0: "\0"
        // 1: "libtest.so.1\0" (13 bytes -> next 14)
        // 14: "libc.so.6\0" (10 bytes -> next 24)
        // 24: "exported_global\0" (16 bytes -> next 40)
        // 40: "exported_weak\0" (14 bytes -> next 54)
        // 54: "undef_sym\0" (10 bytes -> next 64)
        let dynstr = b"\0libtest.so.1\0libc.so.6\0exported_global\0exported_weak\0undef_sym\0";
        buf[0x040..0x040 + dynstr.len()].copy_from_slice(dynstr);

        // .dynsym at 0x080: 4 symbols (null, global defined, weak defined, global undefined)
        // Entry 1: exported_global (offset 24), STB_GLOBAL(1)<<4 | STT_FUNC(2), shndx=1
        let s1 = 0x080 + 24;
        buf[s1..s1 + 4].copy_from_slice(&24u32.to_le_bytes());
        buf[s1 + 4] = (STB_GLOBAL << 4) | 2;
        buf[s1 + 6..s1 + 8].copy_from_slice(&1u16.to_le_bytes());

        // Entry 2: exported_weak (offset 40), STB_WEAK(2)<<4 | STT_FUNC(2), shndx=1
        let s2 = 0x080 + 48;
        buf[s2..s2 + 4].copy_from_slice(&40u32.to_le_bytes());
        buf[s2 + 4] = (STB_WEAK << 4) | 2;
        buf[s2 + 6..s2 + 8].copy_from_slice(&1u16.to_le_bytes());

        // Entry 3: undef_sym (offset 54), STB_GLOBAL(1)<<4 | STT_FUNC(2), shndx=0 (UNDEF)
        let s3 = 0x080 + 72;
        buf[s3..s3 + 4].copy_from_slice(&54u32.to_le_bytes());
        buf[s3 + 4] = (STB_GLOBAL << 4) | 2;
        buf[s3 + 6..s3 + 8].copy_from_slice(&0u16.to_le_bytes());

        // .dynamic at 0x0E0
        // Entry 0: DT_NEEDED -> 14 ("libc.so.6")
        buf[0x0E0..0x0E8].copy_from_slice(&DT_NEEDED.to_le_bytes());
        buf[0x0E8..0x0F0].copy_from_slice(&14u64.to_le_bytes());
        // Entry 1: DT_SONAME -> 1 ("libtest.so.1")
        buf[0x0F0..0x0F8].copy_from_slice(&DT_SONAME.to_le_bytes());
        buf[0x0F8..0x100].copy_from_slice(&1u64.to_le_bytes());
        // Entry 2: DT_NULL
        buf[0x100..0x108].copy_from_slice(&DT_NULL.to_le_bytes());

        // Section headers at 0x120
        // Sec 0: SHT_NULL
        // Sec 1: SHT_STRTAB (3) at 0x040, size 64
        let sh1 = 0x120 + 64;
        buf[sh1 + 4..sh1 + 8].copy_from_slice(&3u32.to_le_bytes());
        buf[sh1 + 24..sh1 + 32].copy_from_slice(&0x040u64.to_le_bytes());
        buf[sh1 + 32..sh1 + 40].copy_from_slice(&64u64.to_le_bytes());

        // Sec 2: SHT_DYNSYM (11) at 0x080, size 96, sh_link = 1, sh_entsize = 24
        let sh2 = 0x120 + 128;
        buf[sh2 + 4..sh2 + 8].copy_from_slice(&SHT_DYNSYM.to_le_bytes());
        buf[sh2 + 24..sh2 + 32].copy_from_slice(&0x080u64.to_le_bytes());
        buf[sh2 + 32..sh2 + 40].copy_from_slice(&96u64.to_le_bytes());
        buf[sh2 + 40..sh2 + 44].copy_from_slice(&1u32.to_le_bytes());
        buf[sh2 + 56..sh2 + 64].copy_from_slice(&24u64.to_le_bytes());

        // Sec 3: SHT_DYNAMIC (6) at 0x0E0, size 64, sh_link = 1, sh_entsize = 16
        let sh3 = 0x120 + 192;
        buf[sh3 + 4..sh3 + 8].copy_from_slice(&SHT_DYNAMIC.to_le_bytes());
        buf[sh3 + 24..sh3 + 32].copy_from_slice(&0x0E0u64.to_le_bytes());
        buf[sh3 + 32..sh3 + 40].copy_from_slice(&64u64.to_le_bytes());
        buf[sh3 + 40..sh3 + 44].copy_from_slice(&1u32.to_le_bytes());
        buf[sh3 + 56..sh3 + 64].copy_from_slice(&16u64.to_le_bytes());

        buf
    }

    fn build_minimal_pe32_plus_dll() -> Vec<u8> {
        // DOS header + PE\0\0 at 0x80 + COFF (20) + Optional PE32+ (240) + 1 Section (40)
        // Section .rdata: VA=0x1000, RawPtr=0x200, Size=0x200
        let mut buf = vec![0u8; 0x400];
        buf[0..2].copy_from_slice(b"MZ");
        buf[0x3C..0x40].copy_from_slice(&0x80u32.to_le_bytes());

        buf[0x80..0x84].copy_from_slice(b"PE\0\0");
        let coff = 0x84;
        buf[coff..coff + 2].copy_from_slice(&PE_MACHINE_AMD64.to_le_bytes());
        buf[coff + 2..coff + 4].copy_from_slice(&1u16.to_le_bytes()); // 1 section
        buf[coff + 16..coff + 18].copy_from_slice(&240u16.to_le_bytes()); // SizeOfOptionalHeader

        let opt = coff + 20;
        buf[opt..opt + 2].copy_from_slice(&PE32_PLUS_MAGIC.to_le_bytes());
        buf[opt + 108..opt + 112].copy_from_slice(&16u32.to_le_bytes()); // NumberOfRvaAndSizes

        // DataDirectory[0]: Export Directory at RVA 0x1000, size 0x80
        let dd = opt + 112;
        buf[dd..dd + 4].copy_from_slice(&0x1000u32.to_le_bytes());
        buf[dd + 4..dd + 8].copy_from_slice(&0x80u32.to_le_bytes());
        // DataDirectory[1]: Import Directory at RVA 0x1080, size 0x40
        buf[dd + 8..dd + 12].copy_from_slice(&0x1080u32.to_le_bytes());
        buf[dd + 12..dd + 16].copy_from_slice(&0x40u32.to_le_bytes());

        // Section 0 at opt + 240
        let sec = opt + 240;
        buf[sec..sec + 6].copy_from_slice(b".rdata");
        buf[sec + 8..sec + 12].copy_from_slice(&0x200u32.to_le_bytes()); // VirtualSize
        buf[sec + 12..sec + 16].copy_from_slice(&0x1000u32.to_le_bytes()); // VirtualAddress
        buf[sec + 16..sec + 20].copy_from_slice(&0x200u32.to_le_bytes()); // SizeOfRawData
        buf[sec + 20..sec + 24].copy_from_slice(&0x200u32.to_le_bytes()); // PointerToRawData

        // Export Directory at raw 0x200 (RVA 0x1000)
        // Name RVA = 0x1040 ("sample_sdk.dll")
        // NumberOfNames = 2
        // AddressOfNames RVA = 0x1030 -> [0x1050 ("SdkInit"), 0x1060 ("SdkShutdown")]
        buf[0x200 + 12..0x200 + 16].copy_from_slice(&0x1040u32.to_le_bytes());
        buf[0x200 + 24..0x200 + 28].copy_from_slice(&2u32.to_le_bytes());
        buf[0x200 + 32..0x200 + 36].copy_from_slice(&0x1030u32.to_le_bytes());

        buf[0x230..0x234].copy_from_slice(&0x1050u32.to_le_bytes());
        buf[0x234..0x238].copy_from_slice(&0x1060u32.to_le_bytes());
        buf[0x240..0x240 + 15].copy_from_slice(b"sample_sdk.dll\0");
        buf[0x250..0x250 + 8].copy_from_slice(b"SdkInit\0");
        buf[0x260..0x260 + 12].copy_from_slice(b"SdkShutdown\0");

        // Import Directory at raw 0x280 (RVA 0x1080)
        // Descriptor 0: Name RVA = 0x10C0 ("KERNEL32.dll")
        buf[0x280 + 12..0x280 + 16].copy_from_slice(&0x10C0u32.to_le_bytes());
        buf[0x2C0..0x2C0 + 13].copy_from_slice(b"KERNEL32.dll\0");

        buf
    }

    fn build_minimal_unix_ar() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"!<arch>\n");

        // SysV "/" symbol table with 2 symbols
        let mut symtab = Vec::new();
        symtab.extend_from_slice(&2u32.to_be_bytes());
        symtab.extend_from_slice(&0u32.to_be_bytes());
        symtab.extend_from_slice(&0u32.to_be_bytes());
        symtab.extend_from_slice(b"NVSDK_NGX_VULKAN_Init\0");
        symtab.extend_from_slice(b"NVSDK_NGX_VULKAN_Shutdown\0");

        let header = format!(
            "{:<16}{:<12}{:<6}{:<6}{:<8}{:<10}`\n",
            "/",
            "0",
            "0",
            "0",
            "0",
            symtab.len()
        );
        assert_eq!(header.len(), 60);
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&symtab);
        if symtab.len() % 2 == 1 {
            buf.push(b'\n');
        }
        buf
    }

    #[test]
    fn test_parse_elf64() {
        let data = build_minimal_elf64_so();
        let parsed = parse_binary_bytes("libtest.so.1", &data).unwrap();
        assert_eq!(parsed.format, BinaryFormat::Elf64);
        assert_eq!(parsed.architecture, "x86_64");
        assert_eq!(parsed.soname_or_dll_name.as_deref(), Some("libtest.so.1"));
        assert_eq!(parsed.dependencies, vec!["libc.so.6".to_string()]);
        assert!(parsed.exported_symbols.contains("exported_global"));
        assert!(parsed.exported_symbols.contains("exported_weak"));
        assert!(!parsed.exported_symbols.contains("undef_sym"));
    }

    #[test]
    fn test_parse_pe32_plus() {
        let data = build_minimal_pe32_plus_dll();
        let parsed = parse_binary_bytes("sample_sdk.dll", &data).unwrap();
        assert_eq!(parsed.format, BinaryFormat::Pe32Plus);
        assert_eq!(parsed.architecture, "x86_64");
        assert_eq!(parsed.soname_or_dll_name.as_deref(), Some("sample_sdk.dll"));
        assert_eq!(parsed.dependencies, vec!["KERNEL32.dll".to_string()]);
        assert!(parsed.exported_symbols.contains("SdkInit"));
        assert!(parsed.exported_symbols.contains("SdkShutdown"));
    }

    #[test]
    fn test_parse_unix_ar() {
        let data = build_minimal_unix_ar();
        let parsed = parse_binary_bytes("libnvsdk_ngx.a", &data).unwrap();
        assert_eq!(parsed.format, BinaryFormat::UnixAr);
        assert!(parsed.exported_symbols.contains("NVSDK_NGX_VULKAN_Init"));
        assert!(
            parsed
                .exported_symbols
                .contains("NVSDK_NGX_VULKAN_Shutdown")
        );
    }
}
