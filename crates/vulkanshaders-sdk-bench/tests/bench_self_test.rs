//! Self-tests for `vulkanshaders-sdk-bench`:
//! 1. Minimal valid in-memory ELF64 `.so` parsing (exports + DT_NEEDED)
//! 2. Minimal valid in-memory PE32+ `.dll` parsing (Export Directory)
//! 3. C-header parser verification (#define strings/hex/shifts, enums, structs, prototypes)
//! 4. Live VulkanShaders repository offline verification suites (0 failures)

use std::path::PathBuf;
use vulkanshaders_sdk_bench::binary_parser::{BinaryFormat, parse_binary_bytes};
use vulkanshaders_sdk_bench::header_parser::parse_c_header;
use vulkanshaders_sdk_bench::source_verifier::{
    CheckStatus, VerifyOptions, run_offline_source_suites, verify_sdk_binary_symbols,
};

fn push_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn align_up(val: usize, align: usize) -> usize {
    (val + align - 1) & !(align - 1)
}

/// Builds a minimal, spec-compliant ELF64 little-endian shared object (`.so`) containing
/// `.dynstr`, `.dynsym`, `.dynamic`, and `.shstrtab`.
fn build_minimal_elf64_so(exports: &[&str], needed: &[&str]) -> Vec<u8> {
    // 1. Build .dynstr
    let mut dynstr = vec![0u8];
    let mut export_offsets = Vec::new();
    for &sym in exports {
        export_offsets.push(dynstr.len() as u32);
        dynstr.extend_from_slice(sym.as_bytes());
        dynstr.push(0);
    }
    let mut needed_offsets = Vec::new();
    for &lib in needed {
        needed_offsets.push(dynstr.len() as u32);
        dynstr.extend_from_slice(lib.as_bytes());
        dynstr.push(0);
    }

    // 2. Build .dynsym (24 bytes per Elf64_Sym)
    let mut dynsym = Vec::new();
    // Entry 0: STN_UNDEF
    dynsym.resize(24, 0u8);
    for &name_off in &export_offsets {
        push_u32(&mut dynsym, name_off); // st_name
        dynsym.push((1u8 << 4) | 2u8); // st_info: STB_GLOBAL (1) | STT_FUNC (2)
        dynsym.push(0); // st_other: STV_DEFAULT
        push_u16(&mut dynsym, 1); // st_shndx: non-zero defined section
        push_u64(&mut dynsym, 0x1000); // st_value
        push_u64(&mut dynsym, 16); // st_size
    }

    // 3. Build .dynamic (16 bytes per Elf64_Dyn)
    let mut dynamic = Vec::new();
    for &lib_off in &needed_offsets {
        push_u64(&mut dynamic, 1); // DT_NEEDED
        push_u64(&mut dynamic, lib_off as u64);
    }
    push_u64(&mut dynamic, 0); // DT_NULL
    push_u64(&mut dynamic, 0);

    // 4. Build .shstrtab
    let mut shstrtab = vec![0u8];
    let shname_dynstr = shstrtab.len() as u32;
    shstrtab.extend_from_slice(b".dynstr\0");
    let shname_dynsym = shstrtab.len() as u32;
    shstrtab.extend_from_slice(b".dynsym\0");
    let shname_dynamic = shstrtab.len() as u32;
    shstrtab.extend_from_slice(b".dynamic\0");
    let shname_shstrtab = shstrtab.len() as u32;
    shstrtab.extend_from_slice(b".shstrtab\0");

    // Layout: [ELF64 header (64B)] [.dynstr] [.dynsym] [.dynamic] [.shstrtab] [Section Headers (5 * 64B)]
    let off_dynstr = 64usize;
    let off_dynsym = align_up(off_dynstr + dynstr.len(), 8);
    let off_dynamic = align_up(off_dynsym + dynsym.len(), 8);
    let off_shstrtab = align_up(off_dynamic + dynamic.len(), 8);
    let off_shdrs = align_up(off_shstrtab + shstrtab.len(), 8);

    let mut elf = Vec::new();
    // e_ident (16 bytes)
    elf.extend_from_slice(&[
        0x7F, b'E', b'L', b'F', // magic
        2,    // ELFCLASS64
        1,    // ELFDATA2LSB
        1,    // EV_CURRENT
        0,    // ELFOSABI_NONE
        0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    push_u16(&mut elf, 3); // e_type = ET_DYN
    push_u16(&mut elf, 62); // e_machine = EM_X86_64
    push_u32(&mut elf, 1); // e_version = EV_CURRENT
    push_u64(&mut elf, 0); // e_entry
    push_u64(&mut elf, 0); // e_phoff
    push_u64(&mut elf, off_shdrs as u64); // e_shoff
    push_u32(&mut elf, 0); // e_flags
    push_u16(&mut elf, 64); // e_ehsize
    push_u16(&mut elf, 0); // e_phentsize
    push_u16(&mut elf, 0); // e_phnum
    push_u16(&mut elf, 64); // e_shentsize
    push_u16(&mut elf, 5); // e_shnum
    push_u16(&mut elf, 4); // e_shstrndx = .shstrtab
    assert_eq!(elf.len(), 64);

    elf.extend_from_slice(&dynstr);
    elf.resize(off_dynsym, 0);
    elf.extend_from_slice(&dynsym);
    elf.resize(off_dynamic, 0);
    elf.extend_from_slice(&dynamic);
    elf.resize(off_shstrtab, 0);
    elf.extend_from_slice(&shstrtab);
    elf.resize(off_shdrs, 0);

    let write_shdr = |buf: &mut Vec<u8>,
                      sh_name: u32,
                      sh_type: u32,
                      sh_flags: u64,
                      sh_offset: usize,
                      sh_size: usize,
                      sh_link: u32,
                      sh_info: u32,
                      sh_addralign: u64,
                      sh_entsize: u64| {
        push_u32(buf, sh_name);
        push_u32(buf, sh_type);
        push_u64(buf, sh_flags);
        push_u64(buf, 0); // sh_addr
        push_u64(buf, sh_offset as u64);
        push_u64(buf, sh_size as u64);
        push_u32(buf, sh_link);
        push_u32(buf, sh_info);
        push_u64(buf, sh_addralign);
        push_u64(buf, sh_entsize);
    };

    // [0] SHT_NULL
    write_shdr(&mut elf, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    // [1] .dynstr (SHT_STRTAB = 3)
    write_shdr(
        &mut elf,
        shname_dynstr,
        3,
        2,
        off_dynstr,
        dynstr.len(),
        0,
        0,
        1,
        0,
    );
    // [2] .dynsym (SHT_DYNSYM = 11, sh_link = 1)
    write_shdr(
        &mut elf,
        shname_dynsym,
        11,
        2,
        off_dynsym,
        dynsym.len(),
        1,
        1,
        8,
        24,
    );
    // [3] .dynamic (SHT_DYNAMIC = 6, sh_link = 1)
    write_shdr(
        &mut elf,
        shname_dynamic,
        6,
        3,
        off_dynamic,
        dynamic.len(),
        1,
        0,
        8,
        16,
    );
    // [4] .shstrtab (SHT_STRTAB = 3)
    write_shdr(
        &mut elf,
        shname_shstrtab,
        3,
        0,
        off_shstrtab,
        shstrtab.len(),
        0,
        0,
        1,
        0,
    );

    elf
}

/// Builds a minimal, spec-compliant PE32+ (`x86_64`) DLL in memory with an Export Directory
/// in `.edata` (RVA `0x1000`, raw file offset `0x200`).
fn build_minimal_pe64_dll(dll_name: &str, exports: &[&str]) -> Vec<u8> {
    const PE_OFFSET: u32 = 0x80;
    const EDATA_RVA: u32 = 0x1000;
    const EDATA_RAW: usize = 0x200;

    let n = exports.len() as u32;
    let mut edata = Vec::new();
    // Reserve 40 bytes for IMAGE_EXPORT_DIRECTORY
    edata.resize(40, 0u8);

    // AddressOfFunctions (n * u32)
    let eat_rva = EDATA_RVA + edata.len() as u32;
    for i in 0..n {
        push_u32(&mut edata, 0x2000 + i * 0x10);
    }

    // AddressOfNameOrdinals (n * u16)
    let ord_rva = EDATA_RVA + edata.len() as u32;
    for i in 0..n {
        push_u16(&mut edata, i as u16);
    }
    while edata.len() % 4 != 0 {
        edata.push(0);
    }

    // Reserve AddressOfNames (n * u32)
    let names_table_off = edata.len();
    let names_rva = EDATA_RVA + names_table_off as u32;
    edata.resize(names_table_off + (n as usize) * 4, 0u8);

    // DLL name string
    let dll_name_rva = EDATA_RVA + edata.len() as u32;
    edata.extend_from_slice(dll_name.as_bytes());
    edata.push(0);

    // Export name strings + patch AddressOfNames table
    for (idx, &sym) in exports.iter().enumerate() {
        let sym_rva = EDATA_RVA + edata.len() as u32;
        edata.extend_from_slice(sym.as_bytes());
        edata.push(0);
        let slot = names_table_off + idx * 4;
        edata[slot..slot + 4].copy_from_slice(&sym_rva.to_le_bytes());
    }

    let edata_size = edata.len() as u32;
    // Patch IMAGE_EXPORT_DIRECTORY fields
    edata[12..16].copy_from_slice(&dll_name_rva.to_le_bytes()); // Name
    edata[16..20].copy_from_slice(&1u32.to_le_bytes()); // Base
    edata[20..24].copy_from_slice(&n.to_le_bytes()); // NumberOfFunctions
    edata[24..28].copy_from_slice(&n.to_le_bytes()); // NumberOfNames
    edata[28..32].copy_from_slice(&eat_rva.to_le_bytes()); // AddressOfFunctions
    edata[32..36].copy_from_slice(&names_rva.to_le_bytes()); // AddressOfNames
    edata[36..40].copy_from_slice(&ord_rva.to_le_bytes()); // AddressOfNameOrdinals

    let edata_raw_size = align_up(edata.len(), 0x200);
    edata.resize(edata_raw_size, 0u8);

    // Assemble PE image
    let mut pe = vec![0u8; EDATA_RAW];
    // DOS header: MZ + e_lfanew
    pe[0] = b'M';
    pe[1] = b'Z';
    pe[0x3C..0x40].copy_from_slice(&PE_OFFSET.to_le_bytes());

    let mut cur = PE_OFFSET as usize;
    // PE\0\0 signature
    pe[cur..cur + 4].copy_from_slice(b"PE\0\0");
    cur += 4;

    // COFF File Header (20 bytes)
    pe[cur..cur + 2].copy_from_slice(&0x8664u16.to_le_bytes()); // Machine = AMD64
    pe[cur + 2..cur + 4].copy_from_slice(&1u16.to_le_bytes()); // NumberOfSections = 1
    pe[cur + 16..cur + 18].copy_from_slice(&240u16.to_le_bytes()); // SizeOfOptionalHeader = 240
    pe[cur + 18..cur + 20].copy_from_slice(&0x2022u16.to_le_bytes()); // Characteristics = DLL | EXECUTABLE
    cur += 20;

    // Optional Header PE32+ (240 bytes)
    pe[cur..cur + 2].copy_from_slice(&0x020Bu16.to_le_bytes()); // Magic = PE32+
    pe[cur + 108..cur + 112].copy_from_slice(&16u32.to_le_bytes()); // NumberOfRvaAndSizes = 16
    // DataDirectory[0] = Export Table (at optional_header + 112)
    pe[cur + 112..cur + 116].copy_from_slice(&EDATA_RVA.to_le_bytes());
    pe[cur + 116..cur + 120].copy_from_slice(&edata_size.to_le_bytes());
    cur += 240;

    // Section Header 0: .edata (40 bytes)
    pe[cur..cur + 8].copy_from_slice(b".edata\0\0");
    pe[cur + 8..cur + 12].copy_from_slice(&edata_size.to_le_bytes()); // VirtualSize
    pe[cur + 12..cur + 16].copy_from_slice(&EDATA_RVA.to_le_bytes()); // VirtualAddress
    pe[cur + 16..cur + 20].copy_from_slice(&(edata_raw_size as u32).to_le_bytes()); // SizeOfRawData
    pe[cur + 20..cur + 24].copy_from_slice(&(EDATA_RAW as u32).to_le_bytes()); // PointerToRawData
    pe[cur + 36..cur + 40].copy_from_slice(&0x4000_0040u32.to_le_bytes()); // INITIALIZED_DATA | MEM_READ

    pe.extend_from_slice(&edata);
    pe
}

#[test]
fn parse_minimal_elf64_so_extracts_exports_and_dt_needed() {
    let bytes = build_minimal_elf64_so(
        &["NVSDK_NGX_VULKAN_Init_Ext2", "CreateInstance"],
        &["libvulkan.so.1"],
    );
    let parsed =
        parse_binary_bytes("test_elf64.so", &bytes).expect("minimal ELF64 .so must parse cleanly");
    assert_eq!(parsed.format, BinaryFormat::Elf64);
    assert!(
        parsed
            .exported_symbols
            .iter()
            .any(|s| s == "NVSDK_NGX_VULKAN_Init_Ext2"),
        "must export NVSDK_NGX_VULKAN_Init_Ext2, got: {:?}",
        parsed.exported_symbols
    );
    assert!(
        parsed
            .exported_symbols
            .iter()
            .any(|s| s == "CreateInstance"),
        "must export CreateInstance, got: {:?}",
        parsed.exported_symbols
    );
    assert!(
        parsed.dependencies.iter().any(|d| d == "libvulkan.so.1"),
        "must list DT_NEEDED libvulkan.so.1, got: {:?}",
        parsed.dependencies
    );
}

#[test]
fn parse_minimal_pe64_dll_extracts_export_directory_symbols() {
    let bytes = build_minimal_pe64_dll(
        "mock_vendor_sdk.dll",
        &["ffxCreateContext", "ffxDispatch", "xessVKExecute"],
    );
    let parsed = parse_binary_bytes("test_pe32p.dll", &bytes)
        .expect("minimal PE32+ .dll must parse cleanly");
    assert_eq!(parsed.format, BinaryFormat::Pe32Plus);
    for expected in ["ffxCreateContext", "ffxDispatch", "xessVKExecute"] {
        assert!(
            parsed.exported_symbols.iter().any(|s| s == expected),
            "PE32+ DLL must export {expected}, got: {:?}",
            parsed.exported_symbols
        );
    }
}

#[test]
fn parse_c_header_extracts_defines_enums_structs_and_prototypes() {
    let sample_header = r#"
        #define NVSDK_NGX_VERSION_API 0x00000015
        #define NVSDK_NGX_Parameter_Color "Color"
        #define XESS_INIT_FLAG_INVERTED_DEPTH (1 << 1)

        typedef enum FfxApiUpscaleQualityMode {
            FFX_UPSCALE_QUALITY_MODE_NATIVEAA = 0,
            FFX_UPSCALE_QUALITY_MODE_QUALITY,
            FFX_UPSCALE_QUALITY_MODE_BALANCED = 2,
            FFX_UPSCALE_QUALITY_MODE_PERFORMANCE
        } FfxApiUpscaleQualityMode;

        typedef struct Xess2d {
            uint32_t x;
            uint32_t y;
        } xess_2d_t;

        int ffxCreateContext(void** context, void* desc);
        int xessVKExecute(void* context, uint64_t cmd, const void* params);
    "#;

    let parsed = parse_c_header("sample.h", sample_header);
    assert_eq!(parsed.defines_int.get("NVSDK_NGX_VERSION_API"), Some(&0x15));
    assert_eq!(
        parsed
            .defines_str
            .get("NVSDK_NGX_Parameter_Color")
            .map(String::as_str),
        Some("Color")
    );
    assert_eq!(
        parsed.defines_int.get("XESS_INIT_FLAG_INVERTED_DEPTH"),
        Some(&2)
    );

    assert_eq!(
        parsed.enum_values.get("FFX_UPSCALE_QUALITY_MODE_NATIVEAA"),
        Some(&0)
    );
    assert_eq!(
        parsed.enum_values.get("FFX_UPSCALE_QUALITY_MODE_QUALITY"),
        Some(&1)
    );
    assert_eq!(
        parsed.enum_values.get("FFX_UPSCALE_QUALITY_MODE_BALANCED"),
        Some(&2)
    );
    assert_eq!(
        parsed
            .enum_values
            .get("FFX_UPSCALE_QUALITY_MODE_PERFORMANCE"),
        Some(&3)
    );

    let xess2d_count = parsed
        .struct_field_counts
        .get("Xess2d")
        .or_else(|| parsed.struct_field_counts.get("xess_2d_t"))
        .copied()
        .expect("struct Xess2d must be parsed");
    assert_eq!(xess2d_count, 2, "Xess2d must have 2 fields");

    assert!(parsed.functions.contains_key("ffxCreateContext"));
    assert!(parsed.functions.contains_key("xessVKExecute"));
}

#[test]
fn live_repo_offline_verification_suites_pass_with_zero_failures() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let report = run_offline_source_suites(&repo_root);
    assert_eq!(
        report.total_fail,
        0,
        "offline source suites reported failures:\n{}",
        serde_json::to_string_pretty(&report).unwrap_or_default()
    );
    assert_eq!(report.overall_status, CheckStatus::Pass);

    // Also verify Suite 1 source extraction against live sdk/*.rs has 0 failures.
    let opts = VerifyOptions::offline(repo_root);
    let suite1 = verify_sdk_binary_symbols(&opts);
    assert_eq!(
        suite1.fail_count(),
        0,
        "Suite 1 source symbol extraction had failures: {:?}",
        suite1.checks
    );
}
