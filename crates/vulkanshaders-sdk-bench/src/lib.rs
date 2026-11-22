//! Cross-verification bench for VulkanShaders vendor SDK ABIs, FFI/FFM parity,
//! capability bitmasks, GLSL shader contracts, and Fabric/NeoForge mixin guards.

pub mod binary_parser;
pub mod downloader;
pub mod header_parser;
pub mod source_verifier;

pub use binary_parser::{BinaryFormat, ParsedBinary, parse_binary_bytes, parse_binary_file};
pub use downloader::{
    DownloadConfig, DownloadedArtifact, SdkDownloadResult, SdkKind, fetch_sdk_artifacts,
};
pub use header_parser::{ParsedHeader, parse_c_header, parse_c_header_file};
pub use source_verifier::{
    BenchReport, CheckResult, CheckStatus, SuiteReport, VerifyOptions, run_all_suites,
    run_offline_source_suites,
};
