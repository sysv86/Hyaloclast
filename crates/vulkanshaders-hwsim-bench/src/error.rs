// SPDX-License-Identifier: GPL-3.0-only
// Additional permissions under GPLv3 Section 7 apply to optional proprietary vendor SDK
// runtime loading; see LICENSE-EXCEPTION.md in the repository root.

//! Domain error hierarchy for `vulkanshaders-hwsim-bench`.

use thiserror::Error;

/// Top-level error type for hardware-free simulation, ABI verification, and SPIR-V auditing.
#[derive(Debug, Error)]
pub enum HwsimError {
    #[error("unknown GPU profile `{0}`")]
    UnknownGpuProfile(String),

    #[error("unknown OS driver model `{0}`")]
    UnknownDriverModel(String),

    #[error("unsupported FSR 4 network override `{requested}` on {gpu}: {reason}")]
    UnsupportedNetworkOverride {
        requested: String,
        gpu: String,
        reason: String,
    },

    #[error("mock Vulkan ICD violation: {0}")]
    MockIcdViolation(String),

    #[error("pNext chain violation: {0}")]
    PNextViolation(String),

    #[error("image layout transition violation: {0}")]
    LayoutViolation(String),

    #[error("FSR 4 / FFX SDK 2.4 contract violation (code {code}): {detail}")]
    FsrContractViolation { code: u32, detail: String },

    #[error("ABI layout mismatch on `{struct_name}`: {detail}")]
    AbiMismatch { struct_name: String, detail: String },

    #[error("SPIR-V parse error in `{source_label}`: {detail}")]
    SpirvParse { source_label: String, detail: String },

    #[error("I/O error for `{path}`: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
}
