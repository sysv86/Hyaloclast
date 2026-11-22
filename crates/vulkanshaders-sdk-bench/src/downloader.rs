use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SdkKind {
    Ngx,
    Nrd,
    Fsr3,
    Xess,
}

impl SdkKind {
    pub fn all() -> &'static [SdkKind] {
        &[SdkKind::Ngx, SdkKind::Nrd, SdkKind::Fsr3, SdkKind::Xess]
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            SdkKind::Ngx => "ngx",
            SdkKind::Nrd => "nrd",
            SdkKind::Fsr3 => "fsr3",
            SdkKind::Xess => "xess",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadedArtifact {
    pub sdk: SdkKind,
    pub kind: String, // "header" | "binary" | "archive"
    pub name: String,
    pub url: String,
    pub local_path: PathBuf,
    pub blake3_hex: String,
    pub size_bytes: u64,
    pub from_cache: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdkDownloadResult {
    pub sdk: SdkKind,
    pub headers: Vec<DownloadedArtifact>,
    pub binaries: Vec<DownloadedArtifact>,
    pub notes: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DownloadConfig {
    pub cache_dir: PathBuf,
    pub force_download: bool,
    pub offline: bool,
    pub include_large_zips: bool,
    pub local_natives_dirs: Vec<PathBuf>,
}

pub fn fetch_sdk_artifacts(sdk: SdkKind, cfg: &DownloadConfig) -> SdkDownloadResult {
    let mut res = SdkDownloadResult {
        sdk,
        headers: Vec::new(),
        binaries: Vec::new(),
        notes: Vec::new(),
        errors: Vec::new(),
    };

    match sdk {
        SdkKind::Ngx => fetch_ngx(cfg, &mut res),
        SdkKind::Nrd => fetch_nrd(cfg, &mut res),
        SdkKind::Fsr3 => fetch_fsr3(cfg, &mut res),
        SdkKind::Xess => fetch_xess(cfg, &mut res),
    }

    scan_local_natives(sdk, cfg, &mut res);
    res
}

// ============================================================================
// Per-SDK Fetchers
// ============================================================================

fn fetch_ngx(cfg: &DownloadConfig, res: &mut SdkDownloadResult) {
    let headers_dir = cfg.cache_dir.join("ngx").join("headers");
    let binaries_dir = cfg.cache_dir.join("ngx").join("binaries");

    let header_urls = [
        (
            "nvsdk_ngx_vk.h",
            "https://raw.githubusercontent.com/NVIDIA/DLSS/main/include/nvsdk_ngx_vk.h",
        ),
        (
            "nvsdk_ngx_defs.h",
            "https://raw.githubusercontent.com/NVIDIA/DLSS/main/include/nvsdk_ngx_defs.h",
        ),
        (
            "nvsdk_ngx_params.h",
            "https://raw.githubusercontent.com/NVIDIA/DLSS/main/include/nvsdk_ngx_params.h",
        ),
        (
            "nvsdk_ngx_helpers_vk.h",
            "https://raw.githubusercontent.com/NVIDIA/DLSS/main/include/nvsdk_ngx_helpers_vk.h",
        ),
    ];

    for (name, url) in header_urls {
        fetch_single(
            SdkKind::Ngx,
            "header",
            name,
            url,
            &headers_dir.join(name),
            cfg,
            &mut res.headers,
            &mut res.notes,
            &mut res.errors,
        );
    }

    let static_binaries = [
        (
            "nvngx_dlss.dll",
            "https://github.com/NVIDIA/DLSS/raw/main/lib/Windows_x86_64/rel/nvngx_dlss.dll",
        ),
        (
            "libnvsdk_ngx.a",
            "https://github.com/NVIDIA/DLSS/raw/main/lib/Linux_x86_64/libnvsdk_ngx.a",
        ),
    ];

    for (name, url) in static_binaries {
        fetch_single(
            SdkKind::Ngx,
            "binary",
            name,
            url,
            &binaries_dir.join(name),
            cfg,
            &mut res.binaries,
            &mut res.notes,
            &mut res.errors,
        );
    }

    // Query GitHub contents API for versioned Linux `libnvidia-ngx-dlss.so.*`
    if !cfg.offline {
        let api_url = "https://api.github.com/repos/NVIDIA/DLSS/contents/lib/Linux_x86_64/rel";
        match curl_fetch_json(api_url) {
            Ok(Value::Array(entries)) => {
                let mut found_so = false;
                for entry in entries {
                    let Some(name) = entry.get("name").and_then(Value::as_str) else {
                        continue;
                    };
                    if name.starts_with("libnvidia-ngx-dlss.so") {
                        if let Some(dl_url) = entry.get("download_url").and_then(Value::as_str) {
                            found_so = true;
                            fetch_single(
                                SdkKind::Ngx,
                                "binary",
                                name,
                                dl_url,
                                &binaries_dir.join(name),
                                cfg,
                                &mut res.binaries,
                                &mut res.notes,
                                &mut res.errors,
                            );
                        }
                    }
                }
                if !found_so {
                    res.notes.push(
                        "NVIDIA/DLSS Linux_x86_64/rel listing contained no libnvidia-ngx-dlss.so.*"
                            .to_string(),
                    );
                }
            }
            Ok(_) => res.errors.push(format!(
                "Unexpected JSON shape from GitHub contents API: {api_url}"
            )),
            Err(err) => res.errors.push(format!(
                "Failed to query NVIDIA/DLSS Linux rel contents ({api_url}): {err}"
            )),
        }
    } else {
        // Offline: pick up any previously cached `libnvidia-ngx-dlss.so.*`
        scan_dir_into_artifacts(
            SdkKind::Ngx,
            "binary",
            &binaries_dir,
            |name| name.starts_with("libnvidia-ngx-dlss.so"),
            &mut res.binaries,
        );
    }
}

fn fetch_nrd(cfg: &DownloadConfig, res: &mut SdkDownloadResult) {
    let headers_dir = cfg.cache_dir.join("nrd").join("headers");
    let binaries_dir = cfg.cache_dir.join("nrd").join("binaries");
    let archives_dir = cfg.cache_dir.join("nrd").join("archives");
    let extracted_dir = cfg.cache_dir.join("nrd").join("extracted");

    let header_urls = [
        (
            "NRD.h",
            "https://raw.githubusercontent.com/NVIDIA-RTX/NRD/master/Include/NRD.h",
        ),
        (
            "NRDDescs.h",
            "https://raw.githubusercontent.com/NVIDIA-RTX/NRD/master/Include/NRDDescs.h",
        ),
        (
            "NRDSettings.h",
            "https://raw.githubusercontent.com/NVIDIA-RTX/NRD/master/Include/NRDSettings.h",
        ),
    ];

    for (name, url) in header_urls {
        fetch_single(
            SdkKind::Nrd,
            "header",
            name,
            url,
            &headers_dir.join(name),
            cfg,
            &mut res.headers,
            &mut res.notes,
            &mut res.errors,
        );
    }

    if !cfg.offline {
        let api_url = "https://api.github.com/repos/NVIDIA-RTX/NRD/releases/latest";
        match curl_fetch_json(api_url) {
            Ok(json) => {
                if let Some(assets) = json.get("assets").and_then(Value::as_array) {
                    let mut found_prebuilt = false;
                    for asset in assets {
                        let Some(name) = asset.get("name").and_then(Value::as_str) else {
                            continue;
                        };
                        let Some(dl_url) =
                            asset.get("browser_download_url").and_then(Value::as_str)
                        else {
                            continue;
                        };
                        let lower = name.to_ascii_lowercase();
                        if lower.ends_with(".dll")
                            || lower.ends_with(".so")
                            || lower.contains(".so.")
                        {
                            found_prebuilt = true;
                            fetch_single(
                                SdkKind::Nrd,
                                "binary",
                                name,
                                dl_url,
                                &binaries_dir.join(name),
                                cfg,
                                &mut res.binaries,
                                &mut res.notes,
                                &mut res.errors,
                            );
                        } else if lower.ends_with(".zip") || lower.ends_with(".tar.gz") {
                            found_prebuilt = true;
                            if cfg.include_large_zips {
                                let archive_path = archives_dir.join(name);
                                let mut archive_list = Vec::new();
                                fetch_single(
                                    SdkKind::Nrd,
                                    "archive",
                                    name,
                                    dl_url,
                                    &archive_path,
                                    cfg,
                                    &mut archive_list,
                                    &mut res.notes,
                                    &mut res.errors,
                                );
                                if archive_path.is_file() {
                                    if let Err(e) = extract_archive(&archive_path, &extracted_dir) {
                                        res.errors.push(format!(
                                            "Failed to extract NRD archive '{}': {e}",
                                            archive_path.display()
                                        ));
                                    }
                                }
                            } else {
                                res.notes.push(format!(
                                    "Skipped NRD release archive '{name}' (include_large_zips=false)"
                                ));
                            }
                        }
                    }
                    if !found_prebuilt {
                        res.notes.push(
                            "NVIDIA-RTX/NRD latest release has no prebuilt binary assets; relying on cache/local natives"
                                .to_string(),
                        );
                    }
                }
            }
            Err(err) => {
                res.notes.push(format!(
                    "NRD latest release query unavailable ({api_url}): {err}"
                ));
            }
        }
    }

    // Scan cached binaries & extracted tree for NRD binaries
    scan_dir_into_artifacts(
        SdkKind::Nrd,
        "binary",
        &binaries_dir,
        |name| is_sdk_binary_name(SdkKind::Nrd, name),
        &mut res.binaries,
    );
    scan_dir_into_artifacts(
        SdkKind::Nrd,
        "binary",
        &extracted_dir,
        |name| is_sdk_binary_name(SdkKind::Nrd, name),
        &mut res.binaries,
    );
}

fn fetch_fsr3(cfg: &DownloadConfig, res: &mut SdkDownloadResult) {
    let headers_dir = cfg.cache_dir.join("fsr3").join("headers");
    let binaries_dir = cfg.cache_dir.join("fsr3").join("binaries");

    let header_urls = [
        (
            "ffx_api_types.h",
            "https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/main/Kits/FidelityFX/api/include/ffx_api_types.h",
        ),
        (
            "ffx_api.h",
            "https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/main/Kits/FidelityFX/api/include/ffx_api.h",
        ),
        (
            "ffx_api_vk.h",
            "https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/main/Kits/FidelityFX/api/include/vk/ffx_api_vk.h",
        ),
        (
            "ffx_upscale.h",
            "https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/main/Kits/FidelityFX/upscalers/include/ffx_upscale.h",
        ),
        (
            "ffx_framegeneration.h",
            "https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/main/Kits/FidelityFX/framegeneration/include/ffx_framegeneration.h",
        ),
    ];

    for (name, url) in header_urls {
        fetch_single(
            SdkKind::Fsr3,
            "header",
            name,
            url,
            &headers_dir.join(name),
            cfg,
            &mut res.headers,
            &mut res.notes,
            &mut res.errors,
        );
    }

    // Prebuilt v1.1.4 Vulkan DLL
    fetch_single(
        SdkKind::Fsr3,
        "binary",
        "amd_fidelityfx_vk.dll",
        "https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/raw/v1.1.4/PrebuiltSignedDLL/amd_fidelityfx_vk.dll",
        &binaries_dir.join("amd_fidelityfx_vk.dll"),
        cfg,
        &mut res.binaries,
        &mut res.notes,
        &mut res.errors,
    );

    // Query GitHub contents API for FidelityFX signedbin Vulkan binaries
    if !cfg.offline {
        let api_url = "https://api.github.com/repos/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/contents/Kits/FidelityFX/signedbin";
        match curl_fetch_json(api_url) {
            Ok(Value::Array(entries)) => {
                for entry in entries {
                    let Some(name) = entry.get("name").and_then(Value::as_str) else {
                        continue;
                    };
                    let lower = name.to_ascii_lowercase();
                    let is_vk_bin = lower.contains("vk")
                        && (lower.ends_with(".dll")
                            || lower.ends_with(".so")
                            || lower.contains(".so."));
                    if is_vk_bin {
                        if let Some(dl_url) = entry.get("download_url").and_then(Value::as_str) {
                            fetch_single(
                                SdkKind::Fsr3,
                                "binary",
                                name,
                                dl_url,
                                &binaries_dir.join(name),
                                cfg,
                                &mut res.binaries,
                                &mut res.notes,
                                &mut res.errors,
                            );
                        }
                    }
                }
            }
            Ok(_) => res.errors.push(format!(
                "Unexpected JSON shape from FidelityFX signedbin API: {api_url}"
            )),
            Err(err) => res.errors.push(format!(
                "Failed to query FidelityFX signedbin contents ({api_url}): {err}"
            )),
        }
    } else {
        scan_dir_into_artifacts(
            SdkKind::Fsr3,
            "binary",
            &binaries_dir,
            |name| is_sdk_binary_name(SdkKind::Fsr3, name),
            &mut res.binaries,
        );
    }
}

fn fetch_xess(cfg: &DownloadConfig, res: &mut SdkDownloadResult) {
    let archives_dir = cfg.cache_dir.join("xess").join("archives");
    let extracted_dir = cfg.cache_dir.join("xess").join("extracted");

    if !cfg.offline {
        let api_url = "https://api.github.com/repos/intel/xess/releases/latest";
        match curl_fetch_json(api_url) {
            Ok(json) => {
                let zip_asset = json
                    .get("assets")
                    .and_then(Value::as_array)
                    .and_then(|assets| {
                        assets.iter().find(|a| {
                            a.get("name")
                                .and_then(Value::as_str)
                                .is_some_and(|n| n.to_ascii_lowercase().ends_with(".zip"))
                        })
                    });

                if let Some(asset) = zip_asset {
                    let name = asset
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("xess_sdk.zip");
                    let dl_url = asset
                        .get("browser_download_url")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    let archive_path = archives_dir.join(name);

                    if cfg.include_large_zips || archive_path.is_file() {
                        let mut archive_list = Vec::new();
                        fetch_single(
                            SdkKind::Xess,
                            "archive",
                            name,
                            dl_url,
                            &archive_path,
                            cfg,
                            &mut archive_list,
                            &mut res.notes,
                            &mut res.errors,
                        );

                        if archive_path.is_file() {
                            let need_extract = cfg.force_download
                                || !extracted_dir.is_dir()
                                || dir_is_empty(&extracted_dir);
                            if need_extract {
                                if let Err(e) = extract_archive(&archive_path, &extracted_dir) {
                                    res.errors.push(format!(
                                        "Failed to extract XeSS archive '{}': {e}",
                                        archive_path.display()
                                    ));
                                } else {
                                    res.notes.push(format!(
                                        "Extracted XeSS SDK archive '{name}' into {}",
                                        extracted_dir.display()
                                    ));
                                }
                            }
                        }
                    } else {
                        res.notes.push(format!(
                            "Skipped XeSS release archive '{name}' (include_large_zips=false)"
                        ));
                    }
                } else {
                    res.errors.push(format!(
                        "No .zip release asset found in intel/xess latest release ({api_url})"
                    ));
                }
            }
            Err(err) => res.errors.push(format!(
                "Failed to query intel/xess latest release ({api_url}): {err}"
            )),
        }
    } else if !extracted_dir.is_dir() || dir_is_empty(&extracted_dir) {
        // Offline: if an archive is already cached, extract it locally
        if let Some(cached_zip) = find_first_zip(&archives_dir) {
            if let Err(e) = extract_archive(&cached_zip, &extracted_dir) {
                res.errors.push(format!(
                    "Failed to extract cached XeSS archive '{}': {e}",
                    cached_zip.display()
                ));
            }
        }
    }

    // Discover all xess*.h headers and libxess*.dll / libxess*.so* binaries inside extracted tree
    scan_dir_into_artifacts(
        SdkKind::Xess,
        "header",
        &extracted_dir,
        |name| {
            let lower = name.to_ascii_lowercase();
            lower.starts_with("xess") && lower.ends_with(".h")
        },
        &mut res.headers,
    );
    scan_dir_into_artifacts(
        SdkKind::Xess,
        "binary",
        &extracted_dir,
        |name| is_sdk_binary_name(SdkKind::Xess, name),
        &mut res.binaries,
    );
}

// ============================================================================
// Core curl Download, Atomic Write & Hashing
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn fetch_single(
    sdk: SdkKind,
    kind: &str,
    name: &str,
    url: &str,
    target_path: &Path,
    cfg: &DownloadConfig,
    dest: &mut Vec<DownloadedArtifact>,
    notes: &mut Vec<String>,
    errors: &mut Vec<String>,
) {
    if !cfg.force_download && target_path.is_file() {
        match hash_file_blake3(target_path) {
            Ok((blake3_hex, size_bytes)) if size_bytes > 0 => {
                push_artifact_unique(
                    dest,
                    DownloadedArtifact {
                        sdk,
                        kind: kind.to_string(),
                        name: name.to_string(),
                        url: url.to_string(),
                        local_path: target_path.to_path_buf(),
                        blake3_hex,
                        size_bytes,
                        from_cache: true,
                    },
                );
                return;
            }
            _ => {}
        }
    }

    if cfg.offline {
        notes.push(format!(
            "Offline mode: cached {kind} '{name}' not found at {}",
            target_path.display()
        ));
        return;
    }

    if let Some(parent) = target_path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            errors.push(format!(
                "Failed to create cache dir '{}': {e}",
                parent.display()
            ));
            return;
        }
    }

    let part_path = part_file_path(target_path);
    match curl_download_to_file(url, &part_path) {
        Ok(()) => {
            let (blake3_hex, size_bytes) = match hash_file_blake3(&part_path) {
                Ok(h) => h,
                Err(e) => {
                    let _ = std::fs::remove_file(&part_path);
                    errors.push(format!("Failed to hash downloaded '{name}': {e}"));
                    return;
                }
            };
            if size_bytes == 0 {
                let _ = std::fs::remove_file(&part_path);
                errors.push(format!("Downloaded empty file for '{name}' from {url}"));
                return;
            }
            if let Err(e) = std::fs::rename(&part_path, target_path) {
                let _ = std::fs::remove_file(&part_path);
                errors.push(format!(
                    "Failed to move '{name}' into cache '{}': {e}",
                    target_path.display()
                ));
                return;
            }
            push_artifact_unique(
                dest,
                DownloadedArtifact {
                    sdk,
                    kind: kind.to_string(),
                    name: name.to_string(),
                    url: url.to_string(),
                    local_path: target_path.to_path_buf(),
                    blake3_hex,
                    size_bytes,
                    from_cache: false,
                },
            );
        }
        Err(err) => {
            let _ = std::fs::remove_file(&part_path);
            errors.push(format!("Failed to download '{name}' from {url}: {err}"));
        }
    }
}

fn part_file_path(target: &Path) -> PathBuf {
    let mut os = target.as_os_str().to_os_string();
    os.push(".part");
    PathBuf::from(os)
}

fn build_curl_cmd(url: &str, accept_github_json: bool) -> Command {
    let mut cmd = Command::new("curl");
    cmd.args([
        "-fsSL",
        "--retry",
        "3",
        "--connect-timeout",
        "15",
        "--max-time",
        "180",
        "-A",
        "vulkanshaders-sdk-bench/0.2",
    ]);
    if accept_github_json {
        cmd.args(["-H", "Accept: application/vnd.github+json"]);
    }
    if let Ok(token) = std::env::var("GITHUB_TOKEN") {
        let trimmed = token.trim();
        if !trimmed.is_empty() {
            cmd.args(["-H", &format!("Authorization: Bearer {trimmed}")]);
        }
    }
    cmd.arg(url);
    cmd
}

fn curl_download_to_file(url: &str, dest: &Path) -> Result<(), String> {
    let mut cmd = build_curl_cmd(url, false);
    cmd.arg("-o").arg(dest);
    let output = cmd
        .output()
        .map_err(|e| format!("Failed to spawn curl: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(format!("curl exited with {}: {stderr}", output.status));
    }
    Ok(())
}

fn curl_fetch_json(url: &str) -> Result<Value, String> {
    let mut cmd = build_curl_cmd(url, true);
    let output = cmd
        .output()
        .map_err(|e| format!("Failed to spawn curl: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(format!("curl exited with {}: {stderr}", output.status));
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("Failed to parse JSON from {url}: {e}"))
}

fn hash_file_blake3(path: &Path) -> Result<(String, u64), String> {
    let mut file =
        File::open(path).map_err(|e| format!("Cannot open '{}': {e}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = [0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("Read error on '{}': {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    Ok((hasher.finalize().to_hex().to_string(), total))
}

// ============================================================================
// Archive Extraction & Local Directory Scanning
// ============================================================================

fn extract_archive(archive_path: &Path, dest_dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dest_dir)
        .map_err(|e| format!("Failed to create extract dir '{}': {e}", dest_dir.display()))?;

    let lower = archive_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    if lower.ends_with(".tar.gz") || lower.ends_with(".tgz") {
        let output = Command::new("tar")
            .arg("-xzf")
            .arg(archive_path)
            .arg("-C")
            .arg(dest_dir)
            .output()
            .map_err(|e| format!("Failed to spawn tar: {e}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            return Err(format!("tar exited with {}: {stderr}", output.status));
        }
        return Ok(());
    }

    if cfg!(target_os = "windows") {
        let output = Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!(
                    "Expand-Archive -Force -Path '{}' -DestinationPath '{}'",
                    archive_path.display(),
                    dest_dir.display()
                ),
            ])
            .output()
            .map_err(|e| format!("Failed to spawn powershell Expand-Archive: {e}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            return Err(format!(
                "Expand-Archive exited with {}: {stderr}",
                output.status
            ));
        }
    } else {
        let output = Command::new("unzip")
            .arg("-o")
            .arg(archive_path)
            .arg("-d")
            .arg(dest_dir)
            .output()
            .map_err(|e| format!("Failed to spawn unzip: {e}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            return Err(format!("unzip exited with {}: {stderr}", output.status));
        }
    }

    Ok(())
}

fn dir_is_empty(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|mut rd| rd.next().is_none())
        .unwrap_or(true)
}

fn find_first_zip(dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("zip"))
        {
            return Some(path);
        }
    }
    None
}

fn scan_local_natives(sdk: SdkKind, cfg: &DownloadConfig, res: &mut SdkDownloadResult) {
    for dir in &cfg.local_natives_dirs {
        if !dir.is_dir() {
            continue;
        }
        scan_dir_into_artifacts(
            sdk,
            "binary",
            dir,
            |name| is_sdk_binary_name(sdk, name),
            &mut res.binaries,
        );
    }
}

fn is_sdk_binary_name(sdk: SdkKind, file_name: &str) -> bool {
    let lower = file_name.to_ascii_lowercase();
    let is_shared_or_static = lower.ends_with(".dll")
        || lower.ends_with(".so")
        || lower.contains(".so.")
        || lower.ends_with(".a");
    if !is_shared_or_static {
        return false;
    }

    match sdk {
        SdkKind::Ngx => {
            lower.starts_with("libnvidia-ngx")
                || lower.starts_with("nvngx_")
                || lower == "libnvsdk_ngx.a"
        }
        SdkKind::Nrd => {
            lower == "nrd.dll" || lower == "libnrd.so" || lower.starts_with("libnrd.so.")
        }
        SdkKind::Fsr3 => {
            (lower.contains("fidelityfx") || lower.contains("ffx")) && lower.contains("vk")
        }
        SdkKind::Xess => lower.starts_with("libxess") || lower.starts_with("xess"),
    }
}

fn scan_dir_into_artifacts<F>(
    sdk: SdkKind,
    kind: &str,
    root: &Path,
    predicate: F,
    dest: &mut Vec<DownloadedArtifact>,
) where
    F: Fn(&str) -> bool,
{
    if !root.is_dir() {
        return;
    }
    let mut stack = vec![root.to_path_buf()];
    let mut visited = BTreeSet::new();

    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        paths.sort();

        for path in paths {
            if path.is_dir() {
                if visited.insert(path.clone()) {
                    stack.push(path);
                }
            } else if path.is_file() {
                let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                if !predicate(name) {
                    continue;
                }
                if let Ok((blake3_hex, size_bytes)) = hash_file_blake3(&path) {
                    if size_bytes == 0 {
                        continue;
                    }
                    let url = format!("file://{}", path.display());
                    push_artifact_unique(
                        dest,
                        DownloadedArtifact {
                            sdk,
                            kind: kind.to_string(),
                            name: name.to_string(),
                            url,
                            local_path: path,
                            blake3_hex,
                            size_bytes,
                            from_cache: true,
                        },
                    );
                }
            }
        }
    }
}

fn push_artifact_unique(dest: &mut Vec<DownloadedArtifact>, artifact: DownloadedArtifact) {
    let already_present = dest.iter().any(|existing| {
        existing.local_path == artifact.local_path
            || (existing.name == artifact.name && existing.blake3_hex == artifact.blake3_hex)
    });
    if !already_present {
        dest.push(artifact);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdk_kind_helpers() {
        assert_eq!(SdkKind::all().len(), 4);
        assert_eq!(SdkKind::Ngx.as_str(), "ngx");
        assert_eq!(SdkKind::Nrd.as_str(), "nrd");
        assert_eq!(SdkKind::Fsr3.as_str(), "fsr3");
        assert_eq!(SdkKind::Xess.as_str(), "xess");
    }

    #[test]
    fn test_binary_name_matchers() {
        assert!(is_sdk_binary_name(
            SdkKind::Ngx,
            "libnvidia-ngx-dlss.so.3.7.0"
        ));
        assert!(is_sdk_binary_name(SdkKind::Ngx, "nvngx_dlss.dll"));
        assert!(is_sdk_binary_name(SdkKind::Nrd, "libNRD.so.4"));
        assert!(is_sdk_binary_name(SdkKind::Nrd, "NRD.dll"));
        assert!(is_sdk_binary_name(
            SdkKind::Fsr3,
            "amd_fidelityfx_upscaler_vk.dll"
        ));
        assert!(is_sdk_binary_name(SdkKind::Xess, "libxess.dll"));
        assert!(!is_sdk_binary_name(
            SdkKind::Fsr3,
            "amd_fidelityfx_dx12.dll"
        ));
    }

    #[test]
    fn test_offline_cache_and_local_natives_scan() {
        let tmp = std::env::temp_dir().join(format!(
            "vulkanshaders_sdk_bench_dl_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        let ngx_hdr = tmp.join("cache").join("ngx").join("headers");
        let natives = tmp.join("natives");
        std::fs::create_dir_all(&ngx_hdr).unwrap();
        std::fs::create_dir_all(&natives).unwrap();

        std::fs::write(ngx_hdr.join("nvsdk_ngx_vk.h"), "#define TEST_NGX 1\n").unwrap();
        std::fs::write(natives.join("libnvidia-ngx-dlss.so.310.1"), b"\x7fELFdummy").unwrap();

        let cfg = DownloadConfig {
            cache_dir: tmp.join("cache"),
            force_download: false,
            offline: true,
            include_large_zips: false,
            local_natives_dirs: vec![natives],
        };

        let res = fetch_sdk_artifacts(SdkKind::Ngx, &cfg);
        assert_eq!(res.headers.len(), 1);
        assert_eq!(res.headers[0].name, "nvsdk_ngx_vk.h");
        assert!(res.headers[0].from_cache);
        assert_eq!(res.binaries.len(), 1);
        assert_eq!(res.binaries[0].name, "libnvidia-ngx-dlss.so.310.1");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
