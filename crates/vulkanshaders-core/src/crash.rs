//! Native crash interception, panic logging, and error collection for `vulkanshaders-core`.
//!
//! Designed to coexist safely with the HotSpot JVM:
//! - Panic hook writes `<crash_dir>/native-panic-<unix_ts>-pid<pid>.log` with payload, location,
//!   thread name, and forced backtrace, queues a summary into the native error ring buffer, and
//!   chains to the previous panic hook (guarded against recursion via `AtomicBool`).
//! - Unix (`linux` / `macos`) signal handler computes this shared library's address range
//!   `[SELF_ADDR_START, SELF_ADDR_END)` once during `install()` (outside any signal handler) and
//!   installs `SA_SIGINFO` handlers for `SIGSEGV`, `SIGBUS`, `SIGABRT`, `SIGFPE`, and `SIGILL`,
//!   preserving `old_act`. For `SIGSEGV`/`SIGBUS`/`SIGILL`, it extracts the instruction pointer (`RIP`/`PC`)
//!   from `ucontext` and checks `pc >= start && pc < end` lock-free (zero `dladdr` calls inside
//!   the signal handler) so normal HotSpot JVM implicit null-checks and safepoint polls pass
//!   straight through. Crash reports are formatted into an async-signal-safe stack buffer
//!   (`[u8; 2048]`) and written via `open`/`write`/`close` before chaining to `old_act` (or
//!   `SIG_DFL` + `raise`).
//! - Windows installs a top-level `SetUnhandledExceptionFilter` (never VEH) that writes exception
//!   code, address, and PID with zero heap allocations via `CreateFileW`/`WriteFile`/`CloseHandle`
//!   using a pre-encoded UTF-16 crash directory path (`CRASH_DIR_PATH_W`), then chains to the
//!   previous filter or returns `EXCEPTION_CONTINUE_SEARCH`.
//! - A minimal global `tracing::Subscriber` captures `ERROR` and `WARN` events into a bounded
//!   64-entry ring buffer (`LEVEL|target|message`) for Java/Kotlin FFI retrieval.

use std::backtrace::Backtrace;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::path::Path;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex, Once};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Record};
use tracing::{Event, Id, Level, Metadata, Subscriber};

/// Capacity of the lock-free crash directory prefix buffers (UTF-8 bytes / UTF-16 code units).
const CRASH_DIR_CAPACITY: usize = 4096;
/// Minimum headroom reserved inside `CRASH_DIR_PATH` / `CRASH_DIR_PATH_W` for appending a crash
/// log filename + NUL terminator.
const CRASH_FILENAME_HEADROOM: usize = 64;
/// Maximum number of native error/warning entries kept in the FFI ring buffer.
const MAX_ERROR_QUEUE: usize = 64;

/// Lock-free UTF-8 crash directory prefix buffer (`<game_dir>/hyaloclast/crash-logs/\0`).
///
/// Writers serialize through `CRASH_DIR_WRITE_LOCK`, clear `READY`, write bytes via raw pointer,
/// publish `CRASH_DIR_LEN` (and `CRASH_DIR_LEN_W` on Windows), and set `READY` with `Release`
/// ordering. Signal/exception handlers only ever read after observing
/// `READY.load(Ordering::Acquire) == true`.
static mut CRASH_DIR_PATH: [u8; CRASH_DIR_CAPACITY] = [0u8; CRASH_DIR_CAPACITY];
static CRASH_DIR_LEN: AtomicUsize = AtomicUsize::new(0);
#[cfg(windows)]
static mut CRASH_DIR_PATH_W: [u16; CRASH_DIR_CAPACITY] = [0u16; CRASH_DIR_CAPACITY];
#[cfg(windows)]
static CRASH_DIR_LEN_W: AtomicUsize = AtomicUsize::new(0);
static READY: AtomicBool = AtomicBool::new(false);
static CRASH_DIR_WRITE_LOCK: Mutex<()> = Mutex::new(());

static INIT_ONCE: Once = Once::new();
static IN_PANIC_HOOK: AtomicBool = AtomicBool::new(false);
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
static IN_CRASH_HANDLER: AtomicBool = AtomicBool::new(false);

static ERROR_QUEUE: LazyLock<Mutex<VecDeque<String>>> =
    LazyLock::new(|| Mutex::new(VecDeque::with_capacity(MAX_ERROR_QUEUE)));
static PENDING_ERROR_COUNT: AtomicU32 = AtomicU32::new(0);

/// Configures the game directory, creates `<dir>/hyaloclast/crash-logs`, and publishes its prefix
/// (ending with `/` and NUL-terminated) into the lock-free `CRASH_DIR_PATH` buffer (and UTF-16
/// `CRASH_DIR_PATH_W` on Windows).
pub fn set_game_dir(dir: &Path) {
    let crash_dir = dir.join("hyaloclast").join("crash-logs");
    let _ = std::fs::create_dir_all(&crash_dir);

    let mut prefix = crash_dir.to_string_lossy().into_owned();
    while prefix.ends_with('/') || prefix.ends_with('\\') {
        prefix.pop();
    }
    if prefix.is_empty() {
        return;
    }
    prefix.push('/');

    let bytes = prefix.as_bytes();
    if bytes.len() >= CRASH_DIR_CAPACITY - CRASH_FILENAME_HEADROOM {
        return;
    }

    #[cfg(windows)]
    let wide_units: Vec<u16> = prefix.encode_utf16().collect();
    #[cfg(windows)]
    if wide_units.len() >= CRASH_DIR_CAPACITY - CRASH_FILENAME_HEADROOM {
        return;
    }

    let _guard = CRASH_DIR_WRITE_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    READY.store(false, Ordering::Release);
    unsafe {
        let dst = (&raw mut CRASH_DIR_PATH) as *mut u8;
        ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        *dst.add(bytes.len()) = 0;
    }
    CRASH_DIR_LEN.store(bytes.len(), Ordering::Release);
    #[cfg(windows)]
    {
        unsafe {
            let dst_w = (&raw mut CRASH_DIR_PATH_W) as *mut u16;
            ptr::copy_nonoverlapping(wide_units.as_ptr(), dst_w, wide_units.len());
            *dst_w.add(wide_units.len()) = 0;
        }
        CRASH_DIR_LEN_W.store(wide_units.len(), Ordering::Release);
    }
    READY.store(true, Ordering::Release);
}

/// Reads the published crash directory prefix for non-signal contexts (such as the panic hook).
fn read_crash_dir_prefix() -> Option<String> {
    if !READY.load(Ordering::Acquire) {
        return None;
    }
    let len = CRASH_DIR_LEN.load(Ordering::Acquire);
    if len == 0 || len >= CRASH_DIR_CAPACITY {
        return None;
    }
    let mut local = vec![0u8; len];
    unsafe {
        let src = (&raw const CRASH_DIR_PATH) as *const u8;
        ptr::copy_nonoverlapping(src, local.as_mut_ptr(), len);
    }
    if !READY.load(Ordering::Acquire) {
        return None;
    }
    String::from_utf8(local).ok()
}

/// Performs one-time early initialization of the tracing subscriber, panic hook, and native
/// crash handler. Safe to call multiple times from any thread.
pub fn init_early() {
    INIT_ONCE.call_once(|| {
        install_tracing_subscriber();
        install_panic_hook();
        install_crash_handler();
    });
}

/// Sanitizes a field for single-line `LEVEL|target|message` serialization.
fn sanitize_segment(input: &str, strip_pipes: bool) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\r' | '\n' => out.push(' '),
            '|' if strip_pipes => out.push('/'),
            other => out.push(other),
        }
    }
    out
}

/// Pushes an error or warning entry (`LEVEL|target|message`) into the bounded 64-entry ring buffer
/// and updates `PENDING_ERROR_COUNT`.
pub fn push_native_error(level: &str, target: &str, message: &str) {
    let lvl = sanitize_segment(level, true);
    let tgt = sanitize_segment(target, true);
    let msg = sanitize_segment(message, false);
    let entry = format!("{lvl}|{tgt}|{msg}");

    let mut queue = ERROR_QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if queue.len() >= MAX_ERROR_QUEUE {
        queue.pop_front();
    } else {
        PENDING_ERROR_COUNT.fetch_add(1, Ordering::AcqRel);
    }
    queue.push_back(entry);
}

/// Returns the current number of queued native error/warning entries.
pub fn pending_error_count() -> u32 {
    PENDING_ERROR_COUNT.load(Ordering::Acquire)
}

/// Returns the total UTF-8 byte length of the currently queued entries joined by `'\n'`,
/// without clearing the queue (used by the FFI two-step buffer sizing protocol).
pub fn peek_native_errors_len() -> usize {
    let queue = ERROR_QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if queue.is_empty() {
        return 0;
    }
    let content_bytes: usize = queue.iter().map(String::len).sum();
    content_bytes + queue.len().saturating_sub(1)
}

/// Drains all queued native error/warning entries into a single newline-delimited string and
/// resets `PENDING_ERROR_COUNT` to zero.
pub fn drain_native_errors() -> String {
    let mut queue = ERROR_QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if queue.is_empty() {
        PENDING_ERROR_COUNT.store(0, Ordering::Release);
        return String::new();
    }
    let total_len: usize =
        queue.iter().map(String::len).sum::<usize>() + queue.len().saturating_sub(1);
    let mut out = String::with_capacity(total_len);
    for (idx, item) in queue.drain(..).enumerate() {
        if idx > 0 {
            out.push('\n');
        }
        out.push_str(&item);
    }
    PENDING_ERROR_COUNT.store(0, Ordering::Release);
    out
}

// ---------------------------------------------------------------------------
// Minimal global `tracing::Subscriber` for ERROR and WARN events
// ---------------------------------------------------------------------------

struct NativeErrorSubscriber;

struct EventMessageVisitor {
    message: Option<String>,
    fields: Vec<String>,
}

impl EventMessageVisitor {
    fn new() -> Self {
        Self {
            message: None,
            fields: Vec::new(),
        }
    }

    fn into_message(self) -> String {
        match (self.message, self.fields.is_empty()) {
            (Some(msg), true) => msg,
            (Some(msg), false) => format!("{msg} ({})", self.fields.join(", ")),
            (None, false) => self.fields.join(", "),
            (None, true) => "(empty tracing event)".to_owned(),
        }
    }
}

impl Visit for EventMessageVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = Some(value.to_owned());
        } else {
            self.fields.push(format!("{}={value}", field.name()));
        }
    }

    fn record_error(&mut self, field: &Field, value: &(dyn std::error::Error + 'static)) {
        if field.name() == "message" {
            self.message = Some(value.to_string());
        } else {
            self.fields.push(format!("{}={value}", field.name()));
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
        if field.name() == "message" {
            self.message = Some(format!("{value:?}"));
        } else {
            self.fields.push(format!("{}={value:?}", field.name()));
        }
    }
}

impl Subscriber for NativeErrorSubscriber {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        matches!(*metadata.level(), Level::ERROR | Level::WARN)
    }

    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _span: &Id, _values: &Record<'_>) {}

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    fn event(&self, event: &Event<'_>) {
        let meta = event.metadata();
        let level_str = match *meta.level() {
            Level::ERROR => "ERROR",
            Level::WARN => "WARN",
            _ => return,
        };
        let mut visitor = EventMessageVisitor::new();
        event.record(&mut visitor);
        let msg = visitor.into_message();
        push_native_error(level_str, meta.target(), &msg);
    }

    fn enter(&self, _span: &Id) {}

    fn exit(&self, _span: &Id) {}
}

fn install_tracing_subscriber() {
    let _ = tracing::subscriber::set_global_default(NativeErrorSubscriber);
}

// ---------------------------------------------------------------------------
// Panic hook with recursion guard and forced backtrace
// ---------------------------------------------------------------------------

struct PanicHookGuard;

impl Drop for PanicHookGuard {
    fn drop(&mut self) {
        IN_PANIC_HOOK.store(false, Ordering::Release);
    }
}

fn install_panic_hook() {
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if IN_PANIC_HOOK.swap(true, Ordering::AcqRel) {
            prev_hook(info);
            return;
        }
        let _guard = PanicHookGuard;

        let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_owned()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic payload>".to_owned()
        };

        let location = info
            .location()
            .map(|loc| format!("{}:{}:{}", loc.file(), loc.line(), loc.column()))
            .unwrap_or_else(|| "<unknown location>".to_owned());

        let thread_name = std::thread::current()
            .name()
            .unwrap_or("<unnamed>")
            .to_owned();

        let backtrace = Backtrace::force_capture();
        let unix_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let pid = std::process::id();

        if let Some(prefix) = read_crash_dir_prefix() {
            let log_path = format!("{prefix}native-panic-{unix_ts}-pid{pid}.log");
            let report = format!(
                "Hyaloclast Native Panic Report\n\
                 Timestamp: {unix_ts}\n\
                 PID: {pid}\n\
                 Thread: {thread_name}\n\
                 Location: {location}\n\
                 Payload: {payload}\n\n\
                 Backtrace:\n{backtrace}\n"
            );
            let _ = std::fs::write(&log_path, report.as_bytes());
        }

        let summary = format!("panic at {location} (thread={thread_name}): {payload}");
        push_native_error("ERROR", "panic", &summary);

        prev_hook(info);
    }));
}

// ---------------------------------------------------------------------------
// Allocation-free stack formatting helpers (async-signal-safe)
// ---------------------------------------------------------------------------

#[cfg(any(target_os = "linux", target_os = "macos", windows))]
struct StackBuf<const N: usize> {
    buf: [u8; N],
    len: usize,
}

#[cfg(any(target_os = "linux", target_os = "macos", windows))]
impl<const N: usize> StackBuf<N> {
    const fn new() -> Self {
        Self {
            buf: [0u8; N],
            len: 0,
        }
    }

    fn push_bytes(&mut self, bytes: &[u8]) {
        let avail = N.saturating_sub(self.len);
        let to_copy = bytes.len().min(avail);
        if to_copy == 0 {
            return;
        }
        let mut i = 0;
        while i < to_copy {
            self.buf[self.len + i] = bytes[i];
            i += 1;
        }
        self.len += to_copy;
    }

    fn push_str(&mut self, s: &str) {
        self.push_bytes(s.as_bytes());
    }

    fn push_u32_dec(&mut self, mut v: u32) {
        let mut tmp = [0u8; 10];
        let mut idx = tmp.len();
        if v == 0 {
            self.push_bytes(b"0");
            return;
        }
        while v > 0 && idx > 0 {
            idx -= 1;
            tmp[idx] = b'0' + (v % 10) as u8;
            v /= 10;
        }
        self.push_bytes(&tmp[idx..]);
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn push_i32_dec(&mut self, v: i32) {
        if v < 0 {
            self.push_bytes(b"-");
            self.push_u32_dec(v.unsigned_abs());
        } else {
            self.push_u32_dec(v as u32);
        }
    }

    #[cfg(windows)]
    fn push_u32_hex(&mut self, v: u32) {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = [0u8; 10];
        out[0] = b'0';
        out[1] = b'x';
        let mut i = 0;
        while i < 8 {
            let shift = (7 - i) * 4;
            out[2 + i] = HEX[((v >> shift) & 0xF) as usize];
            i += 1;
        }
        self.push_bytes(&out);
    }

    fn push_usize_hex(&mut self, v: usize) {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        const DIGITS: usize = std::mem::size_of::<usize>() * 2;
        let mut out = [0u8; 18];
        out[0] = b'0';
        out[1] = b'x';
        let mut i = 0;
        while i < DIGITS {
            let shift = (DIGITS - 1 - i) * 4;
            out[2 + i] = HEX[(v >> shift) & 0xF];
            i += 1;
        }
        self.push_bytes(&out[..(2 + DIGITS)]);
    }

    fn as_slice(&self) -> &[u8] {
        &self.buf[..self.len]
    }
}

/// Builds a NUL-terminated UTF-8 `<crash_dir>/native-crash-pid<pid>.log\0` path into a stack
/// buffer without any heap allocation.
#[cfg(any(target_os = "linux", target_os = "macos"))]
unsafe fn build_crash_file_path(out: &mut [u8; CRASH_DIR_CAPACITY], pid: u32) -> bool {
    if !READY.load(Ordering::Acquire) {
        return false;
    }
    let prefix_len = CRASH_DIR_LEN.load(Ordering::Acquire);
    if prefix_len == 0 || prefix_len >= CRASH_DIR_CAPACITY - CRASH_FILENAME_HEADROOM {
        return false;
    }

    let mut sb = StackBuf::<CRASH_DIR_CAPACITY>::new();
    let src = (&raw const CRASH_DIR_PATH) as *const u8;
    let mut i = 0;
    while i < prefix_len {
        let byte = unsafe { *src.add(i) };
        sb.push_bytes(&[byte]);
        i += 1;
    }
    sb.push_str("native-crash-pid");
    sb.push_u32_dec(pid);
    sb.push_str(".log");

    if sb.len >= CRASH_DIR_CAPACITY {
        return false;
    }
    let mut j = 0;
    while j < sb.len {
        out[j] = sb.buf[j];
        j += 1;
    }
    out[sb.len] = 0;
    true
}

/// Builds a NUL-terminated UTF-16 `<crash_dir>/native-crash-pid<pid>.log\0` path into a stack
/// buffer without any heap allocation (supports Cyrillic and all non-ASCII Windows user paths).
#[cfg(windows)]
unsafe fn build_crash_file_path_w(out: &mut [u16; CRASH_DIR_CAPACITY], pid: u32) -> bool {
    if !READY.load(Ordering::Acquire) {
        return false;
    }
    let prefix_len = CRASH_DIR_LEN_W.load(Ordering::Acquire);
    if prefix_len == 0 || prefix_len >= CRASH_DIR_CAPACITY - CRASH_FILENAME_HEADROOM {
        return false;
    }

    let src = (&raw const CRASH_DIR_PATH_W) as *const u16;
    let mut i = 0;
    while i < prefix_len {
        out[i] = unsafe { *src.add(i) };
        i += 1;
    }

    let mut suffix = StackBuf::<CRASH_FILENAME_HEADROOM>::new();
    suffix.push_str("native-crash-pid");
    suffix.push_u32_dec(pid);
    suffix.push_str(".log");

    let total = prefix_len + suffix.len;
    if total >= CRASH_DIR_CAPACITY {
        return false;
    }
    let mut j = 0;
    while j < suffix.len {
        out[prefix_len + j] = suffix.buf[j] as u16;
        j += 1;
    }
    out[total] = 0;
    true
}

// ---------------------------------------------------------------------------
// Unix (Linux & macOS) async-signal-safe crash handler (HotSpot-JVM-safe)
// ---------------------------------------------------------------------------

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod platform_crash {
    use super::{
        CRASH_DIR_CAPACITY, IN_CRASH_HANDLER, StackBuf, build_crash_file_path, init_early,
    };
    use std::ffi::c_void;
    use std::ptr;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const SIGILL: i32 = 4;
    const SIGABRT: i32 = 6;
    const SIGFPE: i32 = 8;
    const SIGSEGV: i32 = 11;
    #[cfg(target_os = "linux")]
    const SIGBUS: i32 = 7;
    #[cfg(target_os = "macos")]
    const SIGBUS: i32 = 10;

    #[cfg(target_os = "linux")]
    const O_WRONLY: i32 = 1;
    #[cfg(target_os = "linux")]
    const O_CREAT: i32 = 0o100;
    #[cfg(target_os = "linux")]
    const O_TRUNC: i32 = 0o1000;
    #[cfg(target_os = "linux")]
    const O_CLOEXEC: i32 = 0o2000000;
    #[cfg(target_os = "linux")]
    const SA_SIGINFO: i32 = 4;

    #[cfg(target_os = "macos")]
    const O_WRONLY: i32 = 0x0001;
    #[cfg(target_os = "macos")]
    const O_CREAT: i32 = 0x0200;
    #[cfg(target_os = "macos")]
    const O_TRUNC: i32 = 0x0400;
    #[cfg(target_os = "macos")]
    const O_CLOEXEC: i32 = 0x0100_0000;
    #[cfg(target_os = "macos")]
    const SA_SIGINFO: i32 = 0x0040;

    const SIG_DFL: usize = 0;
    const SIG_IGN: usize = 1;
    const FALLBACK_LIB_SPAN: usize = 64 * 1024 * 1024;

    const TRACKED_SIGNALS: [i32; 5] = [SIGSEGV, SIGBUS, SIGABRT, SIGFPE, SIGILL];

    #[cfg(target_os = "linux")]
    #[repr(C)]
    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct SigAction {
        sa_sigaction: usize,
        sa_mask: [u64; 16],
        sa_flags: i32,
        sa_restorer: usize,
    }

    #[cfg(target_os = "linux")]
    impl SigAction {
        const fn zeroed() -> Self {
            Self {
                sa_sigaction: 0,
                sa_mask: [0u64; 16],
                sa_flags: 0,
                sa_restorer: 0,
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[repr(C)]
    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct SigAction {
        sa_sigaction: usize,
        sa_mask: u32,
        sa_flags: i32,
    }

    #[cfg(target_os = "macos")]
    impl SigAction {
        const fn zeroed() -> Self {
            Self {
                sa_sigaction: 0,
                sa_mask: 0,
                sa_flags: 0,
            }
        }
    }

    #[cfg(target_os = "linux")]
    #[repr(C)]
    #[allow(dead_code)]
    struct SigInfo {
        si_signo: i32,
        si_errno: i32,
        si_code: i32,
        #[cfg(target_pointer_width = "64")]
        _pad0: i32,
        si_addr: *const c_void,
    }

    #[cfg(target_os = "macos")]
    #[repr(C)]
    #[allow(dead_code)]
    struct SigInfo {
        si_signo: i32,
        si_errno: i32,
        si_code: i32,
        si_pid: i32,
        si_uid: u32,
        si_status: i32,
        si_addr: *const c_void,
    }

    #[repr(C)]
    #[allow(dead_code)]
    struct DlInfo {
        dli_fname: *const u8,
        dli_fbase: *const c_void,
        dli_sname: *const u8,
        dli_saddr: *const c_void,
    }

    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    #[repr(C)]
    #[allow(dead_code)]
    struct ElfPhdr {
        p_type: u32,
        p_flags: u32,
        p_offset: u64,
        p_vaddr: u64,
        p_paddr: u64,
        p_filesz: u64,
        p_memsz: u64,
        p_align: u64,
    }

    #[cfg(all(target_os = "linux", target_pointer_width = "32"))]
    #[repr(C)]
    #[allow(dead_code)]
    struct ElfPhdr {
        p_type: u32,
        p_offset: u32,
        p_vaddr: u32,
        p_paddr: u32,
        p_filesz: u32,
        p_memsz: u32,
        p_flags: u32,
        p_align: u32,
    }

    #[cfg(target_os = "linux")]
    #[repr(C)]
    #[allow(dead_code)]
    struct DlPhdrInfo {
        dlpi_addr: usize,
        dlpi_name: *const u8,
        dlpi_phdr: *const ElfPhdr,
        dlpi_phnum: u16,
    }

    #[cfg(target_os = "linux")]
    const PT_LOAD: u32 = 1;
    #[cfg(target_os = "linux")]
    const PF_X: u32 = 1;

    #[cfg(target_os = "macos")]
    const MH_MAGIC_64: u32 = 0xfeed_facf;
    #[cfg(target_os = "macos")]
    const LC_SEGMENT_64: u32 = 0x19;

    #[cfg(target_os = "macos")]
    #[repr(C)]
    #[allow(dead_code)]
    struct MachHeader64 {
        magic: u32,
        cputype: i32,
        cpusubtype: i32,
        filetype: u32,
        ncmds: u32,
        sizeofcmds: u32,
        flags: u32,
        reserved: u32,
    }

    #[cfg(target_os = "macos")]
    #[repr(C)]
    struct LoadCommand {
        cmd: u32,
        cmdsize: u32,
    }

    #[cfg(target_os = "macos")]
    #[repr(C)]
    #[allow(dead_code)]
    struct SegmentCommand64 {
        cmd: u32,
        cmdsize: u32,
        segname: [u8; 16],
        vmaddr: u64,
        vmsize: u64,
        fileoff: u64,
        filesize: u64,
        maxprot: i32,
        initprot: i32,
        nsects: u32,
        flags: u32,
    }

    unsafe extern "C" {
        fn sigaction(signum: i32, act: *const SigAction, oldact: *mut SigAction) -> i32;
        fn open(path: *const std::ffi::c_char, oflag: i32, ...) -> i32;
        fn write(fd: i32, buf: *const c_void, count: usize) -> isize;
        fn close(fd: i32) -> i32;
        fn getpid() -> i32;
        fn raise(sig: i32) -> i32;
        fn dladdr(addr: *const c_void, info: *mut DlInfo) -> i32;
        #[cfg(target_os = "linux")]
        fn dl_iterate_phdr(
            callback: unsafe extern "C" fn(*mut DlPhdrInfo, usize, *mut c_void) -> i32,
            data: *mut c_void,
        ) -> i32;
    }

    static SELF_LIB_BASE: AtomicUsize = AtomicUsize::new(0);
    static SELF_ADDR_START: AtomicUsize = AtomicUsize::new(0);
    static SELF_ADDR_END: AtomicUsize = AtomicUsize::new(0);
    static mut OLD_ACTIONS: [SigAction; 5] = [SigAction::zeroed(); 5];

    fn signal_slot(sig: i32) -> Option<usize> {
        match sig {
            SIGSEGV => Some(0),
            SIGBUS => Some(1),
            SIGABRT => Some(2),
            SIGFPE => Some(3),
            SIGILL => Some(4),
            _ => None,
        }
    }

    fn signal_name(sig: i32) -> &'static str {
        match sig {
            SIGSEGV => "SIGSEGV",
            SIGBUS => "SIGBUS",
            SIGABRT => "SIGABRT",
            SIGFPE => "SIGFPE",
            SIGILL => "SIGILL",
            _ => "UNKNOWN",
        }
    }

    #[cfg(target_os = "linux")]
    struct PhdrSearchState {
        fbase: usize,
        fn_addr: usize,
        start: usize,
        end: usize,
    }

    #[cfg(target_os = "linux")]
    unsafe extern "C" fn dl_phdr_callback(
        info: *mut DlPhdrInfo,
        _size: usize,
        data: *mut c_void,
    ) -> i32 {
        if info.is_null() || data.is_null() {
            return 0;
        }
        let info_ref = unsafe { &*info };
        let state = unsafe { &mut *(data as *mut PhdrSearchState) };
        if info_ref.dlpi_phdr.is_null() || info_ref.dlpi_phnum == 0 {
            return 0;
        }

        let mut exec_min = usize::MAX;
        let mut exec_max = 0usize;
        let mut load_min = usize::MAX;
        let mut load_max = 0usize;

        let count = info_ref.dlpi_phnum as usize;
        for i in 0..count {
            let phdr = unsafe { &*info_ref.dlpi_phdr.add(i) };
            if phdr.p_type != PT_LOAD || phdr.p_memsz == 0 {
                continue;
            }
            let seg_start = info_ref.dlpi_addr.wrapping_add(phdr.p_vaddr as usize);
            let seg_end = seg_start.saturating_add(phdr.p_memsz as usize);
            if seg_start < load_min {
                load_min = seg_start;
            }
            if seg_end > load_max {
                load_max = seg_end;
            }
            if (phdr.p_flags & PF_X) != 0 {
                if seg_start < exec_min {
                    exec_min = seg_start;
                }
                if seg_end > exec_max {
                    exec_max = seg_end;
                }
            }
        }

        let matches_base = state.fbase != 0 && info_ref.dlpi_addr == state.fbase;
        let matches_fn =
            load_min < load_max && state.fn_addr >= load_min && state.fn_addr < load_max;
        if matches_base || matches_fn {
            if exec_min < exec_max {
                state.start = exec_min;
                state.end = exec_max;
            } else if load_min < load_max {
                state.start = load_min;
                state.end = load_max;
            }
            return 1;
        }
        0
    }

    #[cfg(target_os = "linux")]
    fn resolve_self_addr_range(fbase: usize, fn_addr: usize) -> (usize, usize) {
        let mut state = PhdrSearchState {
            fbase,
            fn_addr,
            start: 0,
            end: 0,
        };
        unsafe {
            dl_iterate_phdr(dl_phdr_callback, (&raw mut state) as *mut c_void);
        }
        if state.start != 0 && state.end > state.start {
            (state.start, state.end)
        } else if fbase != 0 {
            (fbase, fbase.saturating_add(FALLBACK_LIB_SPAN))
        } else {
            (0, 0)
        }
    }

    #[cfg(target_os = "macos")]
    fn resolve_self_addr_range(fbase: usize, _fn_addr: usize) -> (usize, usize) {
        if fbase == 0 {
            return (0, 0);
        }
        let header = unsafe { ptr::read_unaligned(fbase as *const MachHeader64) };
        if header.magic == MH_MAGIC_64 {
            let cmds_base = (fbase as *const u8).wrapping_add(std::mem::size_of::<MachHeader64>());
            let sizeofcmds = header.sizeofcmds as usize;
            let mut offset = 0usize;
            for _ in 0..header.ncmds {
                if offset.saturating_add(std::mem::size_of::<LoadCommand>()) > sizeofcmds {
                    break;
                }
                let cmd_ptr = cmds_base.wrapping_add(offset);
                let lc = unsafe { ptr::read_unaligned(cmd_ptr as *const LoadCommand) };
                let cmdsize = lc.cmdsize as usize;
                if cmdsize < std::mem::size_of::<LoadCommand>()
                    || offset.saturating_add(cmdsize) > sizeofcmds
                {
                    break;
                }
                if lc.cmd == LC_SEGMENT_64 && cmdsize >= std::mem::size_of::<SegmentCommand64>() {
                    let seg = unsafe { ptr::read_unaligned(cmd_ptr as *const SegmentCommand64) };
                    if &seg.segname[..6] == b"__TEXT" && seg.segname[6] == 0 && seg.vmsize > 0 {
                        return (fbase, fbase.saturating_add(seg.vmsize as usize));
                    }
                }
                offset += cmdsize;
            }
        }
        (fbase, fbase.saturating_add(FALLBACK_LIB_SPAN))
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    unsafe fn extract_pc(ucontext: *const c_void) -> Option<usize> {
        #[repr(C)]
        #[allow(dead_code)]
        struct StackT {
            ss_sp: *mut c_void,
            ss_flags: i32,
            ss_size: usize,
        }
        #[repr(C)]
        #[allow(dead_code)]
        struct MContext {
            gregs: [i64; 23],
        }
        #[repr(C)]
        #[allow(dead_code)]
        struct UContext {
            uc_flags: usize,
            uc_link: *mut c_void,
            uc_stack: StackT,
            uc_mcontext: MContext,
        }
        const REG_RIP: usize = 16;
        if ucontext.is_null() {
            return None;
        }
        let uc = unsafe { &*(ucontext as *const UContext) };
        Some(uc.uc_mcontext.gregs[REG_RIP] as usize)
    }

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    unsafe fn extract_pc(ucontext: *const c_void) -> Option<usize> {
        #[repr(C)]
        #[allow(dead_code)]
        struct StackT {
            ss_sp: *mut c_void,
            ss_flags: i32,
            ss_size: usize,
        }
        #[repr(C, align(16))]
        #[allow(dead_code)]
        struct MContext {
            fault_address: u64,
            regs: [u64; 31],
            sp: u64,
            pc: u64,
            pstate: u64,
        }
        #[repr(C)]
        #[allow(dead_code)]
        struct UContext {
            uc_flags: usize,
            uc_link: *mut c_void,
            uc_stack: StackT,
            uc_sigmask: [u64; 16],
            uc_mcontext: MContext,
        }
        if ucontext.is_null() {
            return None;
        }
        let uc = unsafe { &*(ucontext as *const UContext) };
        Some(uc.uc_mcontext.pc as usize)
    }

    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    unsafe fn extract_pc(ucontext: *const c_void) -> Option<usize> {
        #[repr(C)]
        #[allow(dead_code)]
        struct StackT {
            ss_sp: *mut c_void,
            ss_size: usize,
            ss_flags: i32,
        }
        #[repr(C)]
        #[allow(dead_code)]
        struct X86ExceptionState64 {
            trapno: u16,
            cpu: u16,
            err: u32,
            faultvaddr: u64,
        }
        #[repr(C)]
        #[allow(dead_code)]
        struct X86ThreadState64 {
            rax: u64,
            rbx: u64,
            rcx: u64,
            rdx: u64,
            rdi: u64,
            rsi: u64,
            rbp: u64,
            rsp: u64,
            r8: u64,
            r9: u64,
            r10: u64,
            r11: u64,
            r12: u64,
            r13: u64,
            r14: u64,
            r15: u64,
            rip: u64,
        }
        #[repr(C)]
        #[allow(dead_code)]
        struct MContext64 {
            es: X86ExceptionState64,
            ss: X86ThreadState64,
        }
        #[repr(C)]
        #[allow(dead_code)]
        struct UContext {
            uc_onstack: i32,
            uc_sigmask: u32,
            uc_stack: StackT,
            uc_link: *mut c_void,
            uc_mcsize: usize,
            uc_mcontext: *const MContext64,
        }
        if ucontext.is_null() {
            return None;
        }
        let uc = unsafe { &*(ucontext as *const UContext) };
        if uc.uc_mcontext.is_null() {
            return None;
        }
        Some(unsafe { (*uc.uc_mcontext).ss.rip as usize })
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    unsafe fn extract_pc(ucontext: *const c_void) -> Option<usize> {
        #[repr(C)]
        #[allow(dead_code)]
        struct StackT {
            ss_sp: *mut c_void,
            ss_size: usize,
            ss_flags: i32,
        }
        #[repr(C)]
        #[allow(dead_code)]
        struct ArmExceptionState64 {
            far: u64,
            esr: u32,
            exception: u32,
        }
        #[repr(C)]
        #[allow(dead_code)]
        struct ArmThreadState64 {
            x: [u64; 29],
            fp: u64,
            lr: u64,
            sp: u64,
            pc: u64,
            cpsr: u32,
            pad: u32,
        }
        #[repr(C)]
        #[allow(dead_code)]
        struct MContext64 {
            es: ArmExceptionState64,
            ss: ArmThreadState64,
        }
        #[repr(C)]
        #[allow(dead_code)]
        struct UContext {
            uc_onstack: i32,
            uc_sigmask: u32,
            uc_stack: StackT,
            uc_link: *mut c_void,
            uc_mcsize: usize,
            uc_mcontext: *const MContext64,
        }
        if ucontext.is_null() {
            return None;
        }
        let uc = unsafe { &*(ucontext as *const UContext) };
        if uc.uc_mcontext.is_null() {
            return None;
        }
        Some(unsafe { (*uc.uc_mcontext).ss.pc as usize })
    }

    #[cfg(not(any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        all(
            target_os = "macos",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    unsafe fn extract_pc(_ucontext: *const c_void) -> Option<usize> {
        None
    }

    fn pc_belongs_to_self(pc: usize) -> bool {
        let start = SELF_ADDR_START.load(Ordering::Acquire);
        let end = SELF_ADDR_END.load(Ordering::Acquire);
        start != 0 && end > start && pc >= start && pc < end
    }

    unsafe fn write_all_fd(fd: i32, bytes: &[u8]) {
        let mut offset = 0usize;
        while offset < bytes.len() {
            let written = unsafe {
                write(
                    fd,
                    bytes.as_ptr().add(offset) as *const c_void,
                    bytes.len() - offset,
                )
            };
            if written <= 0 {
                break;
            }
            offset += written as usize;
        }
    }

    unsafe fn chain_previous_handler(sig: i32, info: *mut SigInfo, ucontext: *mut c_void) {
        if let Some(slot) = signal_slot(sig) {
            let old = unsafe { *((&raw const OLD_ACTIONS) as *const SigAction).add(slot) };
            let handler_addr = old.sa_sigaction;
            if handler_addr != SIG_DFL && handler_addr != SIG_IGN {
                if (old.sa_flags & SA_SIGINFO) != 0 {
                    let prev_fn: unsafe extern "C" fn(i32, *mut SigInfo, *mut c_void) =
                        unsafe { std::mem::transmute(handler_addr) };
                    unsafe { prev_fn(sig, info, ucontext) };
                    return;
                } else {
                    let prev_fn: unsafe extern "C" fn(i32) =
                        unsafe { std::mem::transmute(handler_addr) };
                    unsafe { prev_fn(sig) };
                    return;
                }
            }
        }

        // Fall back to default OS disposition and re-raise.
        let mut dfl = SigAction::zeroed();
        dfl.sa_sigaction = SIG_DFL;
        unsafe {
            sigaction(sig, &dfl, ptr::null_mut());
            raise(sig);
        }
    }

    unsafe extern "C" fn unix_signal_handler(sig: i32, info: *mut SigInfo, ucontext: *mut c_void) {
        let (si_code, fault_addr) = if !info.is_null() {
            unsafe { ((*info).si_code, (*info).si_addr) }
        } else {
            (0, ptr::null())
        };
        let pc = unsafe { extract_pc(ucontext as *const c_void) };

        // The JVM handles its own faults, including illegal-instruction traps in JIT code.
        let should_log = match sig {
            SIGSEGV | SIGBUS | SIGILL => pc.is_some_and(pc_belongs_to_self),
            SIGABRT | SIGFPE => true,
            _ => false,
        };

        if should_log && !IN_CRASH_HANDLER.swap(true, Ordering::AcqRel) {
            let pid = unsafe { getpid() }.max(0) as u32;
            let mut path_buf = [0u8; CRASH_DIR_CAPACITY];
            if unsafe { build_crash_file_path(&mut path_buf, pid) } {
                let mut msg = StackBuf::<2048>::new();
                msg.push_str("Hyaloclast Native Crash Report (Unix)\n");
                msg.push_str("Signal: ");
                msg.push_str(signal_name(sig));
                msg.push_str(" (");
                msg.push_i32_dec(sig);
                msg.push_str(")\n");
                msg.push_str("si_code: ");
                msg.push_i32_dec(si_code);
                msg.push_str("\nInstruction Pointer: ");
                msg.push_usize_hex(pc.unwrap_or(0));
                msg.push_str("\nFault Address: ");
                msg.push_usize_hex(fault_addr as usize);
                msg.push_str("\nLibrary Base: ");
                msg.push_usize_hex(SELF_LIB_BASE.load(Ordering::Acquire));
                msg.push_str("\nLibrary Range: ");
                msg.push_usize_hex(SELF_ADDR_START.load(Ordering::Acquire));
                msg.push_str(" - ");
                msg.push_usize_hex(SELF_ADDR_END.load(Ordering::Acquire));
                msg.push_str("\nPID: ");
                msg.push_u32_dec(pid);
                msg.push_str("\n");

                let fd = unsafe {
                    open(
                        path_buf.as_ptr().cast(),
                        O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,
                        0o644u32,
                    )
                };
                if fd >= 0 {
                    unsafe {
                        write_all_fd(fd, msg.as_slice());
                        close(fd);
                    }
                }
            }
            IN_CRASH_HANDLER.store(false, Ordering::Release);
        }

        unsafe {
            chain_previous_handler(sig, info, ucontext);
        }
    }

    pub fn install() {
        let fn_addr = init_early as *const () as usize;
        let mut self_info = DlInfo {
            dli_fname: ptr::null(),
            dli_fbase: ptr::null(),
            dli_sname: ptr::null(),
            dli_saddr: ptr::null(),
        };
        let found = unsafe { dladdr(fn_addr as *const c_void, &mut self_info) };
        let fbase = if found != 0 && !self_info.dli_fbase.is_null() {
            let base = self_info.dli_fbase as usize;
            SELF_LIB_BASE.store(base, Ordering::Release);
            base
        } else {
            0
        };

        let (start, end) = resolve_self_addr_range(fbase, fn_addr);
        if start != 0 && end > start {
            SELF_ADDR_START.store(start, Ordering::Release);
            SELF_ADDR_END.store(end, Ordering::Release);
        }

        for (slot, &sig) in TRACKED_SIGNALS.iter().enumerate() {
            let mut new_act = SigAction::zeroed();
            new_act.sa_sigaction = unix_signal_handler as *const () as usize;
            new_act.sa_flags = SA_SIGINFO;

            let mut old_act = SigAction::zeroed();
            let rc = unsafe { sigaction(sig, &new_act, &mut old_act) };
            if rc == 0 {
                unsafe {
                    *((&raw mut OLD_ACTIONS) as *mut SigAction).add(slot) = old_act;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Windows Unhandled Exception Filter (NOT VEH — HotSpot-JVM-safe)
// ---------------------------------------------------------------------------

#[cfg(windows)]
mod platform_crash {
    use super::{CRASH_DIR_CAPACITY, IN_CRASH_HANDLER, StackBuf, build_crash_file_path_w};
    use std::ffi::c_void;
    use std::ptr;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
    const GENERIC_WRITE: u32 = 0x4000_0000;
    const FILE_SHARE_READ: u32 = 0x0000_0001;
    const CREATE_ALWAYS: u32 = 2;
    const FILE_ATTRIBUTE_NORMAL: u32 = 0x0000_0080;
    const INVALID_HANDLE_VALUE: isize = -1;

    #[repr(C)]
    #[allow(dead_code)]
    struct ExceptionRecord {
        exception_code: u32,
        exception_flags: u32,
        exception_record: *const ExceptionRecord,
        exception_address: *const c_void,
        number_parameters: u32,
        exception_information: [usize; 15],
    }

    #[repr(C)]
    #[allow(dead_code)]
    struct ExceptionPointers {
        exception_record: *const ExceptionRecord,
        context_record: *const c_void,
    }

    type TopLevelExceptionFilter =
        Option<unsafe extern "system" fn(*const ExceptionPointers) -> i32>;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn SetUnhandledExceptionFilter(
            lp_top_level_exception_filter: TopLevelExceptionFilter,
        ) -> TopLevelExceptionFilter;
        fn CreateFileW(
            lp_file_name: *const u16,
            dw_desired_access: u32,
            dw_share_mode: u32,
            lp_security_attributes: *const c_void,
            dw_creation_disposition: u32,
            dw_flags_and_attributes: u32,
            h_template_file: isize,
        ) -> isize;
        fn WriteFile(
            h_file: isize,
            lp_buffer: *const u8,
            n_number_of_bytes_to_write: u32,
            lp_number_of_bytes_written: *mut u32,
            lp_overlapped: *mut c_void,
        ) -> i32;
        fn CloseHandle(h_object: isize) -> i32;
        fn GetCurrentProcessId() -> u32;
    }

    static PREV_EXCEPTION_FILTER: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "system" fn windows_exception_filter(ex_ptrs: *const ExceptionPointers) -> i32 {
        if !IN_CRASH_HANDLER.swap(true, Ordering::AcqRel) {
            let (code, addr) = if !ex_ptrs.is_null() {
                let rec = unsafe { (*ex_ptrs).exception_record };
                if !rec.is_null() {
                    unsafe { ((*rec).exception_code, (*rec).exception_address as usize) }
                } else {
                    (0u32, 0usize)
                }
            } else {
                (0u32, 0usize)
            };

            let pid = unsafe { GetCurrentProcessId() };
            let mut path_buf = [0u16; CRASH_DIR_CAPACITY];
            if unsafe { build_crash_file_path_w(&mut path_buf, pid) } {
                let mut msg = StackBuf::<2048>::new();
                msg.push_str("Hyaloclast Native Crash Report (Windows)\n");
                msg.push_str("Exception Code: ");
                msg.push_u32_hex(code);
                msg.push_str("\nException Address: ");
                msg.push_usize_hex(addr);
                msg.push_str("\nPID: ");
                msg.push_u32_dec(pid);
                msg.push_str("\n");

                let handle = unsafe {
                    CreateFileW(
                        path_buf.as_ptr().cast(),
                        GENERIC_WRITE,
                        FILE_SHARE_READ,
                        ptr::null(),
                        CREATE_ALWAYS,
                        FILE_ATTRIBUTE_NORMAL,
                        0,
                    )
                };
                if handle != INVALID_HANDLE_VALUE && handle != 0 {
                    let mut written = 0u32;
                    let slice = msg.as_slice();
                    unsafe {
                        WriteFile(
                            handle,
                            slice.as_ptr(),
                            slice.len() as u32,
                            &mut written,
                            ptr::null_mut(),
                        );
                        CloseHandle(handle);
                    }
                }
            }
            IN_CRASH_HANDLER.store(false, Ordering::Release);
        }

        let prev = PREV_EXCEPTION_FILTER.load(Ordering::Acquire);
        if prev != 0 {
            let prev_fn: unsafe extern "system" fn(*const ExceptionPointers) -> i32 =
                unsafe { std::mem::transmute(prev) };
            return unsafe { prev_fn(ex_ptrs) };
        }
        EXCEPTION_CONTINUE_SEARCH
    }

    pub fn install() {
        let prev = unsafe { SetUnhandledExceptionFilter(Some(windows_exception_filter)) };
        if let Some(f) = prev {
            PREV_EXCEPTION_FILTER.store(f as usize, Ordering::Release);
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
mod platform_crash {
    pub fn install() {}
}

fn install_crash_handler() {
    platform_crash::install();
}

// ---------------------------------------------------------------------------
// Driver TDR / Device Lost Recovery & Structured Shader Compile Diagnostics
// ---------------------------------------------------------------------------

/// Structured shader compilation diagnostic for UI (`ShaderErrorLogScreen`) and FFI notices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderCompileDiagnostic {
    pub file: String,
    pub line: u32,
    pub column: u32,
    pub message: String,
}

impl ShaderCompileDiagnostic {
    /// Creates a new structured shader diagnostic.
    pub fn new(
        file: impl Into<String>,
        line: u32,
        column: u32,
        message: impl Into<String>,
    ) -> Self {
        Self {
            file: file.into(),
            line,
            column,
            message: message.into(),
        }
    }

    /// Parses common shaderc / glslang / naga error strings into a [`ShaderCompileDiagnostic`].
    ///
    /// Recognizes patterns such as:
    /// - `ERROR: path/to/file.comp:42:10: message`
    /// - `path/to/file.vsl:42: message`
    /// - `ERROR: 0:42: message` (substituting `fallback_file`)
    pub fn parse_from_compiler_error(fallback_file: &str, raw: &str) -> Self {
        let first_line = raw
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .unwrap_or("Unknown shader compilation error");

        let stripped = first_line
            .strip_prefix("ERROR:")
            .or_else(|| first_line.strip_prefix("error:"))
            .map(str::trim)
            .unwrap_or(first_line);

        let parts: Vec<&str> = stripped.splitn(4, ':').collect();
        if parts.len() >= 3 {
            let p0 = parts[0].trim();
            let p1 = parts[1].trim();
            if let Ok(line_num) = p1.parse::<u32>() {
                let file = if p0.is_empty() || p0.chars().all(|c| c.is_ascii_digit()) {
                    fallback_file.to_owned()
                } else {
                    p0.to_owned()
                };
                if parts.len() == 4 {
                    if let Ok(col_num) = parts[2].trim().parse::<u32>() {
                        return Self {
                            file,
                            line: line_num,
                            column: col_num,
                            message: parts[3].trim().to_owned(),
                        };
                    }
                    return Self {
                        file,
                        line: line_num,
                        column: 0,
                        message: format!("{}:{}", parts[2].trim(), parts[3])
                            .trim()
                            .to_owned(),
                    };
                }
                return Self {
                    file,
                    line: line_num,
                    column: 0,
                    message: parts[2].trim().to_owned(),
                };
            }
        }

        Self {
            file: fallback_file.to_owned(),
            line: 0,
            column: 0,
            message: first_line.to_owned(),
        }
    }

    /// Formats the diagnostic as `file:line:column: message` (omitting zero column/line cleanly).
    pub fn format_display(&self) -> String {
        match (self.line, self.column) {
            (0, _) => format!("{}: {}", self.file, self.message),
            (line, 0) => format!("{}:{}: {}", self.file, line, self.message),
            (line, col) => format!("{}:{}:{}: {}", self.file, line, col, self.message),
        }
    }
}

/// High-level reason why RT / VSP rendering entered safe fallback mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecoveryReason {
    #[default]
    None,
    DeviceLost,
    OutOfDeviceMemory,
    ShaderCompileFailure,
}

impl RecoveryReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::DeviceLost => "device_lost",
            Self::OutOfDeviceMemory => "out_of_device_memory",
            Self::ShaderCompileFailure => "shader_compile_failure",
        }
    }
}

/// Immutable snapshot of [`DeviceLostGuard`] state for FFI/UI notification.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RecoveryState {
    pub rt_vsp_disabled: bool,
    pub device_lost_count: u32,
    pub shader_failure_count: u32,
    pub reason: RecoveryReason,
    pub last_notice: Option<String>,
    pub last_shader_diagnostic: Option<ShaderCompileDiagnostic>,
}

#[derive(Debug, Default)]
struct DeviceLostGuardInner {
    reason: RecoveryReason,
    last_notice: Option<String>,
    last_shader_diagnostic: Option<ShaderCompileDiagnostic>,
}

/// Tracks driver TDR (`VK_ERROR_DEVICE_LOST`), OOM (`VK_ERROR_OUT_OF_DEVICE_MEMORY`), and shader
/// compilation failures, automatically tripping `rt_vsp_disabled` so the engine falls back cleanly
/// to safe vanilla/raster rendering without hard-crashing the JVM.
pub struct DeviceLostGuard {
    pub rt_vsp_disabled: AtomicBool,
    device_lost_count: AtomicU32,
    shader_failure_count: AtomicU32,
    inner: Mutex<DeviceLostGuardInner>,
}

impl Default for DeviceLostGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceLostGuard {
    pub const fn new() -> Self {
        Self {
            rt_vsp_disabled: AtomicBool::new(false),
            device_lost_count: AtomicU32::new(0),
            shader_failure_count: AtomicU32::new(0),
            inner: Mutex::new(DeviceLostGuardInner {
                reason: RecoveryReason::None,
                last_notice: None,
                last_shader_diagnostic: None,
            }),
        }
    }

    /// Lock-free hot-path check: returns `true` when RT/VSP passes must be skipped.
    #[inline]
    pub fn is_rt_vsp_disabled(&self) -> bool {
        self.rt_vsp_disabled.load(Ordering::Acquire)
    }

    /// Inspects a raw Vulkan result code (`ash::vk::Result`). If it is `ERROR_DEVICE_LOST` or
    /// `ERROR_OUT_OF_DEVICE_MEMORY`, trips `rt_vsp_disabled`, logs a native error, and returns `true`.
    pub fn record_vk_result(&self, context: &str, result: ash::vk::Result) -> bool {
        match result {
            ash::vk::Result::ERROR_DEVICE_LOST => {
                self.record_device_lost(context);
                true
            }
            ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                self.rt_vsp_disabled.store(true, Ordering::Release);
                let notice = format!(
                    "Vulkan out of device memory during {context}; disabled RT/VSP passes to prevent crash"
                );
                {
                    let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                    guard.reason = RecoveryReason::OutOfDeviceMemory;
                    guard.last_notice = Some(notice.clone());
                }
                push_native_error("ERROR", "gpu::recovery", &notice);
                true
            }
            _ => false,
        }
    }

    /// Records a `VK_ERROR_DEVICE_LOST` (driver TDR / GPU hang), disables RT/VSP passes, and
    /// queues a user-facing diagnostic notice.
    pub fn record_device_lost(&self, context: &str) {
        self.rt_vsp_disabled.store(true, Ordering::Release);
        let count = self.device_lost_count.fetch_add(1, Ordering::AcqRel) + 1;
        let notice = format!(
            "GPU driver reset (VK_ERROR_DEVICE_LOST #{count}) during {context}; RT/VSP passes disabled for safe fallback"
        );
        {
            let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            guard.reason = RecoveryReason::DeviceLost;
            guard.last_notice = Some(notice.clone());
        }
        push_native_error("ERROR", "gpu::tdr", &notice);
    }

    /// Records a structured shader compile failure, disables RT/VSP passes until the next clean
    /// shader pack reload, and queues a diagnostic notice for `ShaderNotices` / UI.
    pub fn record_shader_compile_failure(&self, diagnostic: ShaderCompileDiagnostic) {
        self.rt_vsp_disabled.store(true, Ordering::Release);
        self.shader_failure_count.fetch_add(1, Ordering::AcqRel);
        let formatted = diagnostic.format_display();
        let notice = format!("Shader compile failed ({formatted}); falling back to safe rendering");
        {
            let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            guard.reason = RecoveryReason::ShaderCompileFailure;
            guard.last_notice = Some(notice.clone());
            guard.last_shader_diagnostic = Some(diagnostic);
        }
        push_native_error("ERROR", "shader::compile", &notice);
    }

    /// Clears the fallback state (for example, after a user reloads a valid shader pack or
    /// recreates the Vulkan device).
    pub fn reset(&self) {
        self.rt_vsp_disabled.store(false, Ordering::Release);
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        guard.reason = RecoveryReason::None;
        guard.last_notice = None;
        guard.last_shader_diagnostic = None;
    }

    /// Returns a complete [`RecoveryState`] snapshot safe to expose over FFI.
    pub fn snapshot(&self) -> RecoveryState {
        let guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        RecoveryState {
            rt_vsp_disabled: self.rt_vsp_disabled.load(Ordering::Acquire),
            device_lost_count: self.device_lost_count.load(Ordering::Acquire),
            shader_failure_count: self.shader_failure_count.load(Ordering::Acquire),
            reason: guard.reason,
            last_notice: guard.last_notice.clone(),
            last_shader_diagnostic: guard.last_shader_diagnostic.clone(),
        }
    }

    /// Returns the active user-facing notice string for FFI/UI (`ShaderNotices`) if in fallback mode.
    pub fn format_ffi_notice(&self) -> Option<String> {
        if !self.is_rt_vsp_disabled() {
            return None;
        }
        let guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        guard.last_notice.clone()
    }
}

static GLOBAL_DEVICE_LOST_GUARD: LazyLock<DeviceLostGuard> = LazyLock::new(DeviceLostGuard::new);

/// Returns the process-wide [`DeviceLostGuard`] singleton.
pub fn global_device_lost_guard() -> &'static DeviceLostGuard {
    &GLOBAL_DEVICE_LOST_GUARD
}
