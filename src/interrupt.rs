//! Global SIGINT (Ctrl+C) interrupt flag.
//!
//! Call [`install_handler`] once at startup.  Any solver loop can then call
//! [`is_interrupted`] to check whether the user has pressed Ctrl+C and should
//! stop early, returning its best solution found so far.

use std::sync::atomic::{AtomicBool, Ordering};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// Returns `true` if a SIGINT (Ctrl+C) has been received.
#[inline]
pub fn is_interrupted() -> bool {
    INTERRUPTED.load(Ordering::Relaxed)
}

/// Install the Ctrl+C handler.
///
/// On the first Ctrl+C the flag is set and a warning is logged so solvers can
/// wind down gracefully.  A second Ctrl+C immediately terminates the process
/// (exit code 130) so the user is never fully stuck.
pub fn install_handler() {
    if let Err(e) = ctrlc::set_handler(|| {
        if INTERRUPTED.swap(true, Ordering::Relaxed) {
            // Second Ctrl+C — force exit immediately.
            std::process::exit(130);
        }
        // Use eprintln here because the logger may not be set up yet or may
        // be directing output to a file.
        eprintln!("\nInterrupted — returning best solution found so far (press Ctrl+C again to force quit)...");
    }) {
        log::warn!("failed to install Ctrl+C handler: {e}");
    }
}
