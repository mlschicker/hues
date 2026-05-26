use crate::{
    Coeff,
    bounds::LowerBound,
    solver::bnb::{SearchState, cutting_planes::CoverCutDomain},
};

/// Log a table header if not already printed. The header remains until the end of the search, when `log_table_footer` is called to print a footer.
pub(crate) fn log_table_header() {
    log::info!(target: "table",
        "┌────────────┬──────────┬──────────┬──────────┬──────────┬────────────────┬────────────────┬──────────┬─────────┐"
    );
    log::info!(target: "table",
        "│ event      │ explored │ frontier │ pruned   │ leaves   │ incumbent      │ best_bound     │ gap %    │ time(s) │"
    );
    log::info!(target: "table",
        "├────────────┼──────────┼──────────┼──────────┼──────────┼────────────────┼────────────────┼──────────┼─────────┤"
    );
}

/// Log a table footer if the header was printed, to visually close the log output.
pub(crate) fn log_table_footer() {
    log::info!(target: "table",
        "└────────────┴──────────┴──────────┴──────────┴──────────┴────────────────┴────────────────┴──────────┴─────────┘"
    );
}

/// Format the gap between incumbent and best bound as a percentage string, or "n/a" if no incumbent.
pub(crate) fn format_gap<C: Coeff>(incumbent: Option<C>, best_bound: C) -> String {
    if let Some(inc) = incumbent {
        let incf = inc.to_f64();
        let bdf = best_bound.to_f64();
        let denom = incf.abs().max(bdf.abs()).max(1e-12);
        let mut gap = ((incf - bdf).max(0.0) / denom) * 100.0;
        if !gap.is_finite() {
            gap = 0.0;
        }
        format!("{gap:>7.2}")
    } else {
        "    n/a".to_string()
    }
}

/// Format a coefficient for display in the log table, using scientific notation for very large/small values and trimming trailing zeros.
pub(crate) fn fmt_coeff<C: Coeff>(value: C, width: usize) -> String {
    let v = value.to_f64();
    let abs = v.abs();
    let mut s = if abs >= 1e8 || (abs > 0.0 && abs < 1e-5) {
        format!("{v:.4e}")
    } else {
        let mut t = format!("{v:.6}");
        while t.contains('.') && t.ends_with('0') {
            t.pop();
        }
        if t.ends_with('.') {
            t.pop();
        }
        t
    };
    if s.len() > width {
        s = format!("{v:.3e}");
    }
    if s.len() > width {
        s.truncate(width);
    }
    format!("{s:>width$}")
}

/// Log a row in the progress table with the given event label, current global lower bound, and frontier size.
pub(crate) fn log_table_row<C: Coeff, V: CoverCutDomain, Lb: LowerBound>(
    state: &SearchState<C, V, Lb>,
    event: &str,
    current_lb: C,
    frontier_size: usize,
) {
    let elapsed = state.start.elapsed().as_secs_f64();
    let incumbent = state
        .incumbent_obj
        .map(|v| fmt_coeff(v, 14))
        .unwrap_or_else(|| format!("{:>14}", "n/a"));
    let best_bound_str = fmt_coeff(current_lb, 14);
    let gap = format_gap(state.incumbent_obj, current_lb);
    log::info!(target: "table",
        "│ {:<10} │ {:>8} │ {:>8} │ {:>8} │ {:>8} │ {:>14} │ {:>14} │ {:>8} │ {:>7.3} │",
        event,
        state.explored_nodes,
        frontier_size,
        state.pruned_nodes,
        state.leaf_nodes,
        incumbent,
        best_bound_str,
        gap,
        elapsed
    );
}
