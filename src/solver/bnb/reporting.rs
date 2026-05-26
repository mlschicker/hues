use super::*;

pub(super) fn csv_escape(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

pub(super) fn append_stats_csv<C: Coeff, V: VarDomain, Lb>(
    path: &str,
    instance: &HuboInstance<C, V>,
    config: &Config<Lb>,
    result: &SolveResult<C>,
) -> io::Result<()> {
    let path_ref = Path::new(path);
    let write_header = std::fs::metadata(path_ref).map_or(true, |m| m.len() == 0);
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path_ref)?;
    if write_header {
        writeln!(
            file,
            "instance,var_type,n_vars,n_terms,status,objective,best_bound,time_s,tts_s,\
             explored_nodes,unexplored_nodes,pruned_nodes,cutoff,time_limit,node_limit"
        )?;
    }
    let instance_name = config.instance_name.as_deref().unwrap_or("n/a");
    let var_type = if V::VAR_TYPE == VarType::Bin {
        "BIN"
    } else {
        "SPIN"
    };
    let objective = result
        .objective
        .map(|v| v.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let tts = result
        .tts
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "n/a".to_string());
    let cutoff = config
        .cutoff
        .map(|v| v.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let time_limit = config
        .time_limit
        .map(|v| v.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let node_limit = config
        .node_limit
        .map(|v| v.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    writeln!(
        file,
        "{},{},{},{},{:?},{},{},{:.6},{},{},{},{},{},{},{}",
        csv_escape(instance_name),
        var_type,
        instance.n_vars(),
        instance.n_terms(),
        result.status,
        objective,
        result.best_bound,
        result.solving_time,
        tts,
        result.n_nodes,
        result.unexplored_nodes,
        result.pruned_nodes,
        cutoff,
        time_limit,
        node_limit
    )?;
    Ok(())
}

// ── Internal search outcome ────────────────────────────────────────────────

