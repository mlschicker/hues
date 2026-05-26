use std::collections::HashMap;

use std::sync::Arc;

use crate::coeff::Coeff;
use crate::fixes::Fixes;
use crate::kernelization::error::KernelizationError;
use crate::solver::bnb::Node;
use crate::{
    domain::{VarDomain, VarType},
    instance::HuboInstance,
};

pub use coupling::{
    ConditionalCoupling, ConditionalCouplingKind, coupling_fixes,
    detect_conditionally_coupled_pairs, detect_frustrated_pairs, detect_uncoupled_pairs,
};

pub use dominance::dominance_fixes;

pub use roof_dual::{binary_roof_duality, spin_roof_duality};

mod coupling;
mod dominance;
mod error;
mod roof_dual;
pub mod symmetry;
mod util;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelizationRuleId {
    ExternalFixes,
    Dominance,
    RoofDuality,
    SpinCouplingReduction,
    SflipSymmetry,
}

pub trait KernelizationMethod<C: Coeff, V: VarDomain> {
    fn id(&self) -> KernelizationRuleId;
    fn apply(&self, instance: &HuboInstance<C, V>, node: &Node<C>) -> Vec<(usize, C)>;
}

macro_rules! simple_kernelization_method {
    ($ty:ident, $id:expr, $func:path) => {
        #[derive(Debug, Clone, Copy)]
        struct $ty;

        impl<C: Coeff, V: VarDomain> KernelizationMethod<C, V> for $ty {
            fn id(&self) -> KernelizationRuleId {
                $id
            }

            fn apply(&self, instance: &HuboInstance<C, V>, node: &Node<C>) -> Vec<(usize, C)> {
                $func(instance, node)
            }
        }
    };
}

simple_kernelization_method!(
    RoofDualityMethod,
    KernelizationRuleId::RoofDuality,
    roof_dual::roof_dual_fixes
);
simple_kernelization_method!(
    SflipSymmetryMethod,
    KernelizationRuleId::SflipSymmetry,
    symmetry::sflip_symmetry_fixes
);
simple_kernelization_method!(
    DominanceMethod,
    KernelizationRuleId::Dominance,
    dominance::dominance_fixes
);

#[derive(Debug, Clone, Copy)]
struct SpinCouplingReductionMethod;
impl<C: Coeff, V: VarDomain> KernelizationMethod<C, V> for SpinCouplingReductionMethod {
    fn id(&self) -> KernelizationRuleId {
        KernelizationRuleId::SpinCouplingReduction
    }

    fn apply(&self, instance: &HuboInstance<C, V>, node: &Node<C>) -> Vec<(usize, C)> {
        if V::VAR_TYPE != VarType::Spin {
            return Vec::new();
        }

        let unconditional_pairs =
            coupling::detect_uncoupled_pairs_active(instance, &node.term_status, &node.fixed);

        if unconditional_pairs.is_empty() {
            Vec::new()
        } else {
            coupling::coupling_fixes(&unconditional_pairs)
        }
    }
}

static DOMINANCE_METHOD: DominanceMethod = DominanceMethod;
static ROOF_DUALITY_METHOD: RoofDualityMethod = RoofDualityMethod;
static SPIN_COUPLING_METHOD: SpinCouplingReductionMethod = SpinCouplingReductionMethod;
static SFLIP_SYMMETRY_METHOD: SflipSymmetryMethod = SflipSymmetryMethod;

#[derive(Debug, Clone)]
pub struct KernelizationConfig {
    pub enable_binary_dominance: bool,
    pub enable_spin_dominance: bool,
    /// Roof-duality / QPBO-style persistency for binary HUBO: builds the
    /// implication graph on the quadratic skeleton and propagates strongly
    /// persistent labels.
    pub enable_binary_roof_duality: bool,
    /// Roof-duality / QPBO for spin HUBO: transforms the problem to binary via
    /// `s_i = 2x_i − 1`, builds the s-t flow graph, and propagates persistent
    /// labels.
    pub enable_spin_roof_duality: bool,
    pub remove_zero_coeff_terms: bool,
    pub max_rounds: usize,
    /// Detect unconditionally coupled spin variable pairs (Theorem 2.31) and
    /// eliminate both members of each pair from the instance.  Only applies to
    /// spin-domain instances.
    pub enable_coupling_reduction: bool,
    pub enable_sflip_symmetry: bool,
}

impl Default for KernelizationConfig {
    fn default() -> Self {
        Self {
            enable_binary_dominance: true,
            enable_spin_dominance: true,
            enable_binary_roof_duality: true,
            enable_spin_roof_duality: true,
            remove_zero_coeff_terms: true,
            max_rounds: 32,
            enable_coupling_reduction: false,
            enable_sflip_symmetry: true,
        }
    }
}

impl KernelizationConfig {
    pub fn none() -> Self {
        Self {
            enable_binary_dominance: false,
            enable_spin_dominance: false,
            enable_binary_roof_duality: false,
            enable_spin_roof_duality: false,
            remove_zero_coeff_terms: false,
            max_rounds: 0,
            enable_coupling_reduction: false,
            enable_sflip_symmetry: false,
        }
    }

    pub fn node_level() -> Self {
        Self {
            enable_binary_dominance: true,
            enable_spin_dominance: true,
            enable_binary_roof_duality: false,
            enable_spin_roof_duality: false,
            remove_zero_coeff_terms: true,
            max_rounds: 32,
            enable_coupling_reduction: false,
            enable_sflip_symmetry: false,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct KernelizationReport {
    pub initial_n_vars: usize,
    pub final_n_vars: usize,
    pub initial_n_terms: usize,
    pub final_n_terms: usize,
    pub rounds: usize,
    pub externally_fixed: usize,
    /// Total variables fixed by kernelization rules (excludes external fixes).
    pub rule_fixed: usize,
    /// Variables fixed by dominance.
    pub dominance_fixed: usize,
    /// Variables fixed by LB-gap fixing.
    pub lb_fixing_fixed: usize,
    /// Variables fixed by roof-duality / QPBO.
    pub roof_duality_fixed: usize,
    /// Variables eliminated by spin coupling reduction (Theorem 2.31).
    /// Each detected coupled pair eliminates 2 variables.
    pub coupling_fixed: usize,
    /// Number of unconditional spin-coupling pairs detected.
    pub unconditional_coupling_pairs_detected: usize,
    /// Number of conditional spin-coupling pairs detected.
    pub conditional_coupling_pairs_detected: usize,
    /// Variables fixed via conditional-coupling propagation from same-round fixes.
    pub conditional_coupling_implied_fixed: usize,
    /// Number of frustrated spin pairs detected.
    pub frustrated_pairs_detected: usize,
    /// Variables fixed by S-flip symmetry.
    pub sflip_symmetry_fixed: usize,
}

impl KernelizationReport {
    pub fn verbose_lines(&self) -> Vec<String> {
        let fixed_total = self.initial_n_vars.saturating_sub(self.final_n_vars);
        let reduction_pct = if self.initial_n_vars == 0 {
            0.0
        } else {
            100.0 * fixed_total as f64 / self.initial_n_vars as f64
        };

        let mut lines = vec![
            format!(
                "kernelization summary: rounds={}, vars {} -> {} ({}/{} fixed, {:.1}%)",
                self.rounds,
                self.initial_n_vars,
                self.final_n_vars,
                fixed_total,
                self.initial_n_vars,
                reduction_pct
            ),
            format!(
                "  external_fixed={} | dominance={} | lb_fixing={} | roof_duality={} | sflip={} | coupling={}",
                self.externally_fixed,
                self.dominance_fixed,
                self.lb_fixing_fixed,
                self.roof_duality_fixed,
                self.sflip_symmetry_fixed,
                self.coupling_fixed
            ),
        ];

        if self.coupling_fixed > 0 {
            lines.push(format!(
                "  coupling details: unconditional_pairs={} | conditional_pairs={} | conditional_implied={} | frustrated_pairs={}",
                self.unconditional_coupling_pairs_detected,
                self.conditional_coupling_pairs_detected,
                self.conditional_coupling_implied_fixed,
                self.frustrated_pairs_detected
            ));
        }

        lines
    }
}

#[derive(Debug, Clone)]
pub struct KernelizationResult {
    pub fixes: Fixes,
    pub report: KernelizationReport,
}

impl KernelizationResult {
    pub fn print_report(&self) {
        let total_fixed = self.fixes.assigned.ones().count();
        let mut fixed_preview: Vec<String> = self
            .fixes
            .assigned
            .ones()
            .map(|idx| format!("{idx}={}", self.fixes.values[idx]))
            .take(12)
            .collect();
        if total_fixed > fixed_preview.len() {
            fixed_preview.push(format!("... (+{} more)", total_fixed - fixed_preview.len()));
        }
        log::info!(
            "kernelization: vars {} -> {}, terms {} -> {}, rounds={}, \
                         fixed_external={}, fixed_rules={}, fixed_total={}",
            self.report.initial_n_vars,
            self.report.final_n_vars,
            self.report.initial_n_terms,
            self.report.final_n_terms,
            self.report.rounds,
            self.report.externally_fixed,
            self.report.rule_fixed,
            total_fixed
        );
        if !fixed_preview.is_empty() {
            log::info!(
                "kernelization fixed variables: {}",
                fixed_preview.join(", ")
            );
        }
        for line in self.report.verbose_lines() {
            log::info!("{line}");
        }
    }
}

pub struct Kernelizer {
    config: KernelizationConfig,
}

impl Default for Kernelizer {
    fn default() -> Self {
        Self::new(KernelizationConfig::default())
    }
}

impl Kernelizer {
    pub fn new(config: KernelizationConfig) -> Self {
        Self { config }
    }

    /// Apply kernelization to the given term-status and fixes directly.
    ///
    /// `incumbent` is optional; when provided, LB-gap fixing is run after
    /// structural rules each round.
    pub fn kernelize<C: Coeff, V: VarDomain>(
        &self,
        instance: &Arc<HuboInstance<C, V>>,
        node: &mut Node<C>,
        incumbent: Option<C>,
    ) -> Result<KernelizationReport, KernelizationError> {
        let free_vars_before = node.fixed.num_free();
        let free_terms_before = node.term_status.iter().flatten().count();

        log::debug!(
            "kernelization start: var_type={:?}, vars={}, terms={}, max_rounds={}",
            V::VAR_TYPE,
            free_vars_before,
            free_terms_before,
            self.config.max_rounds
        );

        let mut rounds: usize = 0;
        let externally_fixed: usize = 0;
        let mut rule_fixed: usize = 0;
        let mut dominance_fixed: usize = 0;
        let mut lb_fixing_fixed: usize = 0;
        let mut roof_duality_fixed: usize = 0;
        let mut coupling_fixed: usize = 0;
        let unconditional_coupling_pairs_detected: usize = 0;
        let mut conditional_coupling_pairs_detected: usize = 0;
        let mut conditional_coupling_implied_fixed: usize = 0;
        let frustrated_pairs_detected: usize = 0;
        let mut sflip_symmetry_fixed: usize = 0;

        for _ in 0..self.config.max_rounds {
            let mut rules: Vec<&dyn KernelizationMethod<C, V>> = Vec::new();

            // initialize list of rules to be applied
            match V::VAR_TYPE {
                VarType::Bin => {
                    if self.config.enable_binary_dominance {
                        rules.push(&DOMINANCE_METHOD);
                    }
                    if self.config.enable_binary_roof_duality {
                        rules.push(&ROOF_DUALITY_METHOD);
                    }
                }
                VarType::Spin => {
                    if self.config.enable_coupling_reduction {
                        rules.push(&SPIN_COUPLING_METHOD);
                    }
                    if self.config.enable_spin_dominance {
                        rules.push(&DOMINANCE_METHOD);
                    }
                    if self.config.enable_spin_roof_duality {
                        rules.push(&ROOF_DUALITY_METHOD);
                    }
                    if self.config.enable_sflip_symmetry {
                        rules.push(&SFLIP_SYMMETRY_METHOD);
                    }
                }
            }

            log::debug!(
                "kernelization round {}: vars={}, terms={}, enabled_rules={}",
                rounds + 1,
                instance.n_vars(),
                instance.n_terms(),
                rules.len()
            );

            let conditional_pairs =
                if V::VAR_TYPE == VarType::Spin && self.config.enable_coupling_reduction {
                    let conditional = coupling::detect_conditionally_coupled_pairs_active(
                        instance,
                        &node.term_status,
                        &node.fixed,
                    );
                    conditional_coupling_pairs_detected += conditional.len();
                    conditional
                } else {
                    Vec::new()
                };

            let mut applied = false;

            for rule in rules {
                let candidate_fixes: Vec<(usize, C)> = rule
                    .apply(instance, node)
                    .into_iter()
                    .filter(|(idx, _)| !node.fixed.assigned.contains(*idx))
                    .collect();

                if candidate_fixes.is_empty() {
                    continue;
                }

                let mut candidate_fixes = candidate_fixes;
                if V::VAR_TYPE == VarType::Spin && !conditional_pairs.is_empty() {
                    let before = candidate_fixes.len();
                    let mut known: HashMap<usize, C> = candidate_fixes.iter().copied().collect();
                    let mut changed = true;

                    while changed {
                        changed = false;
                        for coupling in &conditional_pairs {
                            match coupling.kind {
                                coupling::ConditionalCouplingKind::JGivenI => {
                                    if let Some(si) = known.get(&coupling.i).copied()
                                        && !known.contains_key(&coupling.j)
                                        && !node.fixed.assigned.contains(coupling.j)
                                    {
                                        known.insert(coupling.j, coupling.alpha * si);
                                        changed = true;
                                    }
                                }
                                coupling::ConditionalCouplingKind::IGivenJ => {
                                    if let Some(sj) = known.get(&coupling.j).copied()
                                        && !known.contains_key(&coupling.i)
                                        && !node.fixed.assigned.contains(coupling.i)
                                    {
                                        known.insert(coupling.i, coupling.alpha * sj);
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }

                    candidate_fixes = known.into_iter().collect();
                    candidate_fixes.sort_by_key(|(idx, _)| *idx);
                    let implied = candidate_fixes.len().saturating_sub(before);
                    conditional_coupling_implied_fixed += implied;
                }

                // Apply fixes to both current_fixes and term_status.

                apply_fixes(instance, node, &candidate_fixes)?;

                rounds += 1;
                rule_fixed += candidate_fixes.len();

                log::debug!(
                    "kernelization round {}: applied {:?}, fixed {} vars",
                    rounds,
                    rule.id(),
                    candidate_fixes.len(),
                );

                match rule.id() {
                    KernelizationRuleId::Dominance => {
                        dominance_fixed += candidate_fixes.len();
                    }
                    KernelizationRuleId::RoofDuality => {
                        roof_duality_fixed += candidate_fixes.len();
                    }
                    KernelizationRuleId::SpinCouplingReduction => {
                        coupling_fixed += candidate_fixes.len();
                        // unconditional_coupling_pairs_detected +=
                        //     rule_application.unconditional_coupling_pairs_detected;
                        // frustrated_pairs_detected += rule_application.frustrated_pairs_detected;
                    }
                    KernelizationRuleId::SflipSymmetry => {
                        sflip_symmetry_fixed += candidate_fixes.len();
                    }
                    KernelizationRuleId::ExternalFixes => {}
                }

                applied = true;
                break;
            }

            // LB-gap fixing: run when no structural rule fired and an incumbent is known.
            if !applied {
                if let Some(inc) = incumbent {
                    match dominance::lb_fixing(instance, node, inc) {
                        None => {
                            // Both values of some variable are pruned → infeasible.
                            node.lb = C::max_value();
                            break;
                        }
                        Some(fixes) if !fixes.is_empty() => {
                            apply_fixes(instance, node, &fixes)?;
                            rounds += 1;
                            lb_fixing_fixed += fixes.len();
                            rule_fixed += fixes.len();
                            applied = true;
                        }
                        Some(_) => {}
                    }
                }
            }

            if !applied {
                break;
            }
        }

        let free_vars_after = node.fixed.num_free();
        let free_terms_after = node.term_status.iter().flatten().count();

        Ok(KernelizationReport {
            initial_n_vars: free_vars_before,
            final_n_vars: free_vars_after,
            initial_n_terms: free_terms_before,
            final_n_terms: free_terms_after,
            rounds,
            externally_fixed,
            rule_fixed,
            dominance_fixed,
            lb_fixing_fixed,
            roof_duality_fixed,
            coupling_fixed,
            unconditional_coupling_pairs_detected,
            conditional_coupling_pairs_detected,
            conditional_coupling_implied_fixed,
            frustrated_pairs_detected,
            sflip_symmetry_fixed,
        })
    }
}

fn apply_fixes<C: Coeff, V: VarDomain>(
    instance: &Arc<HuboInstance<C, V>>,
    node: &mut Node<C>,
    new_fixes: &[(usize, C)],
) -> Result<(), KernelizationError> {
    for (index, value) in new_fixes {
        node.set_variable(instance, *index, *value == C::one())
            .map_err(|err| KernelizationError::FixingError {
                source: Box::new(err),
            })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::HuboModel,
        solver::bnb::{ConstraintHandler, PartiallyAssignedTerm},
    };

    #[test]
    fn external_fix_reduces_and_lifts_binary_solution() {
        let instance = HuboModel::binary(3)
            .add_linear(0, 1.0)
            .add_linear(1, 2.0)
            .add_term(&[0, 2], -3.0)
            .build();
        let term_status: Vec<_> = instance
            .terms
            .iter()
            .map(|term| Some(PartiallyAssignedTerm::new(term)))
            .collect();
        let fixed = Fixes::new(instance.n_vars());
        let mut node = Node {
            fixed,
            lb: 0.0,
            offset: 0.0,
            term_status,
            term_by_free_vars: None,
            local_constraints: ConstraintHandler::new(),
            lb_warm_start: None,
        };

        let kernelizer = Kernelizer::new(KernelizationConfig {
            enable_binary_dominance: false,
            enable_spin_dominance: false,
            enable_binary_roof_duality: false,
            enable_spin_roof_duality: false,
            remove_zero_coeff_terms: true,
            max_rounds: 1,
            enable_coupling_reduction: false,
            enable_sflip_symmetry: false,
        });

        let _ = kernelizer
            .kernelize(&Arc::new(instance), &mut node, None)
            .unwrap();
    }

    #[test]
    fn iterative_kernelization_composes_trace() {
        let instance = HuboModel::binary(2)
            .add_linear(0, -4.0)
            .add_linear(1, 5.0)
            .add_term(&[0, 1], -10.0)
            .build();
        let term_status: Vec<_> = instance
            .terms
            .iter()
            .map(|term| Some(PartiallyAssignedTerm::new(term)))
            .collect();
        let fixed = Fixes::new(instance.n_vars());
        let mut node = Node {
            fixed,
            lb: 0.0,
            offset: 0.0,
            term_status,
            term_by_free_vars: None,
            local_constraints: ConstraintHandler::new(),
            lb_warm_start: None,
        };

        let kernelizer = Kernelizer::default();

        let report = kernelizer
            .kernelize(&Arc::new(instance), &mut node, None)
            .unwrap();

        assert_eq!(report.rounds, 2);
        assert_eq!(report.dominance_fixed, 2);

        // assert_eq!(full, vec![1.0, 1.0]);
    }

    // -------------------------------------------------------------------
    // Tests for first-order persistency
    // -------------------------------------------------------------------

    /// 1st-order persistency: when we probe x0=0 and x0=1 in
    ///   f = 10*x0 - 10*x0*x1 + 5*x1
    /// we get:
    ///   f|_{x0=0} = 5*x1          → minimised by x1=0
    // -------------------------------------------------------------------
    // Tests for roof duality / QPBO
    // -------------------------------------------------------------------

    /// For f = x0 + x1 - 3*x0*x1, roof duality should detect that
    /// (x0=1, x1=1) is always optimal (both fixed to 1).
    #[test]
    fn roof_duality_fixes_consistent_labels() {
        // f(x) = x0 + x1 - 3*x0*x1.
        // Values: (0,0)=0, (0,1)=1, (1,0)=1, (1,1)=-1.  Min at (1,1).
        let instance = HuboModel::binary(2)
            .add_linear(0, 1.0)
            .add_linear(1, 1.0)
            .add_term(&[0, 1], -3.0)
            .build();
        let term_status: Vec<_> = instance
            .terms
            .iter()
            .map(|term| Some(PartiallyAssignedTerm::new(term)))
            .collect();
        let fixed = Fixes::new(instance.n_vars());
        let mut node = Node {
            fixed,
            lb: 0.0,
            offset: 0.0,
            term_status,
            term_by_free_vars: None,
            local_constraints: ConstraintHandler::new(),
            lb_warm_start: None,
        };

        let kernelizer = Kernelizer::new(KernelizationConfig {
            enable_binary_dominance: false,
            enable_spin_dominance: false,
            enable_binary_roof_duality: true,
            enable_spin_roof_duality: false,
            remove_zero_coeff_terms: true,
            max_rounds: 10,
            enable_coupling_reduction: false,
            enable_sflip_symmetry: false,
        });
        let _ = kernelizer
            .kernelize(&Arc::new(instance), &mut node, None)
            .unwrap();
    }

    // -------------------------------------------------------------------
    // Integration tests: spin persistency through the full kernelizer
    // -------------------------------------------------------------------

    #[test]
    fn spin_roof_duality_via_kernelizer() {
        // f = s0 + s1 - 3*s0*s1.  Min at (-1,-1).
        let instance = HuboModel::spin(2)
            .add_linear(0, 1.0)
            .add_linear(1, 1.0)
            .add_term(&[0, 1], -3.0)
            .build();
        let term_status: Vec<_> = instance
            .terms
            .iter()
            .map(|term| Some(PartiallyAssignedTerm::new(term)))
            .collect();
        let fixed = Fixes::new(instance.n_vars());
        let mut node = Node {
            fixed,
            lb: 0.0,
            offset: 0.0,
            term_status,
            term_by_free_vars: None,
            local_constraints: ConstraintHandler::new(),
            lb_warm_start: None,
        };

        let kernelizer = Kernelizer::new(KernelizationConfig {
            enable_binary_dominance: false,
            enable_spin_dominance: false,
            enable_binary_roof_duality: false,
            enable_spin_roof_duality: true,
            remove_zero_coeff_terms: true,
            max_rounds: 10,
            enable_coupling_reduction: false,
            enable_sflip_symmetry: false,
        });
        let report = kernelizer
            .kernelize(&Arc::new(instance), &mut node, None)
            .unwrap();

        assert!(report.roof_duality_fixed >= 1);
    }

    #[test]
    fn spin_coupling_reduction_via_kernelizer() {
        // f = 3*s0*s1 + s0*s1*s2 has A = 3 + s2 > 0, so s1 = +s0 can be
        // substituted safely and both variables are eliminated.
        let instance = HuboModel::spin(3)
            .add_term(&[0, 1], 3.0)
            .add_term(&[0, 1, 2], 1.0)
            .build();
        let term_status: Vec<_> = instance
            .terms
            .iter()
            .map(|term| Some(PartiallyAssignedTerm::new(term)))
            .collect();
        let fixed = Fixes::new(instance.n_vars());
        let mut node = Node {
            fixed,
            lb: 0.0,
            offset: 0.0,
            term_status,
            term_by_free_vars: None,
            local_constraints: ConstraintHandler::new(),
            lb_warm_start: None,
        };

        let kernelizer = Kernelizer::new(KernelizationConfig {
            enable_binary_dominance: false,
            enable_spin_dominance: false,
            enable_binary_roof_duality: false,
            enable_spin_roof_duality: false,
            remove_zero_coeff_terms: true,
            max_rounds: 10,
            enable_coupling_reduction: true,
            enable_sflip_symmetry: false,
        });

        let report = kernelizer
            .kernelize(&Arc::new(instance), &mut node, None)
            .unwrap();
        assert_eq!(report.coupling_fixed, 2);
        assert_eq!(report.rule_fixed, 2);
    }

    #[test]
    fn conditional_coupling_propagates_same_round_fixes() {
        // s0 is fixed by dominance to -1 via +5*s0.
        // Pair (0,1) is conditionally coupled (JGivenI) via A=3+s2 (>0) with
        // B!=0, C==0, so s1 should be inferred as -1 in the same round.
        let instance = HuboModel::spin(3)
            .add_linear(0, 5.0)
            .add_term(&[0, 1], 3.0)
            .add_term(&[0, 1, 2], 1.0)
            .build();
        let term_status: Vec<_> = instance
            .terms
            .iter()
            .map(|term| Some(PartiallyAssignedTerm::new(term)))
            .collect();
        let fixed = Fixes::new(instance.n_vars());
        let mut node = Node {
            fixed,
            lb: 0.0,
            offset: 0.0,
            term_status,
            term_by_free_vars: None,
            local_constraints: ConstraintHandler::new(),
            lb_warm_start: None,
        };

        let kernelizer = Kernelizer::new(KernelizationConfig {
            enable_binary_dominance: false,
            enable_spin_dominance: true,
            enable_binary_roof_duality: false,
            enable_spin_roof_duality: false,
            remove_zero_coeff_terms: true,
            max_rounds: 10,
            enable_coupling_reduction: true,
            enable_sflip_symmetry: false,
        });

        let report = kernelizer
            .kernelize(&Arc::new(instance), &mut node, None)
            .unwrap();

        assert!(report.dominance_fixed >= 1);
        assert!(report.conditional_coupling_implied_fixed >= 1);
    }
}
