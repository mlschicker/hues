use crate::coeff::Coeff;
use crate::lasserre::{ExactLasserreConfig, lasserre_exact_lower_bound};
use crate::solver::bnb::Node;
use crate::{domain::VarDomain, instance::HuboInstance};

#[derive(Debug, Clone)]
pub struct ExactLasserre(pub ExactLasserreConfig);

pub(crate) fn compute<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &Node<C>,
    cfg: &ExactLasserreConfig,
) -> C {
    let ov = node.to_option_vec(instance);
    C::from_f64_lb(lasserre_exact_lower_bound(instance, &ov, cfg))
}
