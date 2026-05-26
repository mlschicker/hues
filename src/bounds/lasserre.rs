use crate::chordal_sdp::chordal_lasserre_lower_bound;
use crate::coeff::Coeff;
use crate::lasserre::{LasserreConfig, lasserre_lower_bound};
use crate::solver::bnb::Node;
use crate::{domain::VarDomain, instance::HuboInstance};

#[derive(Debug, Clone)]
pub struct Lasserre(pub LasserreConfig);

#[derive(Debug, Clone)]
pub struct ChordalSdp(pub LasserreConfig);

pub(crate) fn compute<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &Node<C>,
    cfg: &LasserreConfig,
) -> C {
    let ov = node.to_option_vec(instance);
    C::from_f64_lb(lasserre_lower_bound(instance, &ov, cfg))
}

pub(crate) fn compute_chordal<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &Node<C>,
    cfg: &LasserreConfig,
) -> C {
    let ov = node.to_option_vec(instance);
    C::from_f64_lb(chordal_lasserre_lower_bound(instance, &ov, cfg))
}
