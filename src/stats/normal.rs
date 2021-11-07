use autograd as ag;
use crate::calc;


pub fn cdf<'graph, F: ag::Float>(
    g: &'graph ag::Graph<F>,
    x: &ag::Tensor<'graph, F>,
    mean: F,
    std: F,
) -> ag::Tensor<'graph, F> {
    let half = F::from(0.5f64).unwrap();
    let sqrt2 = F::from(std::f64::consts::SQRT_2).unwrap();
    let z = g.neg(x - mean) / (std * sqrt2);
    calc::erf::erfc(g, &z) * half
}


pub fn pdf<'graph, F: ag::Float>(
    g: &'graph ag::Graph<F>,
    x: &ag::Tensor<'graph, F>,
    mean: F,
    std: F,
) -> ag::Tensor<'graph, F> {
    let half = F::from(0.5f64).unwrap();
    let sqrt2pi = F::from((2.*std::f64::consts::PI).sqrt()).unwrap();
    let d = (x - mean) / std;
    g.exp(d * d * -half) / (sqrt2pi * std)
}
