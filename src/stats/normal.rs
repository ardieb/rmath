use autograd as ag;
use autograd::ndarray as nd;

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
    let sqrt2pi = F::from((2. * std::f64::consts::PI).sqrt()).unwrap();
    let d = (x - mean) / std;
    g.exp(d * d * -half) / (sqrt2pi * std)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_cdf() {
        let zvals = nd::array![0., -1., 1., -0.5, 0.5, -5., 5.];
        let ans = nd::array![
            0.5,
            0.15865525393145707,
            0.8413447460685429,
            0.3085375387259869,
            0.6914624612740131,
            2.8665157187919333e-07,
            0.9999997133484281
        ];
        ag::with(|g| {
            let z = g.placeholder(&[-1]);
            let mean = 0.;
            let std = 1.;
            let results = cdf(g, &z, mean, std)
                .eval(&[z.given(zvals.view())])
                .expect("Could not evaluate the cdf.");
            assert!(results.all_close(&ans, 1e-5));
        });
    }

    #[test]
    pub fn test_deriv_cdf_equals_pdf() {
        let zvals = nd::array![0., -1., 1., -0.5, 0.5, -5., 5.];
        ag::with(|g| {
            let z = g.placeholder(&[-1]);
            let mean = 0.;
            let std = 1.;
            let grad = g.grad(&[cdf(g, &z, mean, std)], &[z])[0];
            let cdf_results = grad
                .eval(&[z.given(zvals.view())])
                .expect("Could not evaluate the cdf grad.");
            let pdf_results = pdf(g, &z, mean, std)
                .eval(&[z.given(zvals.view())])
                .expect("Could not evaluuate the pdf.");
            assert!(cdf_results.all_close(&pdf_results, 1e-5));
        });
    }
}
