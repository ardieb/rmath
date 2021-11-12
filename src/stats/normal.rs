use autograd as ag;
use autograd::tensor_ops as math;


pub fn cdf<'graph, A, F: ag::Float>(
    x: A,
    mean: F,
    std: F,
) -> ag::Tensor<'graph, F> 
where
    A: AsRef<ag::Tensor<'graph, F>> + Copy
{
    let x = x.as_ref();
    let half = F::from(0.5f64).unwrap();
    let sqrt2 = F::from(std::f64::consts::SQRT_2).unwrap();
    let z = math::neg(x - mean) / (std * sqrt2);
    math::erfc(&z) * half
}

pub fn pdf<'graph, A, F: ag::Float>(
    x: A,
    mean: F,
    std: F,
) -> ag::Tensor<'graph, F> 
where
    A: AsRef<ag::Tensor<'graph, F>> + Copy
{
    let x = x.as_ref();
    let half = F::from(0.5f64).unwrap();
    let sqrt2pi = F::from((2. * std::f64::consts::PI).sqrt()).unwrap();
    let d = (x - mean) / std;
    math::exp(d * d * - half) / (sqrt2pi * std)
}

#[cfg(test)]
mod tests {
    use super::*;
    use autograd::prelude::*;
    use autograd::ndarray as nd;

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
        let mut env = ag::VariableEnvironment::new();
        env.name("z").set(zvals);

        env.run(|ctx| {

            let z = ctx.variable("z");
            let mean: f64 = 0.;
            let std: f64 = 1.;
            let cumulative_dist = cdf(&z, mean, std);
            let results = cumulative_dist
                .eval(ctx)
                .expect("Could not evaluate the cdf!");
            assert!((results - ans).map(|diff| diff.abs()).sum() <= 1e-5);
        });
    }

    #[test]
    pub fn test_deriv_cdf_equals_pdf() {
        let zvals = nd::array![0., -1., 1., -0.5, 0.5, -5., 5.];
        let mut env = ag::VariableEnvironment::new();
        env.name("z").set(zvals);

        env.run(|ctx| {
            let z = ctx.variable("z");
            let mean: f64 = 0.;
            let std: f64 = 1.;
            let cumulative_dist = cdf(&z, mean, std);
            let grad = math::grad(&[cumulative_dist], &[z])[0];
            let results = grad
                .eval(ctx)
                .expect("Could not evaluate the cdf grad.");
            let ans = pdf(&z, mean, std)
                .eval(ctx)
                .expect("Could not evaluuate the pdf.");
            assert!((results - ans).map(|diff| diff.abs()).sum() <= 1e-5);
        });
    }
}
