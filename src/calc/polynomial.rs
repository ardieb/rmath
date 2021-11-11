use autograd as ag;
use autograd::ndarray as nd;
use autograd::ndarray_ext as arr;

fn is_vector(shape: &[usize]) -> bool {
    shape.len() == 1
}

pub struct Polynomial;
struct PolynomialGrad {
    n: usize
}

impl<T: ag::Float> ag::op::Op<T> for Polynomial {
    fn compute(&self, ctx: &mut ag::op::ComputeContext<T>) {
        let z: &ag::NdArrayView<_> = &ctx.input(0);
        let coeffs: &ag::NdArrayView<_> = &ctx.input(1);

        if !is_vector(coeffs.shape()) {
            panic!(
                "Polynomial input is not a vector! Input polynomial {} is invalid.",
                coeffs
            );
        }

        let output = z.mapv(|x| {
            coeffs
                .slice_axis(nd::Axis(0), nd::Slice::from(..))
                .fold((0, T::zero()), |(n, s), c| (n + 1, s + *c * x.powi(n)))
                .1
        });

        ctx.append_output(output);
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<T>) {
        let g = ctx.graph();
        let z = ctx.input(0);
        let coeffs = ctx.input(1);

        let gy = ctx.output_grad();
        let gz = gy
            * ag::Tensor::builder()
                .set_inputs(&[ag::tensor::Input::new(&z), ag::tensor::Input::new(&coeffs)])
                .build(g, PolynomialGrad { n: 1 });

        ctx.append_input_grad(Some(gz));
        ctx.append_input_grad(None);
    }
}

impl<T: ag::Float> ag::op::Op<T> for PolynomialGrad {
    fn compute(&self, ctx: &mut ag::op::ComputeContext<T>) {
        let z: &ag::NdArrayView<_> = &ctx.input(0);
        let coeffs: &ag::NdArrayView<_> = &ctx.input(1);

        if !is_vector(coeffs.shape()) {
            panic!(
                "Polynomial input is not a vector! Input polynomial {} is invalid.",
                coeffs
            );
        }

        let gz = z.mapv(|x| {
            if self.n < coeffs.len() {
                coeffs
                    .slice_axis(nd::Axis(0), nd::Slice::from(self.n..))
                    .fold((0, T::zero()), |(n, s), c| (n + 1, s + *c * x.powi(n)))
                    .1
            } else {
                T::zero()
            }
        });

        ctx.append_output(gz);
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<T>) {
        let g = ctx.graph();
        let z = ctx.input(0);
        let coeffs = ctx.input(1);

        let gy = ctx.output_grad();
        let gz = gy
            * ag::Tensor::builder()
                .set_inputs(&[ag::tensor::Input::new(&z), ag::tensor::Input::new(&coeffs)])
                .build(g, PolynomialGrad { n: self.n + 1 });
 
        ctx.append_input_grad(Some(gz));
        ctx.append_input_grad(None);
    }
}

pub fn poly<'graph, F: ag::Float>(
    g: &'graph ag::Graph<F>,
    x: &ag::Tensor<'graph, F>,
    y: &ag::Tensor<'graph, F>,
) -> ag::Tensor<'graph, F> {
    ag::Tensor::builder()
        .set_inputs(&[ag::tensor::Input::new(x), ag::tensor::Input::new(y)])
        .build(g, Polynomial)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ag::tensor::Variable;

    #[test]
    fn test_eval() {
        ag::with(|g: &mut ag::Graph<_>| {
            let poly5 = g.variable(nd::array![1., 0., 0., 0., 0., 2.]);
            let z = g.variable(nd::array![2.]);
            let y = poly(g, &z, &poly5);
            let output = y.eval(&[]).expect("Failed to evaluate the polynomial");
            assert_eq!(output[0], 2. * 2f64.powi(5) + 1.);
        })
    }

    #[test]
    fn test_grad() {
        ag::with(|g: &mut ag::Graph<_>| {
            let poly5 = g.variable(nd::array![1., 0., 2.]); // z^2 + 1
            let z = g.variable(nd::array![2.]);
            let y = poly(g, &z, &poly5);

            let dydz = g.grad(&[y], &[z])[0]; // 2*z
            let d2ydz2 = g.grad(&[dydz], &[z])[0]; // 2
            let d3ydz3 = g.grad(&[d2ydz2], &[z])[0]; // 0
            let d4ydz4 = g.grad(&[d3ydz3], &[z])[0]; // 0 ...

            assert_eq!(dydz.eval(&[]).unwrap()[0], 4.);
            assert_eq!(d2ydz2.eval(&[]).unwrap()[0], 2.);
            assert_eq!(d3ydz3.eval(&[]).unwrap()[0], 0.);
            assert_eq!(d4ydz4.eval(&[]).unwrap()[0], 0.);
        })
    }
}
