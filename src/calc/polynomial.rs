use autograd as ag;
use autograd::ndarray as nd;
use autograd::ndarray_ext as arr;


fn is_vector(shape: &[usize]) -> bool {
    shape.len() == 1
}

pub struct Polynomial;
pub struct PolynomialGrad;


impl<T: ag::Float> ag::op::Op<T> for Polynomial {
    fn compute(&self, ctx: &mut ag::op::ComputeContext<T>) {
        let z: &ag::NdArrayView<_> = &ctx.input(0);
        let coeffs: &ag::NdArrayView<_> = &ctx.input(1);
        
        if !is_vector(coeffs.shape()) {
            panic!("Polynomial input is not a vector! Input polynomial {} is invalid.", coeffs);
        }

        let output = z.mapv(move |x| {
            coeffs.fold((0, T::zero()), |(n, s), c| (n+1, s + *c * x.powi(n))).1
        });

        ctx.append_output(output);
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<T>) {
        let g = ctx.graph();
        let z = ctx.input(0);
        let coeffs = ctx.input(1);

        let gy = ctx.output_grad();
        let gz = gy * poly_grad(g, &z, &coeffs);
        ctx.append_input_grad(Some(gz));
        ctx.append_input_grad(None);
    }
}


impl<T: ag::Float> ag::op::Op<T> for PolynomialGrad {
    fn compute(&self, ctx: &mut ag::op::ComputeContext<T>) {
        let z: &ag::NdArrayView<_> = &ctx.input(0);
        let coeffs: &ag::NdArrayView<_> = &ctx.input(1);
        
        if !is_vector(coeffs.shape()) {
            panic!("Polynomial input is not a vector! Input polynomial {} is invalid.", coeffs);
        }

        let output = if coeffs.shape()[0] == 1 {
            arr::NdArray::zeros(z.shape())
        } else {
            let dcoeffs = coeffs.slice_axis(nd::Axis(0), nd::Slice::from(1..));
            z.map(move |x| {
                dcoeffs.fold((0, T::zero()), |(n, s), c| (n+1, s + *c * x.powi(n))).1
            })
        };
        ctx.append_output(output);
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<T>) {
        let g = ctx.graph();
        let z = ctx.input(0);
        let coeffs = ctx.input(1);

        let gy = ctx.output_grad();
        let gz = gy * poly_grad(g, &z, &coeffs);
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


pub fn poly_grad<'graph, F: ag::Float>(
    g: &'graph ag::Graph<F>,
    x: &ag::Tensor<'graph, F>,
    y: &ag::Tensor<'graph, F>
) -> ag::Tensor<'graph, F> {
    ag::Tensor::builder()
        .set_inputs(&[ag::tensor::Input::new(x), ag::tensor::Input::new(y)])
        .build(g, PolynomialGrad)
}


#[cfg(test)]
mod tests {
    use super::*;
    use ag::tensor::Variable;

    #[test]
    fn test_forward_eval() {
        ag::with(|g: &mut ag::Graph<_>| {
            let poly5 = g.variable(nd::array![1., 0., 0., 0., 0., 2.]);
            let z = g.variable(nd::array![2.]);
            let y = poly(g, &z, &poly5);
            let output = y.eval(&[])
                .expect("Failed to evaluate the polynomial");
            assert_eq!(output[0], 2. * 2f64.powi(5) + 1.);
            let dydz = &g.grad(&[y], &[z])[0];
            let output = dydz.eval(&[])
                .expect("Failed to differentiate the polynomial");
            assert_eq!(output[0], 2. * 2f64.powi(4));
        })
    }
}
