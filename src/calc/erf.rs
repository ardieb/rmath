use autograd as ag;
use num;
use statrs;

pub struct Erf;
pub struct ErfInv;
pub struct ErfC;
pub struct ErfCInv;

impl<T: ag::Float> ag::op::Op<T> for Erf {
    fn compute(&self, ctx: &mut ag::op::ComputeContext<T>) {
        let z: &ag::NdArrayView<_> = &ctx.input(0);
        let output = z.mapv(move |x| {
            let x64: f64 = num::NumCast::from(x).unwrap();
            let c64: f64 = statrs::function::erf::erf(x64);
            T::from(c64).unwrap()
        });
        ctx.append_output(output);
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<T>) {
        let g = ctx.graph();
        let z = ctx.input(0);
        let gy = ctx.output_grad();

        let two = T::from::<f64>(2.).unwrap();
        let sqrt_pi = T::from::<f64>(std::f64::consts::PI.sqrt()).unwrap();

        let gz = gy * two * g.exp(g.neg(g.square(z))) / sqrt_pi;
        ctx.append_input_grad(Some(gz));
    }
}

impl<T: ag::Float> ag::op::Op<T> for ErfInv {
    fn compute(&self, ctx: &mut ag::op::ComputeContext<T>) {
        let z: &ag::NdArrayView<_> = &ctx.input(0);
        let output = z.mapv(move |x| {
            let x64: f64 = num::NumCast::from(x).unwrap();
            let c64: f64 = statrs::function::erf::erf_inv(x64);
            T::from(c64).unwrap()
        });
        ctx.append_output(output);
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<T>) {
        let half = T::from::<f64>(0.5).unwrap();
        let sqrt_pi = T::from::<f64>(std::f64::consts::PI.sqrt()).unwrap();

        let g = ctx.graph();
        let z = ctx.output();
        let gy = ctx.output_grad();
        let gz = gy * half * sqrt_pi * g.exp(g.neg(g.square(z)));

        ctx.append_input_grad(Some(gz));
    }
}

impl<T: ag::Float> ag::op::Op<T> for ErfC {
    fn compute(&self, ctx: &mut ag::op::ComputeContext<T>) {
        let z: &ag::NdArrayView<_> = &ctx.input(0);
        let output = z.mapv(move |x| {
            let x64: f64 = num::NumCast::from(x).unwrap();
            let c64: f64 = statrs::function::erf::erfc(x64);
            T::from(c64).unwrap()
        });
        ctx.append_output(output);
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<T>) {
        let g = ctx.graph();
        let z = ctx.input(0);
        let gy = ctx.output_grad();

        let two = T::from::<f64>(2.).unwrap();
        let sqrt_pi = T::from::<f64>(std::f64::consts::PI.sqrt()).unwrap();

        let gz = g.neg(gy * two * g.exp(g.neg(g.square(z))) / sqrt_pi);
        ctx.append_input_grad(Some(gz));
    }
}

impl<T: ag::Float> ag::op::Op<T> for ErfCInv {
    fn compute(&self, ctx: &mut ag::op::ComputeContext<T>) {
        let z: &ag::NdArrayView<_> = &ctx.input(0);
        let output = z.mapv(move |x| {
            let x64: f64 = num::NumCast::from(x).unwrap();
            let c64: f64 = statrs::function::erf::erfc_inv(x64);
            T::from(c64).unwrap()
        });
        ctx.append_output(output);
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<T>) {
        let half = T::from::<f64>(0.5).unwrap();
        let sqrt_pi = T::from::<f64>(std::f64::consts::PI.sqrt()).unwrap();

        let g = ctx.graph();
        let z = ctx.output();
        let gy = ctx.output_grad();
        let gz = g.neg(gy * half * sqrt_pi * g.exp(g.neg(g.square(z))));

        ctx.append_input_grad(Some(gz));
    }
}

pub fn erf<'graph, F: ag::Float>(
    g: &'graph ag::Graph<F>,
    x: &ag::Tensor<'graph, F>,
) -> ag::Tensor<'graph, F> {
    ag::Tensor::builder()
        .set_inputs(&[ag::tensor::Input::new(x)])
        .build(g, Erf)
}

pub fn erfc<'graph, F: ag::Float>(
    g: &'graph ag::Graph<F>,
    x: &ag::Tensor<'graph, F>,
) -> ag::Tensor<'graph, F> {
    ag::Tensor::builder()
        .set_inputs(&[ag::tensor::Input::new(x)])
        .build(g, ErfC)
}

pub fn erfinv<'graph, F: ag::Float>(
    g: &'graph ag::Graph<F>,
    x: &ag::Tensor<'graph, F>,
) -> ag::Tensor<'graph, F> {
    ag::Tensor::builder()
        .set_inputs(&[ag::tensor::Input::new(x)])
        .build(g, ErfInv)
}

pub fn erfcinv<'graph, F: ag::Float>(
    g: &'graph ag::Graph<F>,
    x: &ag::Tensor<'graph, F>,
) -> ag::Tensor<'graph, F> {
    ag::Tensor::builder()
        .set_inputs(&[ag::tensor::Input::new(x)])
        .build(g, ErfCInv)
}
