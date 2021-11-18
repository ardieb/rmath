use autograd as ag;
use autograd::array_gen as gen;
use autograd::ndarray as nd;
use autograd::tensor_ops as math;

use crate::options::model::*;
use autograd::prelude::*;
use autograd::rayon::prelude::*;

use autograd::rand::{thread_rng, distributions::Distribution};
use autograd::statrs::distribution::{Continuous, Normal};

pub struct MonteCarloPricingModel;

impl OptionPricingModel for MonteCarloPricingModel {
    fn price<'graph, A, F: ag::Float>(
        ty: OptionType,
        s: A,
        k: A,
        vol: A,
        q: A,
        r: F,
        t: F,
    ) -> ag::Tensor<'graph, F>
    where
        A: AsRef<ag::Tensor<'graph, F>> + Copy,
    {
        let dim: i32 = -1;
        let s = s.as_ref().expand_dims(&[dim]);
        let k = k.as_ref().expand_dims(&[dim]);
        let vol = vol.as_ref().expand_dims(&[dim]);
        let q = q.as_ref().expand_dims(&[dim]);
        let rs = (s / s) * r;
        let ts = (s / s) * t;

        let packed = math::concat(&[s, k, vol, q, rs, ts], 0);
        match ty {
            OptionType::Call => packed.map(|packed| {
                packed.map_axis(nd::Axis(0), |col| {
                    let s = col[0];
                    let k = col[1];
                    let vol = col[2];
                    let q = col[3];
                    let r = col[4];
                    let t = col[5];
                    eval_one_call(s, k, vol, q, r, t, 500)
                })
            }),
            OptionType::Put => packed.map(|packed| {
                packed.map_axis(nd::Axis(0), |col| {
                    let s = col[0];
                    let k = col[1];
                    let vol = col[2];
                    let q = col[3];
                    let r = col[4];
                    let t = col[5];
                    eval_one_put(s, k, vol, q, r, t, 500)
                })
            }),
        }
    }

    fn implied_volatility<F: ag::Float>(
        ty: OptionType,
        p: ag::NdArrayView<F>,
        s: ag::NdArrayView<F>,
        k: ag::NdArrayView<F>,
        q: ag::NdArrayView<F>,
        r: F,
        t: F,
    ) -> ag::NdArray<F> {
        let mut env = ag::VariableEnvironment::new();
        let ret_id = env.name("vol").set(gen::ones(p.shape()));
        let adam = ag::optimizers::adam::Adam::default(
            "AdamIV",
            env.default_namespace().current_var_ids(),
            &mut env,
        );
        for _ in 0..1000 {
            env.run(|ctx| {
                let vol = ctx.variable("vol");
                let price = ctx.placeholder("p", &[-1]);
                let spot = ctx.placeholder("s", &[-1]);
                let strike = ctx.placeholder("k", &[-1]);
                let dividends = ctx.placeholder("q", &[-1]);

                let h = F::from(0.05_f64).unwrap();

                let m: i32 = 1;
                let n: i32 = 2;

                let losses = (-n / 2..n / 2 + 1)
                    .map(|i| {
                        let voli = vol + (h * F::from(i).unwrap());
                        let pred = MonteCarloPricingModel::price(
                            ty, &spot, &strike, &voli, &dividends, r, t,
                        );
                        math::abs(price - pred)
                    })
                    .collect::<Vec<_>>();
                let grad = math::finite_difference(m as usize, n as usize, h, &losses[..]);

                let mut feeder = ag::Feeder::new();
                feeder
                    .push(price, p.view())
                    .push(spot, s.view())
                    .push(strike, k.view())
                    .push(dividends, q.view());

                adam.update(&[vol], &[grad], ctx, feeder);
            });
        }
        env.get_array_by_id(ret_id).unwrap().clone().into_inner()
    }

    fn delta<'graph, A, F: ag::Float>(ty: OptionType, s: A, k: A, vol: A, q: A, r: F, t: F) -> ag::Tensor<'graph, F> 
    where
        A: AsRef<ag::Tensor<'graph, F>> + Copy 
    {
        let stock_price = s.as_ref();

        let m: i32 = 1;
        let n: i32 = 2;
        // Hacky way to get a scalar tensor.
        let h = F::from(0.05_f64).unwrap();

        let stencil_points = (-n / 2..n / 2 + 1)
            .map(|i| {
                let stock_price_i = stock_price + (h * F::from(i).unwrap());
                let pred = MonteCarloPricingModel::price(
                    ty, stock_price_i.as_ref(), k.as_ref(), vol.as_ref(), q.as_ref(), r, t,
                );
                pred
            })
            .collect::<Vec<_>>();
        let grad = math::finite_difference(m as usize, n as usize, h, &stencil_points[..]);
        grad
    }

    fn theta<'graph, A, F: ag::Float>(ty: OptionType, s: A, k: A, vol: A, q: A, r: F, t: F) -> ag::Tensor<'graph, F>
    where
        A: AsRef<ag::Tensor<'graph, F>> + Copy
    {
        let m: i32 = 1;
        let n: i32 = 2;
        // Hacky way to get a scalar tensor.
        let h =  F::from(0.05_f64).unwrap();

        let stencil_points = (-n / 2..n / 2 + 1)
            .map(|i| {
                let ti = t - (F::from(i).unwrap() * F::from(0.05_f64).unwrap());
                let pred = MonteCarloPricingModel::price(
                    ty, s.as_ref(), k.as_ref(), vol.as_ref(), q.as_ref(), r, ti,
                );
                pred
            })
            .collect::<Vec<_>>();
        let grad = math::finite_difference(m as usize, n as usize, h, &stencil_points[..]);
        grad
    }

    fn gamma<'graph, A, F: ag::Float>(ty: OptionType, s: A, k: A, vol: A, q: A, r: F, t: F) -> ag::Tensor<'graph, F>
    where
        A: AsRef<ag::Tensor<'graph, F>> + Copy
    {
        let m: i32 = 1;
        let n: i32 = 2;
        // Hacky way to get a scalar tensor.
        let h = F::from(0.05_f64).unwrap();

        let stencil_points = (-n / 2..n / 2 + 1)
            .map(|i| {
                let ti = t - (F::from(i).unwrap() * F::from(0.05_f64).unwrap());
                let pred = MonteCarloPricingModel::delta(
                    ty, s.as_ref(), k.as_ref(), vol.as_ref(), q.as_ref(), r, ti,
                );
                pred
            })
            .collect::<Vec<_>>();
        let grad = math::finite_difference(m as usize, n as usize, h, &stencil_points[..]);
        grad
    }

    fn vega<'graph, A, F: ag::Float>(ty: OptionType, s: A, k: A, vol: A, q: A, r: F, t: F) -> ag::Tensor<'graph, F>
    where
        A: AsRef<ag::Tensor<'graph, F>> + Copy
    {
        let m: i32 = 1;
        let n: i32 = 2;
        // Hacky way to get a scalar tensor.
        let h =  F::from(0.05_f64).unwrap();

        let stencil_points = (-n / 2..n / 2 + 1)
            .map(|i| {
                let vol_i = vol.as_ref() + (h * F::from(i).unwrap());
                let pred = MonteCarloPricingModel::price(
                    ty, s.as_ref(), k.as_ref(), vol_i.as_ref(), q.as_ref(), r, t,
                );
                pred
            })
            .collect::<Vec<_>>();
        let grad = math::finite_difference(m as usize, n as usize, h, &stencil_points[..]);
        grad
    }
}

fn eval_one_call<F: ag::Float>(s: F, k: F, vol: F, q: F, r: F, t: F, paths: usize) -> F {
    eval_one(s, k, vol, q, r, t, paths, OptionType::Call)
}

fn eval_one_put<F: ag::Float>(s: F, k: F, vol: F, q: F, r: F, t: F, paths: usize) -> F {
    eval_one(s, k, vol, q, r, t, paths, OptionType::Put)
}

fn eval_one<F: ag::Float>(s: F, k: F, vol: F, q: F, r: F, t: F, paths: usize, ty: OptionType) -> F {
    let dt: F = F::one() / F::from(365f64).unwrap();
    let two = F::from(2_f64).unwrap();

    (0..paths)
        .into_par_iter()
        .map(|_| {
            let mut st = s;
            let mut ts = F::zero();
            let mut rng = thread_rng();
            let normal = Normal::new(0., 1.).unwrap();
            while ts < t {
                let x = F::from(normal.sample(&mut rng)).unwrap();
                let epsilon = F::one() / (F::one() + (-x).exp());
                st = st * ((r - (vol.powi(2) / two)) * dt + (vol * epsilon * dt.sqrt()));
                ts += dt;
            }
            match ty {
                OptionType::Call => (-r * t).exp() * (st - k).max(F::zero()),
                OptionType::Put => (-r * t).exp() * (k - st).max(F::zero()),
            }
        })
        .reduce(|| F::zero(), |a, b| a + b)
        / F::from(paths).unwrap()
}
