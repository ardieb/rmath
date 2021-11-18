use autograd as ag;
use autograd::array_gen as gen;
use autograd::ndarray as nd;
use autograd::tensor_ops as math;

use crate::options::model::*;
use autograd::prelude::*;

pub struct BinomialPricingModel;

impl OptionPricingModel for BinomialPricingModel {
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
                    eval_one_call(s, k, vol, q, r, t)
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
                    eval_one_put(s, k, vol, q, r, t)
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
                        let pred = BinomialPricingModel::price(
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
                let pred = BinomialPricingModel::price(
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
                let pred = BinomialPricingModel::price(
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
                let pred = BinomialPricingModel::delta(
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
                let pred = BinomialPricingModel::price(
                    ty, s.as_ref(), k.as_ref(), vol_i.as_ref(), q.as_ref(), r, t,
                );
                pred
            })
            .collect::<Vec<_>>();
        let grad = math::finite_difference(m as usize, n as usize, h, &stencil_points[..]);
        grad
    }
}

fn eval_one_call<F: ag::Float>(s: F, k: F, vol: F, q: F, r: F, t: F) -> F {
    eval_one(s, k, vol, q, r, t, OptionType::Call)
}

fn eval_one_put<F: ag::Float>(s: F, k: F, vol: F, q: F, r: F, t: F) -> F {
    eval_one(s, k, vol, q, r, t, OptionType::Put)
}

fn eval_one<F: ag::Float>(s: F, k: F, vol: F, q: F, r: F, t: F, ty: OptionType) -> F {
    let dt: F = F::one() / F::from(365f64).unwrap();
    let u: F = (vol * dt.sqrt()).exp();
    let d: F = (-vol * dt.sqrt()).exp();
    let p: F = (((r - q) * dt).exp() - d) / (u - d);
    let n: usize = ag::num::NumCast::from(t / dt).unwrap();

    let mut dp: ag::NdArray<F> = gen::zeros(&[n + 1, n + 1]);

    for j in 0..n + 1 {
        let us = j as i32;
        let steps = n as i32;
        let stock_price = s * u.powi(2 * us - steps);
        let exercise_profit = match ty {
            OptionType::Call => (stock_price - k).max(F::zero()),
            OptionType::Put => (k - stock_price).max(F::zero()),
        };
        dp[[n, j]] = exercise_profit;
    }

    for i in (0..n).rev() {
        for j in (0..n).rev() {
            let us = j as i32;
            let steps = i as i32;
            let stock_price = s * u.powi(2 * us - steps);
            let exercise_profit = match ty {
                OptionType::Call => (stock_price - k).max(F::zero()),
                OptionType::Put => (k - stock_price).max(F::zero()),
            };
            let decay = (-r * dt).exp();
            let expected = p * dp[[i + 1, j + 1]] + (F::one() - p) * dp[[i + 1, j]];
            let binom = decay * expected;
            dp[[i, j]] = binom.max(exercise_profit);
        }
    }

    dp[[0, 0]]
}
