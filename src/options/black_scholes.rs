use autograd as ag;
use autograd::array_gen as gen;
use autograd::tensor_ops as math;

use autograd::prelude::*;
use crate::options::model::*;

pub struct BlackScholesPricingModel;

impl OptionPricingModel for BlackScholesPricingModel {
    fn price<'graph, A, F: ag::Float>(ty: OptionType, s: A, k: A, vol: A, q: A, r: F, t: F) -> ag::Tensor<'graph, F> 
    where
        A: AsRef<ag::Tensor<'graph, F>> + Copy {
        match ty {
            OptionType::Call => call(s, k, vol, q, r, t),
            OptionType::Put => put(s, k, vol, q, r, t),
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
        match ty {
            OptionType::Call => call_iv(p, s, k, q, r, t),
            OptionType::Put => put_iv(p, s, k, q, r, t),
        }
    }

    fn delta<'graph, A, F: ag::Float>(ty: OptionType, s: A, k: A, vol: A, q: A, r: F, t: F) -> ag::Tensor<'graph, F> 
    where
        A: AsRef<ag::Tensor<'graph, F>> + Copy
    {
        let stock_price = s.as_ref();
        let price = BlackScholesPricingModel::price(ty, s, k, vol, q, r, t);
        math::grad(&[price], &[stock_price])[0]
    }

    fn theta<'graph, A, F: ag::Float> (ty: OptionType, s: A, k: A, vol: A, q: A, r: F, t: F) -> ag::Tensor<'graph, F>
    where
        A: AsRef<ag::Tensor<'graph, F>> + Copy
    {
        let m: i32 = 1;
        let n: i32 = 2;
        // Hacky way to get a scalar tensor.
        let h = math::reduce_sum(math::flatten(s.as_ref()) * F::zero(), &[-1_i32], false) + F::from(0.05_f64).unwrap();

        let stencil_points = (-n / 2..n / 2 + 1)
            .map(|i| {
                let ti = t - (F::from(i).unwrap() * F::from(0.05_f64).unwrap());
                let pred = BlackScholesPricingModel::price(
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
        let h = math::reduce_sum(math::flatten(s.as_ref()) * F::zero(), &[-1_i32], false) + F::from(0.05_f64).unwrap();

        let stencil_points = (-n / 2..n / 2 + 1)
            .map(|i| {
                let ti = t - (F::from(i).unwrap() * F::from(0.05_f64).unwrap());
                let pred = BlackScholesPricingModel::delta(
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
        let volitility = s.as_ref();
        let price = BlackScholesPricingModel::price(ty, s, k, vol, q, r, t);
        math::grad(&[price], &[volitility])[0]
    }
}

fn call<'graph, A, F: ag::Float>(s: A, k: A, vol: A, q: A, r: F, t: F) -> ag::Tensor<'graph, F>
where
    A: AsRef<ag::Tensor<'graph, F>> + Copy,
{
    let s = s.as_ref();
    let k = k.as_ref();
    let vol = vol.as_ref();
    let q = q.as_ref();

    let half = F::from(0.5f64).unwrap();
    let one = F::one();
    let zero = F::zero();
    // d1 = ln(s/k) + (vol^2 / 2 + r) * t
    //      -----------------------------
    //             vol * sqrt(t)
    let d1 = (math::ln(s / k) + (((math::square(vol) * half) + r) - q) * t) / (vol * t.sqrt());
    // d2 = d1 - vol * sqrt(t)
    let d2 = d1 - (vol * t.sqrt());
    let nd1 = math::normal_cdf(&d1, zero, one);
    let nd2 = math::normal_cdf(&d2, zero, one);
    ((s * math::exp(math::neg(q * t))) * nd1) - ((k * (-t * r).exp()) * nd2)
}

fn put<'graph, A, F: ag::Float>(s: A, k: A, vol: A, q: A, r: F, t: F) -> ag::Tensor<'graph, F>
where
    A: AsRef<ag::Tensor<'graph, F>> + Copy,
{
    let s = s.as_ref();
    let k = k.as_ref();
    let vol = vol.as_ref();
    let q = q.as_ref();
    let half = F::from(0.5f64).unwrap();
    let one = F::one();
    let zero = F::zero();
    // d1 = ln(s/k) + (vol^2 / 2 + r - q) * t
    //      -----------------------------
    //             vol * sqrt(t)

    let d1 = (math::ln(s / k) + (((math::square(vol) * half) + r) - q) * t) / (vol * t.sqrt());
    // d2 = d1 - vol * sqrt(t)
    let d2 = d1 - (vol * t.sqrt());
    let nnegd1 = math::normal_cdf(&math::neg(d1), zero, one);
    let nnegd2 = math::normal_cdf(&math::neg(d2), zero, one);
    ((k * (-r * t).exp()) * nnegd2) - ((s * math::exp(math::neg(q * t))) * nnegd1)
}

fn call_iv<'graph, F: ag::Float>(
    c: ag::NdArrayView<F>,
    s: ag::NdArrayView<F>,
    k: ag::NdArrayView<F>,
    q: ag::NdArrayView<F>,
    r: F,
    t: F,
) -> ag::NdArray<F> {
    let mut env = ag::VariableEnvironment::new();
    let ret_id = env.name("vol").set(gen::ones(c.shape()));

    let adam = ag::optimizers::adam::Adam::default("AdamIV", env.default_namespace().current_var_ids(), &mut env);

    for _ in 0..1000 {
        env.run(|ctx| {
            let vol = ctx.variable("vol");
            let call_price = ctx.placeholder("c", &[-1]);
            let spot = ctx.placeholder("s", &[-1]);
            let strike = ctx.placeholder("k", &[-1]);
            let dividends = ctx.placeholder("q", &[-1]);
            let pred = call(&spot, &strike, &vol, &dividends, r, t);

            let losses = math::abs(call_price - pred);
            let grads = math::grad(&[losses], &[vol]);
            
            let mut feeder = ag::Feeder::new();
            feeder.push(call_price, c.view())
                  .push(spot, s.view())
                  .push(strike, k.view())
                  .push(dividends, q.view());

            adam.update(&[vol], &grads, ctx, feeder);
        });
    }

    env.get_array_by_id(ret_id)
        .unwrap()
        .clone()
        .into_inner()
}

fn put_iv<'graph, F: ag::Float>(
    p: ag::NdArrayView<F>,
    s: ag::NdArrayView<F>,
    k: ag::NdArrayView<F>,
    q: ag::NdArrayView<F>,
    r: F,
    t: F,
) -> ag::NdArray<F> {
    let mut env = ag::VariableEnvironment::new();
    let ret_id = env.name("vol").set(gen::ones(p.shape()));

    let adam = ag::optimizers::adam::Adam::default("AdamIV", env.default_namespace().current_var_ids(), &mut env);

    for _ in 0..1000 {
        env.run(|ctx| {
            let vol = ctx.variable("vol");
            let put_price = ctx.placeholder("p", &[-1]);
            let spot = ctx.placeholder("s", &[-1]);
            let strike = ctx.placeholder("k", &[-1]);
            let dividends = ctx.placeholder("q", &[-1]);
            let pred = call(&spot, &strike, &vol, &dividends, r, t);

            let losses = math::abs(put_price - pred);
            let grads = math::grad(&[losses], &[vol]);
            
            let mut feeder = ag::Feeder::new();
            feeder.push(put_price, p.view())
                  .push(spot, s.view())
                  .push(strike, k.view())
                  .push(dividends, q.view());

            adam.update(&[vol], &grads, ctx, feeder);
        });
    }

    env.get_array_by_id(ret_id)
        .unwrap()
        .clone()
        .into_inner()
}
