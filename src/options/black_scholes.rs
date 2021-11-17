use autograd as ag;
use autograd::array_gen as gen;
use autograd::tensor_ops as math;

use crate::stats;
use autograd::prelude::*;

/// Calculate the price of a call option based on the
/// Black Scholes Merton model for options pricing.
///
/// This function can price multiple options at once by inputing
/// a multidimensional set of inputs. All multi dimensional inputs
/// must have the same shape.
///
/// :param: `s` The underlying stocks' prices per share.
/// :param: `k` The options' strike prices per share.
/// :param: `vol` The volatility of the stocks in decimal.
/// :param: `r` The risk free interest rate as decimal.
/// :param: `q` The continously compounding divdend yield in decimal.
/// :param: `t` The time until option maturity as decimal of a year.
///
/// :return: `prices` The price of the options.
pub fn call<'graph, A, F: ag::Float>(s: A, k: A, vol: A, q: A, r: F, t: F) -> ag::Tensor<'graph, F>
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
    let nd1 = stats::normal::cdf(&d1, zero, one);
    let nd2 = stats::normal::cdf(&d2, zero, one);
    ((s * math::exp(math::neg(q * t))) * nd1) - ((k * (-t * r).exp()) * nd2)
}

/// Calculate the price of a put option based on the
/// Black Scholes Merton model for options pricing.
///
/// This function can price multiple options at once by inputing
/// a multidimensional set of inputs. All multi dimensional inputs
/// must have the same shape.
///
/// :param: `s` The underlying stocks' prices .
/// :param: `k` The options' strike prices.
/// :param: `vol` The volatility of the stocks.
/// :param: `r` The risk free interest rate.
/// :param: `q` The continously compounding divdend yield in decimal.
/// :param: `t` The time until option maturity as decimal of a year.
///
/// :return: `prices` The price of the options.
pub fn put<'graph, A, F: ag::Float>(s: A, k: A, vol: A, q: A, r: F, t: F) -> ag::Tensor<'graph, F>
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
    let nnegd1 = stats::normal::cdf(&math::neg(d1), zero, one);
    let nnegd2 = stats::normal::cdf(&math::neg(d2), zero, one);
    ((k * (-r * t).exp()) * nnegd2) - ((s * math::exp(math::neg(q * t))) * nnegd1)
}

/// Determine the implied volatility for a call option using
/// the Black Scholes option pricing method.
///
/// This function can price multiple options at once by inputing
/// a multidimensional set of inputs. All multi dimensional inputs
/// must have the same shape.
///
/// :param: `c` The call options' prices.
/// :param: `s` The underlying stocks' prices.
/// :param: `k` The options' strike prices.
/// :param: `r` The risk free interest rate.
/// :param: `q` The continously compounding divdend yield in decimal.
/// :param: `t` The time until option maturity.
///
/// :return: `prices` The price of the options.
pub fn call_iv<'graph, F: ag::Float>(
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

/// Determine the implied volatility for a put option using
/// the Black Scholes option pricing method.
///
/// This function can price multiple options at once by inputing
/// a multidimensional set of inputs. All multi dimensional inputs
/// must have the same shape.
///
/// :param: `p` The put options' prices.
/// :param: `s` The underlying stocks' prices.
/// :param: `k` The options' strike prices.
/// :param: `r` The risk free interest rate.
/// :param: `q` The continously compounding divdend yield in decimal.
/// :param: `t` The time until option maturity.
///
/// :return: `prices` The price of the options.
pub fn put_iv<'graph, F: ag::Float>(
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
