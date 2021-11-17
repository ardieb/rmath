use autograd as ag;
use autograd::array_gen as gen;
use autograd::ndarray as nd;
use autograd::tensor_ops as math;

use autograd::prelude::*;

/// Calculate the price of a call option based on the
/// Binomial model for options pricing.
///
/// This function can price multiple options at once by inputing
/// a multidimensional set of inputs. All multi dimensional inputs
/// must have the same shape.
///
/// * `s`: The underlying stocks' prices per share.
/// * `k`: The options' strike prices per share.
/// * `vol`: The volatility of the stocks in decimal.
/// * `r`: The risk free interest rate as decimal.
/// * `q`: The divided of the stock per year as decimal.
/// * `t`: The time until option maturity as decimal of a year.
///
/// * `prices`: The price of the options.
pub fn call<'graph, A, F: ag::Float>(s: A, k: A, vol: A, q: A, r: F, t: F) -> ag::Tensor<'graph, F>
where
    A: AsRef<ag::Tensor<'graph, F>>,
{
    let dim: i32 = -1;
    let s = s.as_ref().expand_dims(&[dim]);
    let k = k.as_ref().expand_dims(&[dim]);
    let vol = vol.as_ref().expand_dims(&[dim]);
    let q = q.as_ref().expand_dims(&[dim]);
    let rs = (s / s) * r;
    let ts = (s / s) * t;

    let packed = math::concat(&[s, k, vol, q, rs, ts], 0);
    packed.map(|packed| {
        packed.map_axis(nd::Axis(0), |col| {
            let s = col[0];
            let k = col[1];
            let vol = col[2];
            let q = col[3];
            let r = col[4];
            let t = col[5];
            eval_one_call(s, k, vol, q, r, t)
        })
    })
}

/// Calculate the price of a put option based on the
/// Binomial model for options pricing.
///
/// This function can price multiple options at once by inputing
/// a multidimensional set of inputs. All multi dimensional inputs
/// must have the same shape.
///
/// * `s`: The underlying stocks' prices per share.
/// * `k`: The options' strike prices per share.
/// * `vol`: The volatility of the stocks in decimal.
/// * `r`: The risk free interest rate as decimal.
/// * `q`: The divided of the stock per year as decimal.
/// * `t`: The time until option maturity as decimal of a year.
///
/// * `prices`: The price of the options.
pub fn put<'graph, A, F: ag::Float>(s: A, k: A, vol: A, q: A, r: F, t: F) -> ag::Tensor<'graph, F>
where
    A: AsRef<ag::Tensor<'graph, F>>,
{
    let dim: i32 = -1;
    let s = s.as_ref().expand_dims(&[dim]);
    let k = k.as_ref().expand_dims(&[dim]);
    let vol = vol.as_ref().expand_dims(&[dim]);
    let q = q.as_ref().expand_dims(&[dim]);
    let rs = (s / s) * r;
    let ts = (s / s) * t;

    let packed = math::concat(&[s, k, vol, q, rs, ts], 0);
    packed.map(|packed| {
        packed.map_axis(nd::Axis(0), |col| {
            let s = col[0];
            let k = col[1];
            let vol = col[2];
            let q = col[3];
            let r = col[4];
            let t = col[5];
            eval_one_put(s, k, vol, q, r, t)
        })
    })
}

/// Calculate the implied volatility based on the
/// Binomial model for options pricing.
///
/// This function can price multiple options at once by inputing
/// a multidimensional set of inputs. All multi dimensional inputs
/// must have the same shape.
///
/// * `c`: The price of the call options.
/// * `s`: The underlying stocks' prices per share.
/// * `k`: The options' strike prices per share.
/// * `r`: The risk free interest rate as decimal.
/// * `q`: The divided of the stock per year as decimal.
/// * `t`: The time until option maturity as decimal of a year.
///
/// * `prices`: The price of the options.
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

    let adam = ag::optimizers::adam::Adam::default(
        "AdamIV",
        env.default_namespace().current_var_ids(),
        &mut env,
    );
    for _ in 0..1000 {
        env.run(|ctx| {
            let vol = ctx.variable("vol");
            let call_price = ctx.placeholder("c", &[-1]);
            let spot = ctx.placeholder("s", &[-1]);
            let strike = ctx.placeholder("k", &[-1]);
            let dividends = ctx.placeholder("q", &[-1]);

            let h = math::ones(&[1i32], ctx) * F::from(0.05f64).unwrap();

            let m: i32 = 1;
            let n: i32 = 2;

            let losses = (-n/2..n/2+1)
                .map(|i| {
                    let voli = vol + (h * F::from(i).unwrap());
                    let pred = call(&spot, &strike, &voli, &dividends, r, t);
                    math::abs(call_price - pred)
                })
                .collect::<Vec<_>>();
            let grad = math::finite_difference(m as usize, n as usize, h, &losses[..]);

            let mut feeder = ag::Feeder::new();
            feeder
                .push(call_price, c.view())
                .push(spot, s.view())
                .push(strike, k.view())
                .push(dividends, q.view());

            adam.update(&[vol], &[grad], ctx, feeder);
        });
    }

    env.get_array_by_id(ret_id).unwrap().clone().into_inner()
}

/// Calculate the implied volatility based on the
/// Binomial model for options pricing.
///
/// This function can price multiple options at once by inputing
/// a multidimensional set of inputs. All multi dimensional inputs
/// must have the same shape.
///
/// * `p`: The price of the put options.
/// * `s`: The underlying stocks' prices per share.
/// * `k`: The options' strike prices per share.
/// * `r`: The risk free interest rate as decimal.
/// * `q`: The divided of the stock per year as decimal.
/// * `t`: The time until option maturity as decimal of a year.
///
/// * `prices`: The price of the options.
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

    let adam = ag::optimizers::adam::Adam::default(
        "AdamIV",
        env.default_namespace().current_var_ids(),
        &mut env,
    );
    for _ in 0..1000 {
        env.run(|ctx| {
            let vol = ctx.variable("vol");
            let put_price = ctx.placeholder("p", &[-1]);
            let spot = ctx.placeholder("s", &[-1]);
            let strike = ctx.placeholder("k", &[-1]);
            let dividends = ctx.placeholder("q", &[-1]);

            let h = math::ones(&[1i32], ctx) * F::from(0.05f64).unwrap();

            let m: i32 = 1;
            let n: i32  = 2;

            let losses = (-n/2..n/2+1)
                .map(|i| {
                    let voli = vol + (h * F::from(i).unwrap());
                    let pred = put(&spot, &strike, &voli, &dividends, r, t);
                    math::abs(put_price - pred)
                })
                .collect::<Vec<_>>();
            let grad = math::finite_difference(m as usize, n as usize, h, &losses[..]);

            let mut feeder = ag::Feeder::new();
            feeder
                .push(put_price, p.view())
                .push(spot, s.view())
                .push(strike, k.view())
                .push(dividends, q.view());

            adam.update(&[vol], &[grad], ctx, feeder);
        });
    }

    env.get_array_by_id(ret_id).unwrap().clone().into_inner()
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum OptionType {
    Call,
    Put,
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
