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

            let m = 1;
            let n = 8;

            let losses = (-4..5)
                .map(|i: i32| {
                    let voli = vol + (h * F::from(i).unwrap());
                    let pred = call(&spot, &strike, &voli, &dividends, r, t);
                    math::abs(call_price - pred)
                })
                .collect::<Vec<_>>();
            let grad = math::finite_difference(m, n, h, &losses[..]);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_call() {
        let spot_price = nd::array![40.74].into_dyn();
        let strike_price = nd::array![30.].into_dyn();
        let volatility = nd::array![0.5654].into_dyn();
        let dividends = nd::array![0.].into_dyn();

        let mut env = ag::VariableEnvironment::new();
        env.name("s").set(spot_price.clone());
        env.name("k").set(strike_price.clone());
        env.name("vol").set(volatility.clone());
        env.name("q").set(dividends.clone());

        let risk_free_interest_rate = 0.025;
        let time_to_maturity = 189. / 365.;
        env.run(|ctx| {
            let s = ctx.variable("s");
            let k = ctx.variable("k");
            let vol = ctx.variable("vol");
            let q = ctx.variable("q");
            let r = risk_free_interest_rate;
            let t = time_to_maturity;

            let call_price = call(&s, &k, &vol, &q, r, t);
            let c = call_price
                .eval(ctx)
                .expect("Could not evaluate call option price!");
            println!(
                "Draft Kings Call Option Price: 189 days @ $30 {:?}",
                c.view()
            );

            let put_price = put(&s, &k, &vol, &q, r, t);
            let p = put_price
                .eval(ctx)
                .expect("Could not evaluate put option price!");
            println!(
                "Draft Kings Put Option Price: 189 days @ $30 {:?}",
                p.view()
            );

            let iv = call_iv(c.view(), spot_price.view(), strike_price.view(), dividends.view(), r, t);
            println!("Draft Kings Call Option Implied Volatility: {:?}", iv.view());
        })
    }

    #[test]
    fn test_concat() {
        ag::run(|ctx: &mut ag::Context<f64>| {
            let arr1 = math::ones(&[2], ctx);
            let arr2 = math::zeros(&[2], ctx);
            let arr1 = arr1.expand_dims(&[-1]);
            let arr2 = arr2.expand_dims(&[-1]);
            let arr3 = math::concat(&[arr1, arr2], 0);
            let output = arr3.eval(ctx).unwrap();
            println!("{:?}", output)
        })
    }
}
