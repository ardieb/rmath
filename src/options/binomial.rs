use autograd as ag;
use autograd::array_gen as gen;
use autograd::ndarray as nd;
use autograd::tensor_ops as math;

use num;

/// Calculate the price of a call option based on the
/// Binomial model for options pricing.
///
/// This function can price multiple options at once by inputing
/// a multidimensional set of inputs. All multi dimensional inputs
/// must have the same shape.
///
/// :param: `s` The underlying stocks' prices per share.
/// :param: `k` The options' strike prices per share.
/// :param: `vol` The volatility of the stocks in decimal.
/// :param: `r` The risk free interest rate as decimal.
/// :param: `q` The divided of the stock per year as decimal.
/// :param: `t` The time until option maturity as decimal of a year.
///
/// :return: `prices` The price of the options.
pub fn call<'graph, A, F: ag::Float>(s: A, k: A, vol: A, q: A, r: F, t: F) -> ag::Tensor<'graph, F>
where
    A: AsRef<ag::Tensor<'graph, F>>,
{
    let dim: i32 = -1;
    let s = s.as_ref().expand_dims(&[dim]);
    let k = k.as_ref().expand_dims(&[dim]);
    let vol = vol.as_ref().expand_dims(&[dim]);
    let q = q.as_ref().expand_dims(&[dim]);
    let rs = (s/s) * r;
    let ts = (s/s) * t;

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
/// :param: `s` The underlying stocks' prices per share.
/// :param: `k` The options' strike prices per share.
/// :param: `vol` The volatility of the stocks in decimal.
/// :param: `r` The risk free interest rate as decimal.
/// :param: `q` The divided of the stock per year as decimal.
/// :param: `t` The time until option maturity as decimal of a year.
///
/// :return: `prices` The price of the options.
pub fn put<'graph, A, F: ag::Float>(s: A, k: A, vol: A, q: A, r: F, t: F) -> ag::Tensor<'graph, F>
where
    A: AsRef<ag::Tensor<'graph, F>>,
{
    let dim: i32 = -1;
    let s = s.as_ref().expand_dims(&[dim]);
    let k = k.as_ref().expand_dims(&[dim]);
    let vol = vol.as_ref().expand_dims(&[dim]);
    let q = q.as_ref().expand_dims(&[dim]);
    let rs = (s/s) * r;
    let ts = (s/s) * t;

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
    let n: usize = num::NumCast::from(t / dt).unwrap();

    let mut dp: ag::NdArray<F> = gen::zeros(&[n + 1, n + 1]);

    for j in 0..n + 1 {
        let us = j as i32;
        let steps = n as i32;
        let stock_price = s * u.powi(2 * (us - steps));
        let exercise_profit = match ty {
            OptionType::Call => (stock_price - k).max(F::zero()),
            OptionType::Put => (k - stock_price).max(F::zero()),
        };
        dp[[n, j]] = exercise_profit;
    }

    for i in (0..n).rev() {
        for j in (0..n).rev() {
            let us = j as i32;
            let steps = n as i32;
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
    use autograd::prelude::*;

    #[test]
    fn test_call() {
        let spot_price = nd::array![40.71].into_dyn();
        let strike_price = nd::array![30.].into_dyn();
        let volatility = nd::array![0.5654].into_dyn();
        let dividends = nd::array![0.].into_dyn();

        let mut env = ag::VariableEnvironment::new();
        env.name("s").set(spot_price);
        env.name("k").set(strike_price);
        env.name("vol").set(volatility);
        env.name("q").set(dividends);

        let risk_free_interest_rate = 0.025;
        let time_to_maturity = 190. / 365.;
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
            println!("{:?}", c.view());
        })
    }
}
