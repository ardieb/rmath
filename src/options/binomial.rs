use autograd as ag;
use autograd::ndarray as nd;
use autograd::ndarray_ext as arr;

use num;
use statrs;

use crate::stats;

/// Calculate the price of a call option based on the
/// Binomial model for options pricing.
///
/// This function can price multiple options at once by inputing
/// a multidimensional set of inputs. All multi dimensional inputs
/// must have the same shape.
///
/// :param: `g` The graph which generated the tensors.
/// :param: `s` The underlying stocks' prices per share.
/// :param: `k` The options' strike prices per share.
/// :param: `vol` The volatility of the stocks in decimal.
/// :param: `r` The risk free interest rate as decimal.
/// :param: `q` The divided of the stock per year as decimal.
/// :param: `t` The time until option maturity as decimal of a year.
///
/// :return: `prices` The price of the options.
pub fn call<'graph, F: ag::Float>(
    g: &'graph ag::Graph<F>,
    s: &ag::Tensor<'graph, F>,
    k: &ag::Tensor<'graph, F>,
    vol: &ag::Tensor<'graph, F>,
    q: &ag::Tensor<'graph, F>,
    r: F,
    t: F,
) -> ag::Tensor<'graph, F> {
    let period = F::from::<f64>(365.).unwrap();
    let dt = F::one() / period;

    let u = g.exp(vol * dt.sqrt());
    let d = g.exp(g.neg(vol * dt.sqrt()));

    let p = g.neg(d - g.exp(g.neg(q - r) * dt)) / (u - d);

    helper(g, &u, &p, s, k, dt, t, r, F::zero(), F::zero(), true)
}

/// Calculate the price of a put option based on the
/// Binomial model for options pricing.
///
/// This function can price multiple options at once by inputing
/// a multidimensional set of inputs. All multi dimensional inputs
/// must have the same shape.
///
/// :param: `g` The graph which generated the tensors.
/// :param: `s` The underlying stocks' prices per share.
/// :param: `k` The options' strike prices per share.
/// :param: `vol` The volatility of the stocks in decimal.
/// :param: `r` The risk free interest rate as decimal.
/// :param: `q` The divided of the stock per year as decimal.
/// :param: `t` The time until option maturity as decimal of a year.
///
/// :return: `prices` The price of the options.
pub fn put<'graph, F: ag::Float>(
    g: &'graph ag::Graph<F>,
    s: &ag::Tensor<'graph, F>,
    k: &ag::Tensor<'graph, F>,
    vol: &ag::Tensor<'graph, F>,
    q: &ag::Tensor<'graph, F>,
    r: F,
    t: F,
) -> ag::Tensor<'graph, F> {
    let period = F::from::<f64>(365.).unwrap();
    let dt = F::one() / period;

    let u = g.exp(vol * dt.sqrt());
    let d = g.exp(g.neg(vol * dt.sqrt()));

    let p = g.neg(d - g.exp(g.neg(q - r) * dt)) / (u - d);

    helper(g, &u, &p, s, k, dt, t, r, F::zero(), F::zero(), false)
}

fn helper<'graph, F: ag::Float>(
    g: &'graph ag::Graph<F>,
    u: &ag::Tensor<'graph, F>,
    p: &ag::Tensor<'graph, F>,
    s: &ag::Tensor<'graph, F>,
    k: &ag::Tensor<'graph, F>,
    dt: F,
    t: F,
    r: F,
    cur_time: F,
    ups: F,
    call: bool,
) -> ag::Tensor<'graph, F> {
    let one = F::one();
    let two = one + one;
    let depth = cur_time / dt;
    let stock_price = s * g.pow(u, two * ups - depth);
    let all_zero = s * F::zero();

    let exercise_profit = if call {
        g.maximum(stock_price - k, all_zero)
    } else {
        g.maximum(k - stock_price, all_zero)
    };

    if t <= cur_time {
        return exercise_profit;
    }

    let decay = (-r * dt).exp();
    let expected = p * helper(g, u, p, s, k, dt, t, r, cur_time + dt, ups + one, call)
        + g.neg(p - one) * helper(g, u, p, s, k, dt, t, r, cur_time + dt, ups, call);
    let binom = expected * decay;
    g.maximum(binom, exercise_profit)
}
