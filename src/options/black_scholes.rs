use autograd as ag;
use autograd::ndarray as nd;
use autograd::ndarray_ext as arr;

use ag::tensor::Variable;

use crate::stats;

/// Calculate the price of a call option based on the
/// Black Scholes Merton model for options pricing.
///
/// This function can price multiple options at once by inputing
/// a multidimensional set of inputs. All multi dimensional inputs
/// must have the same shape.
///
/// :param: `g` The graph which generated the tensors.
/// :param: `s` The underlying stocks' prices per share.
/// :param: `k` The options' strike prices per share.
/// :param: `vol` The volatility of the stocks in decimal.
/// :param: `t` The time until option maturity as decimal of a year.
/// :param: `r` The risk free interest rate as decimal.
/// :param: `q` The continously compounding divdend yield in decimal.
///
/// :return: `prices` The price of the options.
pub fn call<'graph, F: ag::Float>(
    g: &'graph ag::Graph<F>,
    s: &ag::Tensor<'graph, F>,
    k: &ag::Tensor<'graph, F>,
    vol: &ag::Tensor<'graph, F>,
    t: &ag::Tensor<'graph, F>,
    q: &ag::Tensor<'graph, F>,
    r: F,
) -> ag::Tensor<'graph, F> {
    let half = F::from(0.5f64).unwrap();
    let one = F::one();
    let zero = F::zero();
    // d1 = ln(s/k) + (vol^2 / 2 + r) * t
    //      -----------------------------
    //             vol * sqrt(t)
    let d1 = (g.ln(s / k) + ((g.square(vol) * half) + r - q) * t) / (vol * g.sqrt(t));
    // d2 = d1 - vol * sqrt(t)
    let d2 = d1 - (vol * g.sqrt(t));
    let nd1 = stats::normal::cdf(g, &d1, zero, one);
    let nd2 = stats::normal::cdf(g, &d2, zero, one);
    s * g.exp(g.neg(t * q)) * nd1 - k * g.exp(g.neg(t * r)) * nd2
}

/// Calculate the price of a put option based on the
/// Black Scholes Merton model for options pricing.
///
/// This function can price multiple options at once by inputing
/// a multidimensional set of inputs. All multi dimensional inputs
/// must have the same shape.
///
/// :param: `g` The graph which generated the tensors.
/// :param: `s` The underlying stocks' prices .
/// :param: `k` The options' strike prices.
/// :param: `vol` The volatility of the stocks.
/// :param: `t` The time until option maturity.
/// :param: `r` The risk free interest rate.
/// :param: `q` The continously compounding divdend yield in decimal.
///
/// :return: `prices` The price of the options.
pub fn put<'graph, F: ag::Float>(
    g: &'graph ag::Graph<F>,
    s: &ag::Tensor<'graph, F>,
    k: &ag::Tensor<'graph, F>,
    vol: &ag::Tensor<'graph, F>,
    t: &ag::Tensor<'graph, F>,
    q: &ag::Tensor<'graph, F>,
    r: F,
) -> ag::Tensor<'graph, F> {
    let half = F::from(0.5f64).unwrap();
    let one = F::one();
    let zero = F::zero();
    // d1 = ln(s/k) + (vol^2 / 2 + r) * t
    //      -----------------------------
    //             vol * sqrt(t)
    let d1 = (g.ln(s / k) + ((g.square(vol) * half) + r - q) * t) / (vol * g.sqrt(t));
    // d2 = d1 - vol * sqrt(t)
    let d2 = d1 - (vol * g.sqrt(t));
    let nnegd1 = stats::normal::cdf(g, &g.neg(d1), zero, one);
    let nnegd2 = stats::normal::cdf(g, &g.neg(d2), zero, one);
    k * g.exp(g.neg(t * r)) * nnegd2 - s * g.exp(g.neg(t * q)) * nnegd1
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
/// :param: `t` The time until option maturity.
/// :param: `r` The risk free interest rate.
/// :param: `q` The continously compounding divdend yield in decimal.
///
/// :return: `prices` The price of the options.
pub fn call_iv<'graph, F: ag::Float>(
    c: ag::NdArrayView<F>,
    s: ag::NdArrayView<F>,
    k: ag::NdArrayView<F>,
    t: ag::NdArrayView<F>,
    q: ag::NdArrayView<F>,
    r: F,
) -> ag::NdArray<F> {
    let half = F::from(0.5f64).unwrap();
    let iv = arr::into_shared(nd::Array::from_elem(c.shape(), half));
    let adam_state = ag::optimizers::adam::AdamState::new(&[&iv]);

    ag::with(|g| {
        for _ in 0..1000 {
            let vol = g.variable(iv.clone());
            let call_price = g.placeholder(&[-1]);
            let spot = g.placeholder(&[-1]);
            let strike = g.placeholder(&[-1]);
            let expir = g.placeholder(&[-1]);
            let dividends = g.placeholder(&[-1]);
            let pred = call(g, &spot, &strike, &vol, &expir, &dividends, r);

            let losses = g.abs(call_price - pred);
            let grads = g.grad(&[losses], &[vol]);

            let update_ops: Vec<ag::Tensor<F>> =
                ag::optimizers::adam::Adam::default().compute_updates(&[vol], &grads, &adam_state, g);

            g.eval(&update_ops, &[
                call_price.given(c.view()),
                spot.given(s.view()),
                strike.given(k.view()),
                expir.given(t.view()),
                dividends.given(q.view()),
            ]);
        }
    });
    let result = {
        let locked = iv
            .read()
            .expect("Failed to read the iv array!");
        locked.clone()
    };
    result
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
/// :param: `t` The time until option maturity.
/// :param: `r` The risk free interest rate.
/// :param: `q` The continously compounding divdend yield in decimal.
///
/// :return: `prices` The price of the options.
pub fn put_iv<'graph, F: ag::Float>(
    p: ag::NdArrayView<F>,
    s: ag::NdArrayView<F>,
    k: ag::NdArrayView<F>,
    t: ag::NdArrayView<F>,
    q: ag::NdArrayView<F>,
    r: F,
) -> ag::NdArray<F> {
    let half = F::from(0.5f64).unwrap();
    let iv = arr::into_shared(nd::Array::from_elem(p.shape(), half));
    let adam_state = ag::optimizers::adam::AdamState::new(&[&iv]);

    ag::with(|g| {
        for _ in 0..10000 {
            let vol = g.variable(iv.clone());
            let put_price = g.placeholder(&[-1]);
            let spot = g.placeholder(&[-1]);
            let strike = g.placeholder(&[-1]);
            let expir = g.placeholder(&[-1]);
            let dividends = g.placeholder(&[-1]);
            let pred = put(g, &spot, &strike, &vol, &expir, &dividends, r);

            let losses = g.abs(put_price - pred);
            let grads = g.grad(&[losses], &[vol]);

            let update_ops: Vec<ag::Tensor<F>> =
                ag::optimizers::adam::Adam::default().compute_updates(&[vol], &grads, &adam_state, g);

            g.eval(&update_ops, &[
                put_price.given(p.view()),
                spot.given(s.view()),
                strike.given(k.view()),
                expir.given(t.view()),
                dividends.given(q.view()),
            ]);
        }
    });
    let result = {
        let locked = iv
            .read()
            .expect("Failed to read the iv array!");
        locked.clone()
    };
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    use autograd::ndarray as nd;

    #[test]
    fn test_iv() {
        let spot_price = nd::array![100., 200., 300., 400.]
            .into_dyn();
        let time_to_maturity = nd::array![0.05, 0.05, 0.05, 0.05]
            .into_dyn();
        let strike_price = nd::array![110., 300., 700., 440.]
            .into_dyn();
        let volatility = nd::array![0.70, 0.90, 0.40, 0.50]
            .into_dyn();
        let dividends = nd::array![0., 0., 0., 0.,]
            .into_dyn();
        let risk_free_interest_rate = 0.025;

        ag::with(|g| {
            let s = g.variable(spot_price.clone());
            let t = g.variable(time_to_maturity.clone());
            let k = g.variable(strike_price.clone());
            let vol = g.variable(volatility.clone());
            let q = g.variable(dividends.clone());
            let r = risk_free_interest_rate;
            let call_price = call(g, &s, &k, &vol, &t, &q, r);
            let c = call_price.eval(&[])
                .expect("Could not evaluate call option price!");
            let iv = call_iv(
                c.view(),
                spot_price.view(),
                strike_price.view(),
                time_to_maturity.view(),
                dividends.view(),
                r,
            );
            println!("Actual: {:?}, Predicted: {:?}", volatility.view(), iv.view());
        });
    }
}
