use autograd as ag;
use autograd::ndarray_ext as arr;

use crate::stats;

use ag::tensor::Variable;

pub fn price_call_option<'graph, F: ag::Float>(
    g: &'graph ag::Graph<F>,
    spot_price: &ag::Tensor<'graph, F>,
    time_to_maturity: &ag::Tensor<'graph, F>,
    strike_price: &ag::Tensor<'graph, F>,
    volatility: &ag::Tensor<'graph, F>,
    risk_free_interest_rate: F,
) -> ag::Tensor<'graph, F> {
    let zero = F::zero();
    let one = F::one();
    let two = F::from(2f64).unwrap();
    let d1 = g.ln(spot_price / strike_price)
        + time_to_maturity * ((g.pow(volatility, two) / two) + risk_free_interest_rate);
    let d2 = d1 - volatility * g.sqrt(time_to_maturity);

    spot_price * stats::normal::cdf(g, &d1, zero, one)
        - strike_price
            * g.exp(g.neg(time_to_maturity * risk_free_interest_rate))
            * stats::normal::cdf(g, &d2, zero, one)
}

pub fn price_put_option<'graph, F: ag::Float>(
    g: &'graph ag::Graph<F>,
    spot_price: &ag::Tensor<'graph, F>,
    time_to_maturity: &ag::Tensor<'graph, F>,
    strike_price: &ag::Tensor<'graph, F>,
    volatility: &ag::Tensor<'graph, F>,
    risk_free_interest_rate: F,
) -> ag::Tensor<'graph, F> {
    let zero = F::zero();
    let one = F::one();
    let two = F::from(2f64).unwrap();
    let d1 = g.ln(spot_price / strike_price)
        + time_to_maturity * ((g.pow(volatility, two) / two) + risk_free_interest_rate);
    let d2 = d1 - volatility * g.sqrt(time_to_maturity);

    strike_price
        * g.exp(g.neg(time_to_maturity * risk_free_interest_rate))
        * stats::normal::cdf(g, &g.neg(d2), zero, one)
        - spot_price * stats::normal::cdf(g, &g.neg(d1), zero, one)
}

pub fn call_volatility<'graph, F: ag::Float>(
    g: &'graph ag::Graph<F>,
    given_call_price: &ag::Tensor<'graph, F>,
    spot_price: &ag::Tensor<'graph, F>,
    time_to_maturity: &ag::Tensor<'graph, F>,
    strike_price: &ag::Tensor<'graph, F>,
    risk_free_interest_rate: F,
    option_count: usize,
    epochs: usize,
) -> ag::Tensor<'graph, F> {
    let rng = arr::ArrayRng::<F>::default();
    let volatility_arr = arr::into_shared(rng.standard_uniform(&[option_count]));
    let adam_state = ag::optimizers::adam::AdamState::new(&[&volatility_arr]);

    for _ in 0..epochs {
        let volatility = g.variable(volatility_arr.clone());
        let call_price_pred = price_call_option(g, spot_price, time_to_maturity, strike_price, &volatility, risk_free_interest_rate);
        let mean_loss = g.sqrt(g.reduce_mean(g.square(call_price_pred - given_call_price), &[0], false));
        let grads = &g.grad(&[&mean_loss], &[volatility]);
        let update_ops: &[ag::Tensor<F>] = 
            &ag::optimizers::adam::Adam::default().compute_updates(&[volatility], grads, &adam_state, g);
        g.eval(update_ops, &[]);
    }
    g.variable(volatility_arr.clone())
}

pub fn put_volatility<'graph, F: ag::Float>(
    g: &'graph ag::Graph<F>,
    given_put_price: &ag::Tensor<'graph, F>,
    spot_price: &ag::Tensor<'graph, F>,
    time_to_maturity: &ag::Tensor<'graph, F>,
    strike_price: &ag::Tensor<'graph, F>,
    risk_free_interest_rate: F,
    option_count: usize,
    epochs: usize,
) -> ag::Tensor<'graph, F> {
    let rng = arr::ArrayRng::<F>::default();
    let volatility_arr = arr::into_shared(rng.standard_uniform(&[option_count]));
    let adam_state = ag::optimizers::adam::AdamState::new(&[&volatility_arr]);

    for _ in 0..epochs {
        let volatility = g.variable(volatility_arr.clone());
        let put_price_pred = price_put_option(g, spot_price, time_to_maturity, strike_price, &volatility, risk_free_interest_rate);
        let mean_loss = g.sqrt(g.reduce_mean(g.square(put_price_pred - given_put_price), &[0], false));
        let grads = &g.grad(&[&mean_loss], &[volatility]);
        let update_ops: &[ag::Tensor<F>] = 
            &ag::optimizers::adam::Adam::default().compute_updates(&[volatility], grads, &adam_state, g);
        g.eval(update_ops, &[]);
    }
    g.variable(volatility_arr.clone())
}