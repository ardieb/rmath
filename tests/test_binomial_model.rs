extern crate autograd as ag;
extern crate rquant;

use ag::ndarray as nd;
use ag::prelude::*;

#[test]
fn test_price_single_call() {
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
    env.run(|ctx: &mut ag::Context<f64>| {
        let s = ctx.variable("s");
        let k = ctx.variable("k");
        let vol = ctx.variable("vol");
        let q = ctx.variable("q");
        let r = risk_free_interest_rate;
        let t = time_to_maturity;

        let c = rquant::options::binomial::call(&s, &k, &vol, &q, r, t);
        let call_price = c.eval(ctx).expect("Could not evaluate call option price!");

        let implied_volatility = rquant::options::binomial::call_iv(
            call_price.view(),
            spot_price.view(),
            strike_price.view(),
            dividends.view(),
            r,
            t,
        );
        assert!((implied_volatility - volatility).sum().abs() <= 1e-2);
    });
}

#[test]
fn test_price_single_put() {
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
    env.run(|ctx: &mut ag::Context<f64>| {
        let s = ctx.variable("s");
        let k = ctx.variable("k");
        let vol = ctx.variable("vol");
        let q = ctx.variable("q");
        let r = risk_free_interest_rate;
        let t = time_to_maturity;

        let c = rquant::options::binomial::put(&s, &k, &vol, &q, r, t);
        let put_price = c.eval(ctx).expect("Could not evaluate put option price!");

        let implied_volatility = rquant::options::binomial::put_iv(
            put_price.view(),
            spot_price.view(),
            strike_price.view(),
            dividends.view(),
            r,
            t,
        );
        assert!((implied_volatility - volatility).sum().abs() <= 1e-1);
    });
}

#[test]
fn test_price_many_calls() {
    let spot_price = nd::array![40.74, 50.20, 1.2, 1010.].into_dyn();
    let strike_price = nd::array![30., 30., 4.8, 999.].into_dyn();
    let volatility = nd::array![0.5654, 0.2, 1.2, 0.7].into_dyn();
    let dividends = nd::array![0., 0., 0., 0.].into_dyn();

    let mut env = ag::VariableEnvironment::new();
    env.name("s").set(spot_price.clone());
    env.name("k").set(strike_price.clone());
    env.name("vol").set(volatility.clone());
    env.name("q").set(dividends.clone());

    let risk_free_interest_rate = 0.025;
    let time_to_maturity = 189. / 365.;
    env.run(|ctx: &mut ag::Context<f64>| {
        let s = ctx.variable("s");
        let k = ctx.variable("k");
        let vol = ctx.variable("vol");
        let q = ctx.variable("q");
        let r = risk_free_interest_rate;
        let t = time_to_maturity;

        let c = rquant::options::binomial::call(&s, &k, &vol, &q, r, t);
        let call_price = c.eval(ctx).expect("Could not evaluate call option price!");

        let implied_volatility = rquant::options::binomial::call_iv(
            call_price.view(),
            spot_price.view(),
            strike_price.view(),
            dividends.view(),
            r,
            t,
        );
        assert!((implied_volatility - volatility).iter().all(|diff| diff.abs() <= 1e-1));
    });
}

#[test]
fn test_price_many_puts() {
    let spot_price = nd::array![40.74, 50.20, 1.2, 1010.].into_dyn();
    let strike_price = nd::array![30., 30., 4.8, 999.].into_dyn();
    let volatility = nd::array![0.5654, 0.2, 1.2, 0.7].into_dyn();
    let dividends = nd::array![0., 0., 0., 0.].into_dyn();

    let mut env = ag::VariableEnvironment::new();
    env.name("s").set(spot_price.clone());
    env.name("k").set(strike_price.clone());
    env.name("vol").set(volatility.clone());
    env.name("q").set(dividends.clone());

    let risk_free_interest_rate = 0.025;
    let time_to_maturity = 189. / 365.;
    env.run(|ctx: &mut ag::Context<f64>| {
        let s = ctx.variable("s");
        let k = ctx.variable("k");
        let vol = ctx.variable("vol");
        let q = ctx.variable("q");
        let r = risk_free_interest_rate;
        let t = time_to_maturity;

        let p = rquant::options::binomial::put(&s, &k, &vol, &q, r, t);
        let put_price = p.eval(ctx).expect("Could not evaluate call option price!");

        let implied_volatility = rquant::options::binomial::call_iv(
            put_price.view(),
            spot_price.view(),
            strike_price.view(),
            dividends.view(),
            r,
            t,
        );
        assert!((implied_volatility - volatility).iter().all(|diff| diff.abs() <= 1e-1));
    });
}
