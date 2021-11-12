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
    ag::Tensor::builder()
        .set_inputs(&[ag::tensor::Input::new(s), ag::tensor::Input::new(k), ag::tensor::Input::new(vol), ag::tensor::Input::new(q)])
        .build(g, Bopm { r, t, ty: OptionType::Call })
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
    ag::Tensor::builder()
        .set_inputs(&[ag::tensor::Input::new(s), ag::tensor::Input::new(k), ag::tensor::Input::new(vol), ag::tensor::Input::new(q)])
        .build(g, Bopm { r, t, ty: OptionType::Put })
}


#[derive(Copy, Clone, Eq, PartialEq)]
enum OptionType {
    Call,
    Put,
}

struct SingleBopm<F: ag::Float> {
    s: F,
    k: F,
    vol: F,
    q: F,
    r: F,
    t: F,
    ty: OptionType
}

impl<F: ag::Float> SingleBopm<F> {

    fn eval(&self) -> F {
        let dt: F = F::one() / F::from(365f64).unwrap();
        let u: F = (self.vol * dt.sqrt()).exp();
        let d: F = (-self.vol * dt.sqrt()).exp();
        let p: F = (((self.r - self.q) * dt).exp() - d) / (u - d);
        let n: usize = num::NumCast::from(self.t / dt).unwrap();
        
        let mut dp: ag::NdArray<F> = arr::zeros(&[n+1, n+1]);

        for j in 0..n+1 {
            let us = j as i32;
            let steps = n as i32;
            let stock_price = self.s * u.powi(2 * (us - steps));
            let exercise_profit = match self.ty {
                OptionType::Call => (stock_price - self.k).max(F::zero()),
                OptionType::Put => (self.k - stock_price).max(F::zero())
            };
            dp[[n, j]] = exercise_profit;
        }

        for i in (0..n).rev() {
            for j in (0..n).rev() {
                let us = j as i32;
                let steps = n as i32;
                let stock_price = self.s * u.powi(2 * us - steps);
                let exercise_profit = match self.ty {
                    OptionType::Call => (stock_price - self.k).max(F::zero()),
                    OptionType::Put => (self.k - stock_price).max(F::zero())
                };
                let decay = (-self.r * dt).exp();
                let expected = p * dp[[i+1, j+1]]
                    + (F::one() - p) * dp[[i+1, j]];
                let binom = decay * expected;
                dp[[i, j]] = binom.max(exercise_profit);
            }
        }

        dp[[0, 0]]
    }

}

struct Bopm<F: ag::Float> {
    r: F,
    t: F,
    ty: OptionType,
}

impl<F: ag::Float> ag::op::Op<F> for Bopm<F> {
    fn compute(&self, ctx: &mut ag::op::ComputeContext<F>) {
        let s = ctx.input(0);
        let k = ctx.input(1);
        let vol = ctx.input(2);
        let q = ctx.input(3);

        let mut res: ag::NdArray<F> = arr::zeros(s.shape());
        nd::Zip::from(&mut res).and(&s).and(&k).and(&vol).and(&q).apply(|res, &s, &k, &vol, &q| {
            let opt = SingleBopm { s, k, vol, q, r: self.r, t: self.t, ty: self.ty };
            *res = opt.eval();
        });

        ctx.append_output(res);
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<F>) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ag::tensor::Variable;

    #[test]
    fn test_expand() {
        ag::with(|g| {
            let arr: ag::Tensor<f64> = g.zeros(&[10]);
            let arr2 = g.expand_dims(arr, &[-1i32]);
            println!("{:?}", arr2.eval(&[]).expect("Failed").shape());
        });
    }

    #[test]
    fn test_call() {
        let spot_price = nd::array![40.71].into_dyn();
        let strike_price = nd::array![30.].into_dyn();
        let volatility = nd::array![0.5654].into_dyn();
        let dividends = nd::array![0.].into_dyn();
        let risk_free_interest_rate = 0.025;
        let time_to_maturity = 190. / 365.;
        ag::with(|g| {
            let s = g.variable(spot_price.clone());
            let k = g.variable(strike_price.clone());
            let vol = g.variable(volatility.clone());
            let q = g.variable(dividends.clone());
            let r = risk_free_interest_rate;
            let t = time_to_maturity;
            let call_price = call(g, &s, &k, &vol, &q, r, t);
            let c = call_price
                .eval(&[])
                .expect("Could not evaluate call option price!");
            println!("{:?}", c.view());
        })
    }
}
