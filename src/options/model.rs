use autograd as ag;


#[derive(Copy, Clone, Eq, PartialEq)]
pub enum OptionType {
    Call,
    Put,
}

pub trait OptionPricingModel {
    /// Calculate the price of an option based on the
    /// model's pricing solution.
    ///
    /// This function can price multiple options at once by inputing
    /// a multidimensional set of inputs. All multi dimensional inputs
    /// must have the same shape.
    ///
    /// * `ty`: The type of the option, `Call` or `Put`.
    /// * `s`: The underlying stocks' prices per share.
    /// * `k`: The options' strike prices per share.
    /// * `vol`: The volatility of the stocks in decimal.
    /// * `r`: The risk free interest rate as decimal.
    /// * `q`: The divided of the stock per year as decimal.
    /// * `t`: The time until option maturity as decimal of a year.
    ///
    /// * `prices`: The price of the options.
    fn price<'graph, A, F: ag::Float>(ty: OptionType, s: A, k: A, vol: A, q: A, r: F, t: F) -> ag::Tensor<'graph, F> 
    where
        A: AsRef<ag::Tensor<'graph, F>> + Copy;
    
    /// Calculate the implied volatility based on the
    /// model's pricing solution.
    ///
    /// This function can price multiple options at once by inputing
    /// a multidimensional set of inputs. All multi dimensional inputs
    /// must have the same shape.
    ///
    /// * `ty`: The type of the option, `Call` or `Put`.
    /// * `p`: The price of the options.
    /// * `s`: The underlying stocks' prices per share.
    /// * `k`: The options' strike prices per share.
    /// * `r`: The risk free interest rate as decimal.
    /// * `q`: The divided of the stock per year as decimal.
    /// * `t`: The time until option maturity as decimal of a year.
    ///
    /// * `volatility`: The implied volatility of the options.
    fn implied_volatility<F: ag::Float>(
        ty: OptionType, 
        p: ag::NdArrayView<F>,
        s: ag::NdArrayView<F>,
        k: ag::NdArrayView<F>,
        q: ag::NdArrayView<F>,
        r: F,
        t: F,
    ) -> ag::NdArray<F>;

    /// Calculate the `delta` e.g. change in option price per change in underlying
    /// stock price.
    /// 
    /// This function can price multiple options at once by inputing
    /// a multidimensional set of inputs. All multi dimensional inputs
    /// must have the same shape.
    ///
    /// * `ty`: The type of the option, `Call` or `Put`.
    /// * `s`: The underlying stocks' prices per share.
    /// * `k`: The options' strike prices per share.
    /// * `vol`: The volatility of the stocks in decimal.
    /// * `r`: The risk free interest rate as decimal.
    /// * `q`: The divided of the stock per year as decimal.
    /// * `t`: The time until option maturity as decimal of a year.
    ///
    /// * `delta`: The change in option value per change in underlying stock price.
    fn delta<'graph, A, F: ag::Float>(ty: OptionType, s: A, k: A, vol: A, q: A, r: F, t: F) -> ag::Tensor<'graph, F> 
    where
        A: AsRef<ag::Tensor<'graph, F>> + Copy;

    /// Calculate the `theta` e.g. change in option price per
    /// change in time to expiration.
    /// 
    /// This function can price multiple options at once by inputing
    /// a multidimensional set of inputs. All multi dimensional inputs
    /// must have the same shape.
    ///
    /// * `ty`: The type of the option, `Call` or `Put`.
    /// * `s`: The underlying stocks' prices per share.
    /// * `k`: The options' strike prices per share.
    /// * `vol`: The volatility of the stocks in decimal.
    /// * `r`: The risk free interest rate as decimal.
    /// * `q`: The divided of the stock per year as decimal.
    /// * `t`: The time until option maturity as decimal of a year.
    ///
    /// * `theta`: The change in option value per change in time to experiation.
    fn theta<'graph, A, F: ag::Float>(ty: OptionType, s: A, k: A, vol: A, q: A, r: F, t: F) -> ag::Tensor<'graph, F> 
    where
        A: AsRef<ag::Tensor<'graph, F>> + Copy;

    /// Calculate the `gamma` e.g. change in option `delta` per
    /// change in time to expiration.
    /// 
    /// This function can price multiple options at once by inputing
    /// a multidimensional set of inputs. All multi dimensional inputs
    /// must have the same shape.
    ///
    /// * `ty`: The type of the option, `Call` or `Put`.
    /// * `s`: The underlying stocks' prices per share.
    /// * `k`: The options' strike prices per share.
    /// * `vol`: The volatility of the stocks in decimal.
    /// * `r`: The risk free interest rate as decimal.
    /// * `q`: The divided of the stock per year as decimal.
    /// * `t`: The time until option maturity as decimal of a year.
    ///
    /// * `gamma`: The change in option `delta` per change in time to expiration.
    fn gamma<'graph, A, F: ag::Float>(ty: OptionType, s: A, k: A, vol: A, q: A, r: F, t: F) -> ag::Tensor<'graph, F> 
    where
        A: AsRef<ag::Tensor<'graph, F>> + Copy;

    /// Calculate the `vega` e.g. change in option price per
    /// change in volatility.
    /// 
    /// This function can price multiple options at once by inputing
    /// a multidimensional set of inputs. All multi dimensional inputs
    /// must have the same shape.
    ///
    /// * `ty`: The type of the option, `Call` or `Put`.
    /// * `s`: The underlying stocks' prices per share.
    /// * `k`: The options' strike prices per share.
    /// * `vol`: The volatility of the stocks in decimal.
    /// * `r`: The risk free interest rate as decimal.
    /// * `q`: The divided of the stock per year as decimal.
    /// * `t`: The time until option maturity as decimal of a year.
    ///
    /// * `vega`: The change in option price per change in volatility.
    fn vega<'graph, A, F: ag::Float>(ty: OptionType, s: A, k: A, vol: A, q: A, r: F, t: F) -> ag::Tensor<'graph, F> 
    where
        A: AsRef<ag::Tensor<'graph, F>> + Copy;
}