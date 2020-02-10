import numpy as np
import scipy as scp
import scipy.stats as ss
import sklearn.metrics as sm
from scipy.integrate import quad


def cf_Heston_schoutens(u, S, Tau, r, v0, kappa, theta, rho, vol_vol):
    """
    Heston characteristic function as proposed by Schoutens (2004)
    S      : Spot Price
    Tau    : Time to Maturity
    r      : Risk-Free Rate
    v0     : Instantaneous vol
    kappa  : Mean reversion rate
    theta  : Long term vol
    rho    : Corr of two Brownian Motion
    vol_vol: Vol of vol
    """
    phi = kappa - vol_vol * rho * u * 1j
    d = np.sqrt(((-phi) ** 2) - ((vol_vol ** 2) * (-1j * u - u ** 2)))
    g = (phi - d) / (phi + d)

    cf = np.exp((1j * u) * (np.log(S) + r * Tau)) * \
         np.exp((kappa * theta) / (vol_vol ** 2) * ((phi - d) * Tau - 2 * np.log((1 - g * np.exp(-d * Tau)) / (1 - g)))) * \
         np.exp((v0 / vol_vol ** 2) * (phi - d) * (1 - np.exp(-d * Tau)) / (1 - g * np.exp(-d * Tau)))
    return cf


def xi(u, alpha, S, Tau, r, v0, kappa, theta, rho, vol_vol):
    """
    Intermediate step of calculating the call price
    alpha : alpha th moment
    """
    cf = cf_Heston_schoutens(u - (alpha+1)*1j,
                             S, Tau, r, v0, kappa, theta, rho, vol_vol)
    numerator = np.exp(-r * Tau)*cf
    denominator = alpha**2 + alpha - u**2 + 1j*(2*alpha+1)*u
    return numerator/denominator


def integrand(u, alpha, S, K, Tau, r, v0, kappa, theta, rho, vol_vol):
    return (np.exp(-1j*u*np.log(K)))*xi(u, alpha, S, Tau, r, v0, kappa, theta, rho, vol_vol)


def quad_integrand(alpha, S, K, Tau, r, v0, kappa, theta, rho, vol_vol):
    return quad(integrand, limit=10000, a=0, b=np.inf, args=(alpha, S, K, Tau, r, v0, kappa, theta, rho, vol_vol))[0]


vec_quad_integrand = np.vectorize(quad_integrand)


def call_price_HS(alpha, S, K, Tau, r, par):
    v0, kappa, theta, rho, vol_vol = par
    multiplier = np.exp(-alpha*np.log(K))/np.pi
    return multiplier*vec_quad_integrand(alpha, S, K, Tau, r, v0, kappa, theta, rho, vol_vol)


if __name__=='__main__':
    par_1 = [0.013681, 1.605179, 0.053318, -0.6201, 0.590506]
    print(call_price_HS(0.75, 1, 1.2,1.6,0,par_1))