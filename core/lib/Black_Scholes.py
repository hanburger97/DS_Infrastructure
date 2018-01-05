
import scipy.stats as stats
import numpy as np
from numpy import shape
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import animation

class Black_Scholes:
    # Base class for all of the theoretical model calculations
    #   BSM pricing, and greek letters
    #   Each Object should correspond to one single contract... yes very inneficient

    def __init__(self, option_type, price, strike, interest_rate, expiry, volatility, dividend_yield=0):
        self.s = price  # Underlying asset price
        self.k = strike  # Option strike K
        self.r = interest_rate  # Continuous risk fee rate
        self.q = dividend_yield  # Dividend continuous rate
        self.T = expiry  # time to expiry (year)
        self.sigma = volatility  # Underlying volatility
        self.type = option_type  # option type "p" put option "c" call option

        self.d1 = self.d1()
        self.d2 = self.d2()

        #Not used since the input for sigma is the IV counter derived from price
        #self.theoretical_price = self.bsm_price()


        self.calc_greeks()

    def calc_greeks(self):

        self.delta = self.delta()
        self.gamma = self.gamma()
        self.theta = self.theta()
        self.kappa = self.kappa()
        self.rho = self.rho()

    def n(self, d):
        # cumulative probability distribution function of standard normal distribution
        return stats.norm.cdf(d)

    def dn(self, d):
        # the first order derivative of n(d)
        return stats.norm.pdf(d)

    # Implementing d1 and d2 for BSM
    def d1(self):
        if (self.sigma * np.sqrt(self.T) == 0):
            return (np.log(self.s / self.k) + (self.r - self.q + (self.sigma ** 2) * 0.5) * self.T) / 0.0000001
        else: return (np.log(self.s / self.k) + (self.r - self.q + (self.sigma ** 2) * 0.5) * self.T) / float(self.sigma * np.sqrt(self.T))


    def d2(self):
        #d2 = (np.log(self.s / self.k) + (self.r - self.q - (self.sigma ** 2 * 0.5) * self.T) / (self.sigma * np.sqrt(self.T))
        return self.d1 - (self.sigma * np.sqrt(self.T))

    def bsm_price(self):
        d1 = self.d1
        d2 = self.d2
        if self.type == 'c':
            price = np.exp(-self.r * self.T) * (
            self.s * np.exp((self.r - self.q) * self.T) * self.n(d1) - self.k * self.n(d2))
            return price
        elif self.type == 'p':
            price = np.exp(-self.r * self.T) * (
            self.k * self.n(-d2) - (self.s * np.exp((self.r - self.q) * self.T) * self.n(-d1)))
            return price
        else:
            print("option type can only be c or p")


    def bsm_iv(self, price=None):

        epsilon = 0.00001
        upper_sigma = 500.0
        max_sigma = 500.0
        min_sigma = 0.0001
        lower_sigma = 0.0001
        mid_sigma = 0.0
        iteration = 0

        oprice = price

        while True:

            iteration += 1
            mid_sigma = (upper_sigma + lower_sigma) / 2.0
            price = self.bsm_price()

            if self.type == 'c':

                lower_price = self.bsm_static_pricing(self.type, lower_sigma, self.s, self.k, self.r, self.T, self.q)
                if (lower_price - oprice) * (price - oprice) > 0:
                    lower_sigma = mid_sigma
                else:
                    upper_sigma = mid_sigma
                if abs(price - oprice) < epsilon: break
                if mid_sigma > max_sigma - 5:
                    mid_sigma = 0.000001
                    break
                    #             print("mid_vol=%f" %mid_vol)
                    #             print("upper_price=%f" %lower_price)

            elif self.type == 'p':
                upper_price = self.bsm_static_pricing(self.type, upper_sigma, self.s, self.k, self.r, self.T, self.q)

                if (upper_price - oprice) * (price - oprice) > 0:
                    upper_sigma = mid_sigma
                else:
                    lower_sigma = mid_sigma
                    #             print("mid_vol=%f" %mid_vol)
                    #             print("upper_price=%f" %upper_price)
                if abs(price - oprice) < epsilon: break
                if iteration > 100: break

        return mid_sigma


    # Just a static function callable without initializing an object mostly for reverse calculate IV
    @staticmethod
    def bsm_static_pricing(
            option_type,
            sigma,
            underlying_price,
            strike,
            time,
            riskfreerate=None,
            dividendrate=None,
            ):

        o, s, k, t, r, q = option_type, underlying_price, strike, time, riskfreerate, dividendrate
        q = q if q else 0.0
        r = r if r else 0.0

        sigma = float(sigma)

        price = 0.0

        d1 = (np.log(s / k) + (r - q + 0.5 * (sigma ** 2)) * t) / (sigma * np.sqrt(t))
        d2 = d1 - (sigma * np.sqrt(t))

        try:
            assert o == 'c' or o == 'p'
            if o == 'c':
                price = np.exp(-r * t) * (s * np.exp((r - q) * t) * stats.norm.cdf(d1) - k * stats.norm.cdf(d2))
            elif o == 'p':
                price = np.exp(-r * t) * (k * stats.norm.cdf(-d2) - s * np.exp((r - q) * t) * stats.norm.cdf(-d1))

        except Exception as e:
            print(e.args)
        except AssertionError:
            print('Option type must be "c" or "p"')

        return price



    ''' Theoretical Greek letters computations for European options on an asset that provides a yield at rate q '''

    def delta(self):
        d1 = self.d1
        if self.type == "c":
            return np.exp(-self.q * self.T) * self.n(d1)
        elif self.type == "p":
            return np.exp(-self.q * self.T) * (self.n(d1) - 1)

    def gamma(self):
        d1 = self.d1
        dn1 = self.dn(d1)
        return dn1 * np.exp(-self.q * self.T) / (self.s * self.sigma * np.sqrt(self.T))

    def theta(self):
        d1 = self.d1
        d2 = self.d2
        dn1 = self.dn(d1)

        if self.type == "c":
            theta = -self.s * dn1 * self.sigma * np.exp(-self.q * self.T) / (2 * np.sqrt(self.T)) \
                    + self.q * self.s * self.n(d1) * np.exp(-self.q * self.T) \
                    - self.r * self.k * np.exp(-self.r * self.T) * self.n(d2)


            return theta
        elif self.type == "p":
            theta = -self.s * dn1 * self.sigma * np.exp(-self.q * self.T) / (2 * np.sqrt(self.T)) \
                    - self.q * self.s * self.n(-d1) * np.exp(-self.q * self.T) \
                    + self.r * self.k * np.exp(-self.r * self.T) * self.n(-d2)
            return theta

    def kappa(self):
        d1 = self.d1
        dn1 = self.dn(d1)
        return self.s * np.sqrt(self.T) * dn1 * np.exp(-self.q * self.T)

    def rho(self):

        d2 = self.d2
        rho = 0.0
        if self.type == "c":
            rho = self.k * self.T * (np.exp(-self.r * self.T)) * self.n(d2)
        elif self.type == "p":
            rho = -self.k * self.T * (np.exp(-self.r * self.T)) * self.n(-d2)

        return rho