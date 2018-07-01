"""
[ RnD ]  Research Lab :: Options Library

Written by: Han Y. Xiao, Technical Partner

 Xiao Theodore & Co. | All Rights Reserved 2018

===============================================================

Nomenclature and Naming: Most Client's Methods are CAPITALIZED

"""

from __future__ import print_function

from threading import Thread, ThreadError

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from pandas_datareader.data import Options

from core.Options import Black_Scholes

############################################   Some Global Var         #################################################

SYNCFLAG = 0

############################################   Custom Error Classes    #################################################

class DataFormatError(Exception):

    def __init__(self, message):
        self.message = message

#############################################   Some Method Wrapper    #################################################


def parallelProcess(func):

    def wrapper(object):
        try:
            worker = Thread(target=func, args=(object,))
            worker.setDaemon(True)
            worker.start()
        except ThreadError as te:
            print(te)
        except Exception as e:
            print(e)
    return wrapper


def requireSyncFlag(func):

    def wrapper(object, *args):
        try:
            global SYNCFLAG
            #print('Data Building [ IN PROGRESS ] ')
            SYNCFLAG = 1
            func(object, *args)
            #print('Data Building [ COMPLETED ] ')
            SYNCFLAG = 0
        except Exception as e:
            print(e)
    return wrapper

def waitSyncFlag(func):
    def wrapper(object):
        try:
            global SYNCFLAG
            while SYNCFLAG == 1:
                pass
            func(object)
        except Exception as e:
            print(e)
    return wrapper



################################################      Main Class      ##################################################


class OptionLib:


    def __init__(self,
                 symbol='SPY',
                 dataprovider='yahoo',
                 riskfree=0.01,
                 dividendrate=0.01
                 ):

        self.SYMBOL = symbol
        self.data_provider = dataprovider

        self.__oldsymbol = symbol
        self.__olddataprovider = dataprovider

        self.risk_free_rate = riskfree
        self.dividend_rate = dividendrate

        self.opt = None

        self.IVs = {'c': [], 'p': []}

        self.__last_quote=None
        self.__underlying_price = None

        self.__data=None
        self.__data_core = None

        self.data_selection = (0, 'c')

        self.tickSize = 0.5
        self.data_init()





    #########################################################################################################
    #                                   Data Building and Housekeeping                                      #
    #########################################################################################################
    @waitSyncFlag
    def data_init(self):

        self.opt = Options(self.SYMBOL, self.data_provider)
        self.__underlying_price = self.opt.underlying_price
        self.data_building_core()
        self.lastGreeks = {
            'strike': None,
            'data': {},
            'S': [],
            'T': []
        }
        self.IVs = {'c': [], 'p': []}

        self.data_aggregate_IV()




    @waitSyncFlag
    def data_refresh(self):

        self.__underlying_price = self.opt.underlying_price
        self.data_building_core()
        self.lastGreeks = {
            'strike': None,
            'data': {},
            'S': [],
            'T': []
        }
        self.IVs = {'c': [],'p': []}
        self.data_aggregate_IV()

    @parallelProcess
    def data_auto_refresh(self):
        # This should be running in another thread
        while True:
            if not (self.SYMBOL == self.__oldsymbol and self.data_provider == self.__olddataprovider):
                self.__oldsymbol, self.__olddataprovider = self.SYMBOL, self.data_provider
                self.__last_quote = self.opt.get_call_data().iloc[0]['Quote_Time'].to_pydatetime()
                self.data_init()
                print('Data Initialization for {} [ COMPLETE ]'.format(self.SYMBOL))
            if not (self.opt.get_call_data().iloc[0]['Quote_Time'].to_pydatetime() == self.__last_quote):
                self.__last_quote = self.opt.get_call_data().iloc[0]['Quote_Time'].to_pydatetime()
                self.data_refresh()
                print('Data Refreshing for {} [ COMPLETE ]'.format(self.SYMBOL))


    @requireSyncFlag
    def data_building_core(self):
        try:
            assert self.opt
            df = self.opt.get_all_data()
            dflen = len(df.index.values)
            d = {}
            for i in range(dflen):

                dfindex = df.index.values[i]
                row = df.loc[dfindex]
                exp = dfindex[1].to_pydatetime()
                curr = row['Quote_Time'].to_pydatetime()
                toe = float((exp - curr).days)/365.0
                # Days until expiration
                dte = round(toe * 365) + 1
                otype = 'c' if dfindex[2] == 'call' else 'p'
                # Index will be using expiry_date index
                expd = exp.date()
                j = self.opt.expiry_dates.index(expd)
                bso = Black_Scholes(
                    option_type=otype,
                    price=row['Underlying_Price'],
                    strike=dfindex[0],
                    interest_rate=self.risk_free_rate,
                    dividend_yield=self.dividend_rate,
                    volatility=row['IV'],
                    expiry=toe
                )
                # Check if the index exists or not
                if not (j in d):
                    d[j] = {
                        'matrix': [],
                        'indexes': []
                    }
                # [ [ Ask, Bid, Last, Vol, %, sigma, delta, gamma, kappa, theta, rho, dte, symbol] , (...) ]
                d[j]['matrix'].append([
                    row['Ask'], row['Bid'], row['Last'], int(row['Vol']), row['PctChg'], round(row['IV'], 2), round(bso.delta, 2), round(bso.gamma,2),
                    round(bso.kappa, 2), round(bso.theta,  2), round(bso.rho, 2), dte, dfindex[3]
                ])
                # [ ( type, strike), ... ]
                d[j]['indexes'].append(
                    (otype, dfindex[0])
                )
            self.__data_core = d
        except AssertionError:
            raise DataFormatError('No Data from Yahoo, Check Internet Connection')

    @requireSyncFlag
    def data_aggregate_IV(self):
        """
        Aggregate Contract's sigma by call and puts then store them in [ Time to Expiration, Strike, Volatility ]
        format for later plotting
        """
        try:
            assert self.__data_core
            d = self.__data_core.copy()

            for t in d.keys():
                matrix = d[t]['matrix']
                indexes = d[t]['indexes']
                for i in range(len(matrix)):
                # Format : [ Time to Expiration, Strike, Volatility ]
                    self.IVs[indexes[i][0]].append(
                        [ matrix[i][11], indexes[i][1], matrix[i][5] ]
                    )

        except AssertionError:
            raise DataFormatError('Must input a pandas.DataFrame')

    def data_IVpT(self,
                expiry_index=0
                ):
        """
        Compute IV per timestamp
        :return: calls and puts
        """
        dt = self.__data_core[expiry_index].copy()
        matrix = dt['matrix']
        indexes = dt['indexes']
        calls, puts = [], []
        # calls = [ [ Strike, IV] , ... ]
        for i in range(len(matrix)):
            if indexes[i][0] == 'c':
                calls.append([indexes[i][1], matrix[i][5]])
            else:
                puts.append([indexes[i][1], matrix[i][5]])
        return calls, puts


    def data_aggregate_greeks(self,strike=None):
        """

        :return:
        """
        try:
            assert self.__data_core
            assert strike
            d = self.__data_core.copy()
            times = []
            sigmas = []

            for t in d.keys():
                matrix = d[t]['matrix']
                indexes = d[t]['indexes']
                for i in range(len(matrix)):
                    if indexes[i][0] == 'c' and indexes[i][1] == strike:
                        sigmas.append(matrix[i][5])
                        times.append(float(matrix[i][11]) / 365.0)

            # Matrices are n x m
            #   where n : the nb of contract_dates
            # compute underlyingPrice matrix

            underlyingPriceMatrix = np.array([
                np.arange(0.0,
                          self.__underlying_price * 2,
                          self.tickSize
                          )
                for i in range(len(sigmas))
            ])

            MatrixShape = underlyingPriceMatrix.shape

            sigmaMatrix = np.array([np.ones(MatrixShape[1]) * sigmas[i] for i in range(MatrixShape[0])]).reshape(MatrixShape)

            expiryMatrix = np.array([np.ones(MatrixShape[1]) * times[i] for i in range(MatrixShape[0])]).reshape(MatrixShape)

            ccontracts = []
            pcontracts = []

            for i in range(MatrixShape[0]):

                for j in range(MatrixShape[1]):
                    cbso = Black_Scholes(
                        option_type='c',
                        strike=strike,
                        price=underlyingPriceMatrix[i][j],
                        interest_rate=self.risk_free_rate,
                        dividend_yield=self.dividend_rate,
                        expiry=expiryMatrix[i][j],
                        volatility=sigmaMatrix[i][j]
                    )

                    pbso = Black_Scholes(
                        option_type='p',
                        strike=strike,
                        price=underlyingPriceMatrix[i][j],
                        interest_rate=self.risk_free_rate,
                        dividend_yield=self.dividend_rate,
                        expiry=expiryMatrix[i][j],
                        volatility=sigmaMatrix[i][j]
                    )
                    #Access elements in matrix[i][j]
                    ccontracts.append(cbso)
                    pcontracts.append(pbso)


            cdelta = np.array([c.delta for c in ccontracts]).reshape(MatrixShape)
            cgamma = np.array([c.gamma for c in ccontracts]).reshape(MatrixShape)
            ctheta = np.array([c.theta for c in ccontracts]).reshape(MatrixShape)
            ckappa = np.array([c.kappa for c in ccontracts]).reshape(MatrixShape)
            crho = np.array([c.rho for c in ccontracts]).reshape(MatrixShape)

            pdelta = np.array([p.delta for p in pcontracts]).reshape(MatrixShape)
            pgamma = np.array([p.gamma for p in pcontracts]).reshape(MatrixShape)
            ptheta = np.array([p.theta for p in pcontracts]).reshape(MatrixShape)
            pkappa = np.array([p.kappa for p in pcontracts]).reshape(MatrixShape)
            prho = np.array([p.rho for p in pcontracts]).reshape(MatrixShape)

            greeks = {
                'c':{},
                'p':{}
            }
            greeks['c']['delta'] = cdelta
            greeks['c']['gamma'] = cgamma
            greeks['c']['theta'] = ctheta
            greeks['c']['kappa'] = ckappa
            greeks['c']['rho'] = crho

            greeks['p']['delta'] = pdelta
            greeks['p']['gamma'] = pgamma
            greeks['p']['theta'] = ptheta
            greeks['p']['kappa'] = pkappa
            greeks['p']['rho'] = prho

            self.lastGreeks['data'] = greeks
            self.lastGreeks['strike'] = strike
            self.lastGreeks['S'] = underlyingPriceMatrix
            self.lastGreeks['T'] = expiryMatrix

        except AssertionError:
            pass
        pass


    #########################################################################################################
    #                                   Client's Methods and Properties                                     #
    #########################################################################################################


    @property
    def index(self):
        opt = self.opt
        d ={'Expiry Dates':opt.expiry_dates}
        return pd.DataFrame(data=d).transpose()

    @property
    def data(self):

        print('Underlying @ {:.2f} \nLatest Option Quote @: {}\n'.format(self.__underlying_price, self.__last_quote))
        print('Current Contracts Expires @ {}\n'.format(self.opt.expiry_dates[self.data_selection[0]]))
        obj = self.__data_core[self.data_selection[0]]
        data = np.array(obj['matrix'])
        indexes = obj['indexes']
        filtered_matrix = []
        filtered_indexes = []

        for i in range(len(data)):
            if indexes[i][0] == self.data_selection[1]:
                filtered_matrix.append(data[i])
                filtered_indexes.append(indexes[i][1])

        columns = [
            'Ask',
            'Bid',
            'Last',
            'Vol',
            '%',
            '\u03C3',
            '\u0394',
            '\u0393',
            '\u039A',
            '\u0398',
            '\u03A1',
            'Days to Expiry',
            'Symbol'
        ]
        df = pd.DataFrame(data=filtered_matrix, index=filtered_indexes, columns=columns)
        return df


    @property
    def VIEW_SELECTION(self):
        return 'CURRENTLY SELECTED DATA :: [ INDEX : {} | TYPE : {} | SYMBOL : {} ]'.format(
            self.data_selection[0], self.data_selection[1], self.SYMBOL
        )

    def select(self,
               INDEX=0,
               TYPE='c',
               ):
        """
        Setting the data-selecting tuple
        """
        try:
            assert TYPE == 'c' or TYPE == 'p'
            assert INDEX < len(self.opt.expiry_dates)
            self.data_selection = (INDEX, TYPE)
        except AssertionError:
            raise DataFormatError('Expiry index and option type ("c" or "p") must be valid')


    #########################################################################################################
    #                                       Plotting                                                        #
    #########################################################################################################

    def plot_smile(self,
                   expiry_index=None,
                   ):
        """
        Plot the IV smile for both calls and puts per timestamp
        """
        expiry_index = expiry_index if expiry_index else self.data_selection[0]
        calls, puts = self.data_IVpT(expiry_index=expiry_index)

        k_calls, IV_calls = [], []
        k_puts, IV_puts = [], []

        for el in calls:
            k_calls.append(el[0])
            IV_calls.append(el[1])
        for el in puts:
            k_puts.append(el[0])
            IV_puts.append(el[1])

        plt.figure(figsize=(16, 7))
        e = plt.scatter(k_calls, IV_calls, c='white', label="IV(call options)")
        f = plt.scatter(k_puts, IV_puts, c='red', label="IV(put options)")
        plt.xlabel('strike')
        plt.ylabel('Implied Volatility')
        plt.legend((e, f), ("IV (call options)", "IV (put options)"))


    def plot_surface(self,
                     option_type=None,
                     ):
        try:
            if option_type:
                assert option_type == 'c' or option_type =='p'
            option_type = option_type if option_type else self.data_selection[1]
            plotdata = self.IVs[option_type]
            color = 'red' if option_type == 'p' else 'green'

            xaxis = [plotdata[i][0] for i in range(len(plotdata))]
            yaxis = [plotdata[i][1] for i in range(len(plotdata))]
            zaxis = [plotdata[i][2] for i in range(len(plotdata))]

            fig1 = plt.figure(figsize=(20, 12))
            ax = fig1.add_subplot(111, projection='3d')
            ax.view_init()
            ax.scatter(xaxis, yaxis, zaxis, c=color)
            plt.xlabel("Time to Expiration (days)")
            plt.ylabel("Strikes")
            plt.title("Implied Volatility")

            fig2 = plt.figure(figsize=(20, 12))
            ax2 = fig2.add_subplot(111, projection='3d')
            ax2.view_init()
            ax2.plot_trisurf(xaxis, yaxis, zaxis, cmap=cm.jet)
            plt.xlabel("Time to Expiration (days)")
            plt.ylabel("Strikes")
            plt.title("Implied Volatility")

        except AssertionError:
            print('You must specify option type as first argument and/or Invalid option type')
        except Exception as e:
            print(e)

    def plot_letter(self,
                    LETTER=None,
                    STRIKE=None,
                    otype='c'
                    ):
        try:
            assert LETTER in ['delta','gamma', 'theta','kappa','rho']
            data ={}
            S, T = [], []
            strike = STRIKE if STRIKE else np.round(self.__underlying_price)

            if self.lastGreeks['strike'] and self.lastGreeks['strike'] == strike:
                data, S, T = self.lastGreeks['data'], self.lastGreeks['S'], self.lastGreeks['T']
            else:
                self.data_aggregate_greeks(strike=strike)
                data, S, T = self.lastGreeks['data'], self.lastGreeks['S'], self.lastGreeks['T']

            print('Plotting {} with Strike {} for {} - {}\n'.format(LETTER, strike, self.SYMBOL, otype))

            Z = data[otype][LETTER]
            fig = plt.figure(figsize=(20, 11))
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(40, 290)
            ax.plot_wireframe(S, T, Z, rstride=1, cstride=1)
            ax.plot_surface(S, T, Z, facecolors=cm.jet(Z), linewidth=0.001, rstride=1, cstride=1, alpha=0.75)
            ax.set_zlim3d(0, Z.max())
            ax.set_xlabel('Stock Price')
            ax.set_ylabel('Time to Expiration')
            ax.set_zlabel(LETTER)
            m = cm.ScalarMappable(cmap=cm.jet)
            m.set_array(Z)
            cbarDelta = plt.colorbar(m)

        except AssertionError:
            raise DataFormatError('Invalid Letter')
        except ValueError:
            raise DataFormatError('Invalid Calculations for {} :: Something went wrong on our end :$'.format(LETTER))
if __name__ == '__main__':

    OL = OptionLib()
    OL.data_building_core()
    OL.data_aggregate_greeks()
    OL.PLOT_LETTER('theta')