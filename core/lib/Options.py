import pandas as pd
import numpy as np
from pandas_datareader.data import Options
import pandas_datareader.data as web
import datetime
import scipy.stats as stats
from scipy import interpolate
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import animation
from threading import Thread, ThreadError

from core.lib.Black_Scholes import Black_Scholes

############################################   Some Global Var         #################################################

SYNCFLAG = 0

############################################   Custom Error Classes    #################################################

class DataFormatError(Exception):

    def __init__(self, message):
        self.message = message

#############################################   Some Method Wrapper    #################################################


def autorefresh(func):

    def wrapper(object):
        try:
            worker = Thread(target=func, args=(object,))
            worker.setDaemon(True)
            worker.start()
            print('Data Auto-Refresh [ STARTED ] ')
        except ThreadError as te:
            print(te)
        except Exception as e:
            print(e)
    return wrapper

def waitsync(syncflag):
    def g(func):
        def wrapper(object):
            try:
                while syncflag == 1:
                    print('Data Sync [ IN PROGRESS ] ')
                    if syncflag == 0:
                        print('Data Sync [ COMPLETED ] ')
                        break
                func(object)
            except Exception as e:
                print(e)
        return wrapper
    return g


################################################      Main Class      ##################################################


class OptionLib:


    def __init__(self,
                 symbol='SPY',
                 dataprovider='yahoo',
                 riskfree=0.00,
                 dividendrate=0.00
                 ):

        self.symbol = symbol
        self.data_provider = dataprovider

        self.__oldsymbol = symbol
        self.__olddataprovider = dataprovider

        self.risk_free_rate = riskfree
        self.dividend_rate = dividendrate


        self.opt = None

        # each c or p array contains array of [ [ (T-t), strike, IV ] , ... ]
        self.IVs = {
            'c':[],
            'p':[]
        }

        self.lastIVP = None
        self.lastIVC = None

        self.__last_quote=None
        self.__underlying_price = None
        self.__data=None

        self.data_init()




    #########################################################################################################
    #                                   Data Building and Housekeeping                                      #
    #########################################################################################################

    def data_init(self):

        self.opt = Options(self.symbol, self.data_provider)
        self.__last_quote = self.opt.get_call_data().iloc[0]['Quote_Time'].to_pydatetime()
        self.__underlying_price = self.opt.underlying_price
        self.IVs = {'c': [], 'p': []}
        self.__data = self.data_building(r=self.risk_free_rate, q=self.dividend_rate)

    @waitsync(syncflag=SYNCFLAG)
    def data_refresh(self):

        # Check if symbol has been updated
        if not (self.symbol == self.__oldsymbol and self.data_provider == self.__olddataprovider):
            self.__oldsymbol, self.__olddataprovider = self.symbol, self.data_provider
            self.data_init()
        # Check if there is a new quote
        if not self.opt.get_call_data().iloc[0]['Quote_Time'].to_pydatetime() == self.__last_quote:
            self.__underlying_price = self.opt.underlying_price
            self.__data = self.data_building(r=self.risk_free_rate, q=self.dividend_rate)

            self.IVs = {'c': [],'p': []}
            self.data_aggregate_IV()

            self.__last_quote = self.opt.get_call_data().iloc[0]['Quote_Time'].to_pydatetime()

            #Legacy
            self.lastIVP = None
            self.lastIVC = None


    @autorefresh
    def data_auto_refresh(self):
        # This should be running in another thread
        while True:
            self.data_refresh()

    def explore_expiry(self, opt=None):
        #Legacy...
        opt = opt if opt else self.opt
        d ={'Expiry Dates':opt.expiry_dates}
        return pd.DataFrame(data=d)


    @property
    def data(self):
        # Making self.__data read only
        print('Underlying @ {:.2f} \n Last Option Quote: {}'.format(self.__underlying_price, self.__last_quote))
        df = self.__data.copy()
        return self.display_format(df)


    def display_format(self, df):
        # Reformatting greek letters with UTF-8 Encoding for Esthetics
        df['\u03C3'] = df['sigma']
        del df['sigma']
        df['\u0394'] = df['delta']
        del df['delta']
        df['\u0393'] = df['gamma']
        del df['gamma']
        df['\u0398'] = df['theta']
        del df['theta']
        df['\u039A'] = df['kappa']
        del df['kappa']
        df['\u03A1'] = df['rho']
        del df['rho'], df['toe']
        df['%'] = (df['%'] * 100)//100
        return df

    def data_building(self, r=None, q=None):

        try:
            assert self.opt
            global SYNCFLAG
            SYNCFLAG = 1
            rawdf = self.opt.get_all_data()
            rowLen = len(rawdf.index.values)
            r = r if r else self.risk_free_rate
            q = q if q else self.dividend_rate
            delta, gamma, theta, kappa, rho = [], [], [], [], []
            toes = []
            # O(n) Linear runtime
            #   For each row, compute Black-Scholes model calculations
            for i in range(rowLen):

                # Index tuple: (strike, timestamp, type, symbol)
                index_tuple = rawdf.index.values[i]
                row = rawdf.loc[index_tuple]
                otype = 'c' if index_tuple[2] == 'call' else 'p'

                # toe: time until expiration in years (floating  pt)
                exp = index_tuple[1].to_pydatetime()
                curr = row['Quote_Time'].to_pydatetime()
                toe = float((exp-curr).days)/365.0

                #initiate a black scholes object which also makes the necessary calculations
                #   then append to corresponding array to be zipped
                bso = Black_Scholes(
                    option_type=otype,
                    price=row['Underlying_Price'],
                    strike=index_tuple[0],
                    interest_rate=r,
                    dividend_yield=q,
                    volatility=row['IV'],
                    expiry=toe
                )
                delta.append(bso.delta)
                gamma.append(bso.gamma)
                theta.append(bso.theta)
                kappa.append(bso.kappa)
                rho.append(bso.rho)
                toes.append(toe)
                # Free em memory
                del bso

            # Check length before zipping in the Dataframe
            if not (len(delta) == len(gamma) ==
                    len(theta) == len(rho) ==
                    len(kappa) == rowLen
                    ): raise DataFormatError(
                        'Array of Greeks cannot be zipped since their length is not' +
                        ' compatible with DF Height'
                    )

            rawdf['delta'], rawdf['gamma'], rawdf['theta'], rawdf['kappa'], rawdf['rho'], rawdf['toe'] = \
                delta, gamma, theta, kappa, rho, toes

            # Housekleeping
            rawdf['%'], rawdf['sigma'] = rawdf['PctChg'], rawdf['IV']

            # Delete useless data from raw dataframe, still accessible through opt object
            del rawdf['IV'], rawdf['Chg'], rawdf['PctChg'], rawdf['Open_Int'], rawdf['Root'], rawdf['Underlying'], \
                rawdf['IsNonstandard'], rawdf['Quote_Time'], rawdf['Last_Trade_Date'], rawdf['Underlying_Price'], \
                rawdf['JSON']

            # Free some mem
            del kappa, delta, gamma, rho

            print('\nData Updated & Built for {} :: [ COMPLETE ] \nLatest quote: {}'.format(self.symbol, self.__last_quote))
            SYNCFLAG = 0

            return rawdf

        except AssertionError:
            print('You must first declare a symbol')

        except RuntimeWarning:
            print('Warning')

        except Exception as e:
            raise DataFormatError(e.args)






    #########################################################################################################
    #                                      Legacy Calculations...                                           #
    #########################################################################################################


    def BSM(self,
            option_type,
            sigma,
            underlying_price,
            strike,
            time,
            riskfreerate=None,
            dividendrate=None,
            ):

        """
        :param option_type: string 'c' for call and 'p' for puts
        :param sigma: stddev ratio (e.g. 1.0 := 100%)
        :param underlying_price: denoted in floating decimal
        :param strike: denoted in floating decimal
        :param time: nb of days??
        :param riskfreerate: percentage over decimal (0.01 := 1%)
        :param dividendrate:percentage over decimal (0.01 := 1%)
        :return: floating decimal of theoretical price of the option
        """


        return Black_Scholes.bsm_static_pricing(
            option_type=option_type,
            sigma=sigma,
            underlying_price=underlying_price,
            strike=strike,
            time=time,
            riskfreerate=riskfreerate,
            dividendrate=dividendrate
        )



    #########################################################################################################
    #                                       Helpers and Tools                                               #
    #########################################################################################################



    def data_aggregate_IV(self):
        """
        Aggregate Contract's sigma by call and puts then store them in [ Time to Expiration, Strike, Volatility ]
         format for later plotting
        """

        try:
            assert type(self.__data) == pd.DataFrame
            df = self.__data.copy()

            rowlen = len(df.index.values)

            for i in range(rowlen):
                index = df.index.values[i]
                row = df.loc[index]
                # Format : [ Time to Expiration, Strike, Volatility ]
                if index[2] == 'call':
                    self.IVs['c'].append([row['toe'], index[0], row['sigma']])
                elif index[2] == 'put':
                    self.IVs['p'].append([row['toe'], index[0], row['sigma']])

        except AssertionError:
            raise DataFormatError('Must input a pandas.DataFrame')







    #########################################################################################################
    #                                       Plotting                                                        #
    #########################################################################################################


    def plot_smile(self,
                   expiry_index,
                   ):
        """
        Plot the IV smile for both calls and puts per timestamp
        :return:
        """
        pass
        opt = None
        data_call, k_call, IV_call = self.IV_TT('c', expiry_index, opt)
        data_put, k_put, IV_put = self.IV_TT('p', expiry_index, opt)


        plt.figure(figsize=(16, 7))
        e = plt.scatter(k_call, IV_call, c='red', label="IV(call options)")
        f = plt.scatter(k_put, IV_put, c='white', label="IV(put options)")
        plt.xlabel('strike')
        plt.ylabel('Implied Volatility')
        plt.legend((e, f), ("IV (call options)", "IV (put options)"))


    def plot_surface(self,
                     option_type=None,
                     opt=None
                     ):
        try:
            assert option_type == 'c' or option_type =='p'
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

if __name__ == '__main__':


    olib = OptionLib()
