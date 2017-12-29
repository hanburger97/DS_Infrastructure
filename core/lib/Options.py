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

    """
    # LEGACY METHOD - TO BE DELETED AFTER ENSURING BACKWARD COMPATIBILITY
    def IV(self,
           option_type,
           option_price,
           underlying_price,
           strike,
           time,
           riskfreerate=None,
           dividendrate=None
           ):


        otype, oprice, s, k, t, r, q = option_type, option_price, underlying_price, strike, time, riskfreerate, dividendrate
        q = q if q else self.dividend_rate
        r = r if r else self.risk_free_rate

        epsilon = 0.00001
        upper_sigma = 500.0
        max_sigma = 500.0
        min_sigma = 0.0001
        lower_sigma = 0.0001
        iteration = 0


        # Using a Bisection Algorithm to find the Implied Sigma
        while True:
            iteration += 1
            mid_sigma = (upper_sigma + lower_sigma) / 2.0
            price = self.BSM(otype, mid_sigma, s, k, r, t, q)

            if otype == 'c':

                lower_price = self.BSM(otype, lower_sigma, s, k, r, t, q)
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

            elif otype == 'p':
                upper_price = self.BSM(otype, upper_sigma, s, k, r, t, q)

                if (upper_price - oprice) * (price - oprice) > 0:
                    upper_sigma = mid_sigma
                else:
                    lower_sigma = mid_sigma
                    #             print("mid_vol=%f" %mid_vol)
                    #             print("upper_price=%f" %upper_price)
                if abs(price - oprice) < epsilon: break
                if iteration > 100: break

        return mid_sigma

    """

    #########################################################################################################
    #                                       Helpers and Tools                                               #
    #########################################################################################################

    """
        # LEGACY METHOD - TO BE DELETED AFTER ENSURING BACKWARD COMPATIBILITY

    def IV_TT(self,
            option_type,
            expiry_index,
            opt=None
            ):

        opt = opt if opt else self.opt

        expiry = opt.expiry_dates[expiry_index]
        if option_type == 'c':
            data = opt.get_call_data(expiry=expiry)
        elif option_type == 'p':
            data = opt.get_put_data(expiry=expiry)

        s = opt.underlying_price  # data_call['Underlying_Price']  undelying price
        expiry = data.index.get_level_values('Expiry')[0]  # get the expiry
        current_date = opt.quote_time  # current_date = datetime.datetime.now() # get the current date
        time_to_expire = float((expiry - current_date).days) / 365  # compute time to expiration
        premium = (data['Ask'] + data['Bid']) / 2  # option premium
        strike = list(data.index.get_level_values('Strike'))  # get the strike price
        IV = []

        for i in range(len(data)):
            IV.append(self.IV(option_type, premium.values[i], s, strike[i], time_to_expire))

        return data, strike, IV

    """

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



    def IVP(self,
            riskfreerate=None,
            dividendRate=None,
            opt=None
            ):

        """
        :return: All the Implied Vol for all strikes and expiration for Puts
        """
        opt = opt if opt else self.opt
        r = riskfreerate if riskfreerate else self.risk_free_rate
        q = dividendRate if dividendRate else self.dividend_rate

        current_date = opt.quote_time.date()  ## get the current date
        expiry_dates = [date for date in opt.expiry_dates if date > current_date]
        s = opt.underlying_price  # undelying price
        num_expiry = len(expiry_dates)
        res = []

        for ei in range(num_expiry):
            expiry = expiry_dates[ei]
            data = opt.get_put_data(expiry=expiry)
            toe = ((expiry - current_date).days)
            premium = (data['Ask'] + data['Bid']) / 2.0
            strikes = data.index.get_level_values('Strike')
            datalen = len(data)
            for j in range(datalen):
                res.append([toe, strikes[j], data['IV'].values[j]])
        self.lastIVP = res
        return res


    def IVC(self,
            riskfreerate=None,
            dividendRate=None,
            opt=None
            ):

        """
        :return: All the Implied Vol for all strikes and expiration for Calls
        """
        opt = opt if opt else self.opt
        r = riskfreerate if riskfreerate else self.risk_free_rate
        q = dividendRate if dividendRate else self.dividend_rate

        current_date = opt.quote_time.date()  ## get the current date
        expiry_dates = [date for date in opt.expiry_dates if date > current_date]
        s = opt.underlying_price  # undelying price
        num_expiry = len(expiry_dates)
        res = []

        for ei in range(num_expiry):
            expiry = expiry_dates[ei]
            data = opt.get_call_data(expiry=expiry)
            toe = (expiry - current_date).days
            premium = (data['Ask'] + data['Bid']) / 2.0
            strikes = data.index.get_level_values('Strike')

            for j in range(len(data)):
                res.append([toe, strikes[j], data['IV'].values[j]])
        self.lastIVC = res
        return res



    #########################################################################################################
    #                                       Plotting                                                        #
    #########################################################################################################

    """
        # LEGACY METHOD - TO BE DELETED AFTER ENSURING BACKWARD COMPATIBILITY

    def plot_IV_d(self,
                option_type,
                expiry_index,
                opt=None
                ):

        opt = opt if opt else self.opt

        data, strike, IV = self.IV_TT(option_type, expiry_index, opt)


        plt.figure(figsize=(16, 7))
        a = plt.scatter(strike, IV, c='r', label="IV by solving BSM")
        b = plt.scatter(strike, data['IV'], c='b', label="IV from Yahoo Finance")

        plt.grid()
        plt.xlabel('strike')


        if option_type == 'c':
            plt.ylabel('Implied Volatility for call option')
            plt.legend((a, b), ("IV(call) by solving BSM", "IV(call) from Yahooe"))
        elif option_type == 'p':
            plt.ylabel('Implied Volatility for put options')
            plt.legend((a, b), ("IV(put) by solving BSM", "IV(put) from Yahoo"))
    """

    def plot_smile(self,
                   expiry_index,
                   opt=None
                   ):
        """

        :param k_call: Array of strike prices for call
        :param IV_call: Corresponding IV for call
        :param k_put: Array of strike prices for puts
        :param IV_put: Corresponding IV for puts
        :return:
        """
        opt = opt if opt else self.opt

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
            plotdata = []
            color = None
            if option_type == 'c':
                plotdata = self.lastIVC if self.lastIVC else self.IVC(opt=opt)
                color = 'lime'
            elif option_type == 'p':
                plotdata = self.lastIVP if self.lastIVP else self.IVP(opt=opt)
                color = 'red'

            xaxis = [plotdata[i][0] for i in range(len(plotdata))]
            yaxis = [plotdata[i][1] for i in range(len(plotdata))]
            zaxis = [plotdata[i][2] for i in range(len(plotdata))]

            fig = plt.figure(figsize=(20, 12))
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init()
            ax.scatter(xaxis, yaxis, zaxis, c=color)
            plt.xlabel("Time to Expiration (days)")
            plt.ylabel("Strikes")
            plt.title("Implied Volatility")

            f2 = plt.figure(figsize=(20, 12))
            ax2 = f2.add_subplot(111, projection='3d')
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
