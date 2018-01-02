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
"""
 def IVP(self,
            riskfreerate=None,
            dividendRate=None,
            opt=None
            ):


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
"""

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