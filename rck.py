import numpy as np
import torch
from tqdm import tqdm


####################################################
##################### RCK Model ####################
####################################################


class RCK:
    def __init__(
        self,
        alpha,
        delta,
        n,
        rho,
        T,
        c0,
        k0,
        u0=1,
        deltaT=1,
        n_scenarios=3,
        noise_std=0.01,
        flex = 0.01,
        dt=0.1,
        model=None,
        scalerx=None,
        scalery=None,
        seed = 0
    ):
        # Problem Parameters
        self.alpha = alpha
        self.delta = delta
        self.n = n
        self.T = T
        self.rho = rho
        self.u0 = u0
        self.deltaT = deltaT
        self.dt = dt
        self.flex = flex
        self.u0 = 1

        # Model Variables
        self.c = np.append(c0, np.zeros(T - 1))
        self.k = np.append(k0, np.zeros(T - 1))
        self.forecasted_k = np.append(k0, np.zeros(T - 1))
        self.time_horizon = 1
        self.number_possible_scenarios = n_scenarios  # Number of Different Value of k
        self.noise_std = noise_std

        # iAgent
        self.model = model
        self.scalerx = scalerx
        self.scalery = scalery
        np.random.seed(seed)

    def equilibrium_capital_growth(self, t):
        dkdt = self.k[t] ** self.alpha - (self.delta + self.n) * self.k[t] - self.c[t]
        
        return dkdt * self.dt

    def equilibrium_consumption_growth(self, t):
        dcdt = (
            self.alpha * self.k[t] ** (self.alpha - 1) - self.delta - self.rho
        ) * self.c[t]

        return dcdt * self.dt

    def run_equilibrium_solution(self):
        for t in np.arange(1, self.T):
            self.k[t] = self.k[t - 1] + self.equilibrium_capital_growth(t - 1)
            self.c[t] = self.c[t - 1] + self.equilibrium_consumption_growth(t - 1)

        # return self.k, self.c

    def utility_equation(self, c: float, t: float):
        return np.exp(-(self.rho - self.n) * t) * self.u0 * np.log(np.maximum(1e-8, c))

    def consumption_equation(self, km1: float, k0: float):
        km1[km1 < 0] = np.nan
        return km1**self.alpha - (self.n + self.delta) * km1 - (k0 - km1) / self.dt

    def obtain_forecasted_capital_vectorized(self, c, k, t):
        X = [x for x in zip(c, k)]
        X = self.scalerx.transform(np.array(X))
        X = torch.tensor(X, dtype=torch.float32)
        Y = self.model(X)
        Y = self.scalery.inverse_transform(Y.detach().numpy())
        # flatten Y
        Y = Y.flatten()

        return Y

    def obtain_forecasted_capital(self, c, k, t):
        """Return kforecasted for c and k.
        Parameters
        ----------
        c : float
            Consumption.
        k : float
            Capital.
        model : torch.nn.Module
            Trained neural network model.
        scalerx : sklearn.preprocessing.MinMaxScaler
            Scaler for x.
        scalery : sklearn.preprocessing.MinMaxScaler
            Scaler for y.
        Returns
        -------
        kforecasted : float
            Forecasted capital.
        """
        # Here the NN should take as input only two value of c_t-1 and k_t-1 and return the prediction of k_t
        # The noise is then summed up to build the possible scenarios
        # The time is not relevant and thus it's not usefull to keep track here

        # scale c and k
        X = [c, k]
        X = self.scalerx.transform(np.array(X).reshape(1, -1))
        X = torch.tensor(X, dtype=torch.float32)
        Y = self.model(X)
        Y = self.scalery.inverse_transform(Y.detach().numpy())

        self.forecasted_k[t] = Y[0][0]

        return Y[0][0]

    def EXACT_obtain_forecasted_capital(self, c, k, t):
        num_iterations = 3
        num_samples = len(c)

        k_eq = np.zeros((num_samples, num_iterations))
        c_eq = np.zeros((num_samples, num_iterations))

        k_eq[:, 0] = k
        c_eq[:, 0] = c

        alpha_minus_1 = self.alpha - 1
        delta_plus_n = self.delta + self.n

        for j in range(1, num_iterations):
            dcdt_eq = (
                self.alpha * k_eq[:, j - 1] ** alpha_minus_1 - self.delta - self.rho
            ) * c_eq[:, j - 1]
            dkdt_eq = (
                k_eq[:, j - 1] ** self.alpha
                - delta_plus_n * k_eq[:, j - 1]
                - c_eq[:, j - 1]
            )

            c_eq[:, j] = c_eq[:, j - 1] + dcdt_eq * self.dt
            k_eq[:, j] = k_eq[:, j - 1] + dkdt_eq * self.dt

        return k_eq[:, -1]

    def create_new_array(self, c):
        return np.linspace(c*(1 - self.flex), c*(1 + self.flex), self.number_possible_scenarios).T

    def budget_constrain(self, c, a, r, w):
        return (a + self.dt * (r * a + w - c)).T

    def utility_equation(self, c, t):
            return np.exp(-(self.rho - self.n) * t) * self.u0 * np.log(np.maximum(1e-8, c))

    def optimalc(self, t):

        k_eq = np.append(self.k[t-1], np.zeros(self.deltaT))
        c_eq = np.append(self.c[t-1], np.zeros(self.deltaT))

        for j in np.arange(1, self.deltaT):
            dkdt = k_eq[j] ** self.alpha - (self.delta + self.n) * k_eq[j] - c_eq[j]
            dcdt = (self.alpha * k_eq[j] ** (self.alpha - 1) - self.delta - self.rho) * c_eq[j]
            k_eq[j] = (k_eq[j - 1] + dkdt * self.dt) * (1 + np.random.normal(0, self.noise_std))
            c_eq[j] = (c_eq[j - 1] + dcdt * self.dt) * (1 + np.random.normal(0, self.noise_std))

        
        c = np.zeros((self.deltaT + 1, self.number_possible_scenarios**self.deltaT))
        a = np.zeros((self.deltaT + 1, self.number_possible_scenarios**self.deltaT))
        u = np.zeros((self.deltaT + 1, self.number_possible_scenarios**self.deltaT))

        a[0, :] = self.k[t-1]
        c[0, :] = self.c[t-1]
        
        r = self.alpha * k_eq**(self.alpha -1)
        w = (1 - self.alpha) * k_eq**self.alpha

        for j in np.arange(1, self.deltaT + 1): 
            k_range = np.arange(0, self.number_possible_scenarios ** self.deltaT, step=self.number_possible_scenarios ** (self.deltaT - j + 1)) 
            c[j, :] = np.repeat(self.create_new_array(c[0, k_range]).flatten(), self.number_possible_scenarios ** (self.deltaT - j))
            a[j, :] = np.repeat(self.budget_constrain(c[j - 1, k_range], a[j-1, k_range], r[j-1], w[j-1]).flatten(), self.number_possible_scenarios ** (self.deltaT - j + 1))
            u[j, :] = u[j-1,:] + self.utility_equation(c[j, :], (j-1) * self.dt)

        print('a   ', a)
        print('c   ', c)
        allowed_consumption = c[-1, :] < a[-1, :]
        

        return c[1, allowed_consumption][np.argmax(u[-1, allowed_consumption])]

    def capital_growth(self, t):
        dkdt = (
            self.k[t - 1] ** self.alpha
            - (self.delta + self.n) * self.k[t - 1]
            - self.c[t - 1]
        )

        return dkdt * self.dt

    def run_simulation(self):
        for t in tqdm(np.arange(1, self.T)):
            self.k[t] = self.k[t - 1] + self.capital_growth(t)
            optc = self.optimalc(t)
            if np.isnan(optc):
                return np.nan
            self.c[t] = optc
       
            
