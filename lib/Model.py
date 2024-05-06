import numpy as np
from typing import Tuple

from config.params import Params

# initialize
class CustModel:
    """
    Custom Model class used to predict and update the state estimate.

    Attributes
    ----------
    parameters : dict
        Dictionary containing the model parameters.
    dithered : bool
        Whether to use dithered dilution.
    gr : np.ndarray, dim: (3,)
        The steady state growth/production rates of the bacteria/Pyoverdine.
    lp_fac : np.ndarray, dim: (2,)
        The low pass filter factor for the bacteria growth rates.
    gr_fp_delayed : np.ndarray, dim: (60*pr_fp_delay_min,)
        Array containing the delayed production rate.
    """
    def __init__(self):
        """
        Initialize the model. Sets the variables and calculates the steady state
        growth rates.

        Parameters
        ----------
        None
        """
        self.parameters = Params().default

        self.dithered = False
        self.gr = np.array([self.getSteadyStateGrowthRates(self.parameters['temp_pre_e'])[0],
                            self.getSteadyStateGrowthRates(self.parameters['temp_pre_p'])[1],
                            self.getSteadyStateGrowthRates(self.parameters['temp_pre_p'])[2]])
        self.lp_fac = np.exp(-self.parameters['ts']/3600/np.array(self.parameters['lp_ht']))
        self.gr_fp_delayed = np.full(60*self.parameters['pr_fp_delay_min'], self.gr[2], dtype=float)

        # Static helper variables
        self.ts = self.parameters['ts']
        self.ts_h = self.ts/3600

        self.Q = np.diag([self.parameters['sigma_e']**2, self.parameters['sigma_p']**2, self.parameters['sigma_fp']**2])
        self.Q_dil = np.diag([self.parameters['sigma_e_dil']**2, self.parameters['sigma_p_dil']**2, self.parameters['sigma_fp_dil']**2])
        self.Q_dil_dit = np.diag([self.parameters['sigma_e_dil_dit']**2, self.parameters['sigma_p_dil_dit']**2, self.parameters['sigma_fp_dil_dit']**2])

        self.M = np.eye(5)

    def dilute(self, x_prev: np.ndarray, p_prev: np.ndarray):
        '''
        Continuous Dilution

        Similar to the chiBio, this model checks every iteration if dilution is required.
        '''
        x_dil = x_prev.copy()
        p_dil = p_prev.copy()

        # Dilution
        if x_prev[0] + x_prev[1] > self.parameters['od_setpoint']:
            if x_prev[0] + x_prev[1] < 1.1*self.parameters['od_setpoint']:
                x_dil *= (2*self.parameters['od_setpoint']/(x_prev[0] + x_prev[1]) - 1)
            else: # Prevent too extensive bounce back from high od values
                print('diluting differently')
                x_dil *= self.parameters['od_setpoint']/(x_prev[0] + x_prev[1])
            den = (x_prev[0] + x_prev[1])**2
            # Covariance estimation after dilution
            A_dil = np.array([[x_prev[1]/den-1, -x_prev[0]/den, 0],
                                    [-x_prev[1]/den, x_prev[0]/den-1, 0],
                                    [-x_prev[2]/den, -x_prev[2]/den, 1/(x_prev[0]+x_prev[1]) - 1]])
            L_dil = np.diag(x_prev)
            p_dil = A_dil @ p_prev @ A_dil.T + L_dil @ self.Q_dil @ L_dil.T
        return x_dil, p_dil

    def dilute_dithered(self, x_prev: np.ndarray, p_prev: np.ndarray):
        '''
        Dithered Dilution

        This model tries to dilute the same amount as the chiBio during dithered dilution.
        '''
        dil_r = self.parameters['dil_rate']
        od = x_prev[0] + x_prev[1]
        # Dilution
        x_dil = x_prev * (1 - dil_r/od)
        # Covariance estimation after dilution
        A_dil = np.array([[1 - dil_r*x_prev[1]/od**2, dil_r*x_prev[0]/od**2, 0],
                            [dil_r*x_prev[1]/od**2, 1 - dil_r*x_prev[0]/od**2, 0],
                            [dil_r*x_prev[2]/od**2, dil_r*x_prev[2]/od**2, 1 - dil_r/od]])
        L_dil = np.diag(x_prev)
        p_dil = A_dil @ p_prev @ A_dil.T + L_dil @ self.Q_dil_dit @ L_dil.T
        return x_dil, p_dil
    
    def predict(self, x_prev_dic: dict, p_prev: np.ndarray, u: np.ndarray, dt: float) -> Tuple[dict, np.ndarray]:
        """
        Prediction Step

        Dilutes if needed and predicts states and their variance after dt seconds.
        To calculate the variance, the Jacobian of the system model is calculated.

        Parameters
        ----------
        x_prev_dic : dict
            Dictionary containing the previous states.
        p_prev : np.ndarray
            The previous state variance.
        u : np.ndarray, dim: (2,)
            The input to the system.
        dt : float
            The time in seconds to predict.
        
        Returns
        ----------
        dict
            Dictionary containing the predicted states.
        np.ndarray
            The predicted state variance.
        """
        x_prev = np.fromiter(x_prev_dic.values(),dtype=float)
        x_pred = x_prev.copy()
        p_pred = p_prev.copy()

        # Dilute if needed
        if self.dithered:
            if u[1]:
                x_pred, p_pred = self.dilute_dithered(x_prev, p_prev)
        else:
            x_pred, p_pred = self.dilute(x_prev, p_prev)

        # Approximate abundance and their varaince after dt seconds of growth
        gr_ss = self.getSteadyStateGrowthRates(u[0])
        for i in range(round(dt/self.ts)):
            # Get growth rates
            self.updateGrowthRates(gr_ss)
            # Jacobian of growth model
            A = np.array([[1 + self.ts_h*self.gr[0], 0, 0],
                      [0, 1 + self.ts_h*self.gr[1], 0],
                      [0, self.ts_h*self.gr[2], 1]])
            L = np.diag([self.ts_h*x_pred[0], self.ts_h*x_pred[1], self.ts_h*x_pred[1]])

            # Abundance after Ts seconds
            # Euler forward (not significantly less accurate than Runge - Kutta 4th order)
            x_pred = x_pred + self.ts_h*self.gr*np.array([x_pred[0], x_pred[1], x_pred[1]])

            p_pred = A @ p_pred @ A.T + L @ self.Q @ L.T

        return dict(zip(x_prev_dic, x_pred)), p_pred
    
    def update(self, x_pred_dic: dict, p_pred: np.ndarray, y: np.ndarray, p_est_od: float, p_est_fl: float, p_est_od_res: float, p_est_fl_res, gr_avg: np.ndarray, temp: float) -> Tuple[dict, np.ndarray]:
        '''
        Update Step

        Updates the states and their variance after a measurement y.

        Parameters
        ----------
        x_pred_dic : dict
            Dictionary containing the predicted states.
        p_pred : np.ndarray
            The predicted state variance.
        y : np.ndarray, dim: (2,)
            The measurement of the system.
        p_est_od : float
            The estimated P. putida abundance based on the optical density curvature. If zero, the estimate is not used.
        p_est_fl : float
            The estimated P. putida abundance based on the fluorescence curvature. If zero, the estimate is not used.
        p_est_od_res : float
            The residual of the least squares solution for the optical density curvature.
        p_est_fl_res : float
            The residual of the least squares solution for the fluorescence curvature.
        gr_avg : np.ndarray, dim: (3,)
            The average growth rates in the production phase.
        temp : float
            The current temperature of the system.

        Returns
        ----------
        dict
            Dictionary containing the updated states.
        np.ndarray
            The updated state variance.
        '''
        x_pred = np.fromiter(x_pred_dic.values(),dtype=float)
        od = x_pred[0] + x_pred[1]

        xm = x_pred.copy()
        Pm = p_pred.copy()

        H = np.array([[]])
        y_est = np.array([])
        y_new = np.array([])
        R = np.diag([])
        
        m = 0

        if self.parameters['od_update']:
            xm[0] *= y[0]/od
            xm[1] *= y[0]/od
            H = np.array([[1, 1, 0]])
            y_est = np.array([y[0]])
            y_new = np.array([y[0]])
            R = np.diag([self.parameters['sigma_od']**2])
            m += 1
        if y[1] and self.parameters['fl_update']:
            H = np.reshape(np.append(H, [self.parameters['od_fac'], self.parameters['od_fac'], 1]),(-1,3)) # [-x_pred[2]/(od+self.parameters['od_ofs'])**2, -x_pred[2]/(od+self.parameters['od_ofs'])**2, 1/(od+self.parameters['od_ofs'])]
            y_est = np.append(y_est, x_pred[2] + self.parameters['od_fac']*od) # x_pred[2]/(od+self.parameters['od_ofs']) + self.parameters['fl_ofs'][r_ind]
            y_new = np.append(y_new, y[1])
            R = np.diag(np.append(np.diag(R), [(self.parameters['sigma_fl']
                                               + np.exp(-abs(temp - 29.5)/self.parameters['fl_gr_temp2_sigma_decay'])*self.parameters['fl_temp2_prox_sigma_max'])
                                               **2]))
            m += 1
        if p_est_od and self.parameters['od_gr_update']:
            H = np.reshape(np.append(H, [[1, 0, 0], [0, 1, 0]]),(-1,3))
            y_est = np.append(y_est, [x_pred[0], x_pred[1]])
            y_new = np.append(y_new, [y[0] - p_est_od, p_est_od])
            # Increase the variance of the measurement according to the residual of the lsq solution and the proximity of both growth rates
            R = np.diag(np.append(np.diag(R),
                                  [(self.parameters['sigma_od_gr']
                                    + p_est_od_res*self.parameters['gr_res_sigma']*x_pred[0]
                                    + np.exp(-abs(gr_avg[1] - gr_avg[0])/self.parameters['od_gr_prox_sigma_decay'])*self.parameters['od_gr_prox_sigma_max'])**2,
                                   (self.parameters['sigma_od_gr']
                                    + p_est_od_res*self.parameters['gr_res_sigma']*x_pred[1]
                                    + np.exp(-abs(gr_avg[1] - gr_avg[0])/self.parameters['od_gr_prox_sigma_decay'])*self.parameters['od_gr_prox_sigma_max'])**2]))
            m += 2
        if p_est_fl and self.parameters['fl_gr_update']:
            H = np.reshape(np.append(H, [0, 1, 0]),(-1,3))
            y_est = np.append(y_est, x_pred[1])
            y_new = np.append(y_new, p_est_fl)
            # Increase the variance of the measurement according to the residual of the lsq solution and the proximity of 35ºC (strong change in production rate)
            R = np.diag(np.append(np.diag(R), (self.parameters['sigma_fl_gr']
                                               + p_est_fl_res*self.parameters['gr_res_sigma']*x_pred[1]
                                               + np.exp(-abs(temp - 35.5)/self.parameters['fl_gr_temp_sigma_decay'])*self.parameters['fl_gr_temp_prox_sigma_max']
                                               + np.exp(-abs(temp - 29)/self.parameters['fl_gr_temp2_sigma_decay'])*self.parameters['fl_gr_temp2_prox_sigma_max']
                                               )**2))
            m += 1
        
        # K = np.linalg.solve(H @ p_pred.T @ H.T + self.M @ self.parameters['r'].T @ self.M.T, H @ p_pred.T).T
        if m > 0:
            K = p_pred @ H.T @ np.linalg.inv(H @ p_pred @ H.T + self.M[:m,:m] @ R @ self.M[:m,:m].T)
            if self.parameters['fl_update'] and not self.parameters['fl_gr_update']:
                xm[:2] += (K @ (y_new - y_est))[:2]
            else:
                xm += (K @ (y_new - y_est))
            xm[np.isnan(xm)] = x_pred[np.isnan(xm)]
            Pm = (np.eye(3) - K @ H) @ p_pred

        # Constrain states to be positive
        if xm[0] < 0:
            xm[1] += xm[0]
            xm[0] = 1e-4
        if xm[1] < 0:
            xm[0] += xm[1]
            xm[1] = 1e-4
        if xm[2] < 0:
            xm[2] = 1e-4

        return dict(zip(x_pred_dic, xm)), Pm
    
    def getSteadyStateGrowthRates(self, temp: float) -> np.ndarray:
        """
        Given the current temperature, return the corresponding steady state growth and production rates
        """
        temp = min(36, temp)
        temp = max(29, temp)
        gr_e_ss = self.parameters['gr_e'][0]*temp + self.parameters['gr_e'][1]
        gr_p_ss = self.parameters['gr_p'][0]*temp**2 + self.parameters['gr_p'][1]*temp + self.parameters['gr_p'][2]
        pr_f_ss = 0
        for r in range(len(self.parameters['gr_fp'])):
            pr_f_ss += self.parameters['gr_fp'][r]*(temp-32.5)**(len(self.parameters['gr_fp']) - 1 - r)
        pr_f_ss = max(self.parameters['min_gr_fp'],pr_f_ss)

        return np.array([gr_e_ss, gr_p_ss, pr_f_ss])

    def updateGrowthRates(self, gr_ss: np.ndarray):
        """
        Filter the steady state steady state growth and production rates according to the dynamic model
        """
        # Low pass filter bacteria growth rates
        self.gr[0:2] = self.lp_fac*self.gr[0:2] + (1-self.lp_fac)*gr_ss[0:2]

        # Add damped oscillation to the production rate
        self.gr[2] = self.gr[2] + self.parameters['gr_fp_k']*(gr_ss[2] - self.gr_fp_delayed[0])
        self.gr[2] = max(self.parameters['min_gr_fp'],self.gr[2])
        self.gr_fp_delayed = np.append(self.gr_fp_delayed[1:],self.gr[2])