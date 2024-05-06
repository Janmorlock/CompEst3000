import numpy as np
import math

from lib.Model import CustModel

class EKF:
    """
    Extended Kalman Filter class used to estimate the states (abundance of P. putida, E.coli and fluorescent protein) of the system.

    Attributes
    ----------
    model : CustModel
        Model that is used to predict and update the state estimate.
    est : np.ndarray, dim: (num_states,)
        The mean of the previous state estimate. The order of states is
        given by x = [e, p, fp, fl_ofs].
    var : np.ndarray, dim: (num_states, num_states)
        The covariance of the previous state estimate. The order of states
        is given by x = [e, p, fp, fl_ofs].
    dev_ind : int
        The index of the reactor, i.e. the integer in 'M0', 'M1', ...
    update : bool
        Whether to perform the measurement update step.
    e_ofs : float
        The reactor specific offset for the fluorescence measurements.
    e_fac : float
        The reactor specific factor for the fluorescence measurements.
    """
    def __init__(self, model: CustModel = CustModel(), dev_ind: int = 0, update: bool = True):
        """
        Initialize the estimator. Sets the mean and covariance of the initial
        estimate.

        Parameters
        ----------
        model : CustModel, optional
            Model that is used to predict and update the state estimate. Default is CustModel().
        dev_ind : int, optional
            The index of the reactor. Default is 0.
        update : bool, optional
            Whether to perform the measurement update step. Default is True.
        """
        self.model = model
        self.est = {  # Initial state
            'e': model.parameters['od_init']*model.parameters['e_rel_init'],
            'p': model.parameters['od_init']*(1-model.parameters['e_rel_init']),
            'fp': model.parameters['fp_init']
        }
        self.var = np.diag([model.parameters['sigma_e_init']**2, model.parameters['sigma_p_init']**2, model.parameters['sigma_fp_init']**2])
        self.dev_ind = dev_ind
        self.update = update
        self.e_ofs = 0
        self.e_fac = 1

        # Helper variables
        self.time_prev = -1
        self.time_lst = []
        self.gr_lst = []
        self.od_lst = []
        self.fp_lst = []
        self.u_prev = np.array([29, False])

        self.p_est_fl = 0
        self.p_est_od = 0

        self.p_est_od_res = 0
        self.p_est_fl_res = 0

        self.gr_avg = np.zeros(3)


    def set_r_coeff(self, m_key):
        """
        Set the reactor specific offset and factor for the fluorescence measurements.
        """
        self.e_ofs = self.model.parameters['e_ofs'][m_key]
        self.e_fac = self.model.parameters['e_fac'][m_key]

    def estimate(self, time: float, u: np.ndarray, y: np.ndarray = np.zeros(0)):
        """
        Update system state and variance estimate by performing prediction and measurement update step with the Extended Kalman Filter.

        Parameters
        ----------
        time : float
            The time in s at which the measurement is obtained.
        u : np.ndarray, dim: (num_inputs,)
            The next input u = [temp, dilute] to the system.
        y : np.ndarray, dim: (num_outputs,), optional
            The measurement of the system. The order of outputs is given by y = [od, fl].
            Will be ignored when self.update set to False.

        Returns
        ----------
        None
        """
        # Prediction
        if self.time_prev >= 0: # Skip on first time step
            dt = time - self.time_prev
            self.est, self.var = self.model.predict(self.est, self.var, self.u_prev, dt)

        # Measurement Update
        if self.update:
            # Normalize fluorescent measurements
            y[1] = (y[1] - self.e_ofs)/self.e_fac
            # Store time, od and fl for later measurement update
            self.time_lst.append(time/3600)
            self.gr_lst.append(self.model.gr.copy())
            self.od_lst.append(y[0])
            self.fp_lst.append(y[1] - self.model.parameters['od_fac']*(self.est['e'] + self.est['p']))
            self.p_est_od = 0
            self.p_est_fl = 0
            self.p_est_od_res = 0
            self.p_est_fl_res = 0
            self.gr_avg = np.zeros(3)
            if u[1]:
                if not self.u_prev[1] and len(self.gr_lst) > 4: # Require at least 5 measurements to minimize noise fitting
                    # calculate self.p_est_od and self.p_est_fl just before dilution
                    self.gr_avg = np.mean(self.gr_lst[:-1], axis = 0) # exclude current growth rate as it did not influence the curvature
                    self.time_lst = np.array(self.time_lst) - self.time_lst[-1]

                    A_od = np.vstack(np.exp(self.gr_avg[1]*self.time_lst) - np.exp(self.gr_avg[0]*self.time_lst))
                    b_od = self.od_lst - self.od_lst[-1]*np.exp(self.gr_avg[0]*self.time_lst)
                    [self.p_est_od], [self.p_est_od_res] = np.linalg.lstsq(A_od, b_od, rcond = None)[0:2]
                    self.p_est_od_res = self.p_est_od_res/(len(b_od)*np.var(b_od))
                    if self.p_est_od < 0:
                        self.p_est_od = 0
                    if self.p_est_od >= y[0]:
                        self.p_est_od = 0

                    A_fl = np.vstack(self.gr_avg[2]/self.gr_avg[1]*(np.exp(self.gr_avg[1]*self.time_lst) - 1))
                    b_fl = self.fp_lst - self.fp_lst[-1]
                    [self.p_est_fl], [self.p_est_fl_res] = np.linalg.lstsq(A_fl, b_fl, rcond = None)[0:2]
                    self.p_est_fl_res = self.p_est_fl_res/(len(b_fl)*np.var(b_fl))
                    if self.p_est_fl < 0:
                        self.p_est_fl = 0
                    if self.p_est_fl >= y[0]:
                        self.p_est_fl = 0
                self.time_lst, self.gr_lst, self.od_lst, self.fp_lst = [], [], [], []
            if abs(self.est['e'] + self.est['p'] - y[0]) > 0.3:
                print('WARNING: od measurement far away from estimation [{}] [{}:{}]'.format(self.dev_ind, math.floor(time/3600), math.floor((time/3600-math.floor(time/3600))*60)))
            self.est, self.var = self.model.update(self.est, self.var, y, self.p_est_od, self.p_est_fl, self.p_est_od_res, self.p_est_fl_res, self.gr_avg, self.u_prev[0])

        self.time_prev = time
        self.u_prev = u