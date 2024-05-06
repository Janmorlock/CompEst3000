# Import the model and the estimator libraries
from lib.Model import CustModel
from lib.Estimator import EKF

# Import other packages
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    # Simulation parameters
    pre_time_m = 60
    timeEnd_m = 60 * 17
    alpha_temp = 0.8
    e_init = 0.35
    od_th = 0.53
    od_l_th = 0.46
    dithered = True
    noise = True
    test_var = 'single_est' # '' or 'single_est'
    single_est = {'od_update':      [True, True, True, True, True],
                    'fl_update':    [False, True, False, False, True],
                    'od_gr_update': [False, False, True, False, True],
                    'fl_gr_update': [False, False, False, True, True]}
    n_culumns = 1
    n_rows = 4
    pp_color = 'g'

    # Initialize plot
    matplotlib.style.use('default')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 8})
    plt.rcParams["mathtext.fontset"] = 'stix'
    fig, ax = plt.subplots(n_rows,n_culumns, height_ratios = [1,1,1,4], sharex='all')
    fig.set_figheight(7)
    fig.set_figwidth(7.5)

    # Run simulation in different configurations
    len_test = len(single_est['od_update'])
    for j in range(len_test):
        model_sim = CustModel()
        model_est = CustModel()
        model_sim.parameters['e_rel_init'] = e_init
        model_sim.parameters['temp_pre_e'] =  24
        model_sim.parameters['temp_pre_p'] =  24
        x0 = {'e': model_sim.parameters['od_init']*model_sim.parameters['e_rel_init'],
            'p': model_sim.parameters['od_init']*(1-model_sim.parameters['e_rel_init']),
            'fp': model_sim.parameters['fp_init']}
        p0 = np.diag([model_sim.parameters['sigma_e_init']**2, model_sim.parameters['sigma_p_init']**2, model_sim.parameters['sigma_fp_init']**2])
        x = x0
        p = p0
        model_sim.dithered = dithered
        model_est.dithered = dithered
        ekf = EKF(model_est, 0, True)
        ekf.u_prev = np.array([24, False])
        ekf.set_r_coeff('Faith')
        dil = False
        temp = 24
        temp_target = 0
        time_m = 0
        time_s = 0
        time_s_prev = 0
        if test_var == 'single_est':
            model_est.parameters['od_update'] = single_est['od_update'][j]
            model_est.parameters['fl_update'] = single_est['fl_update'][j]
            model_est.parameters['od_gr_update'] = single_est['od_gr_update'][j]
            model_est.parameters['fl_gr_update'] = single_est['fl_gr_update'][j]
        if j < len_test-1:
            if single_est['fl_update'][j]:
                ekf.model.parameters['sigma_fl'] = 1
                ekf.model.parameters['fl_temp2_prox_sigma_max'] = 0.0
                ekf.model.parameters['sigma_fp'] = 1e-4
                ekf.model.parameters['sigma_fp_dil_dit'] = 1e-4
            if single_est['od_gr_update'][j]:
                ekf.model.parameters['sigma_od_gr'] = 1e-2
                ekf.model.parameters['gr_res_sigma'] = 0
                ekf.model.parameters['od_gr_prox_sigma_max'] = 0
            if single_est['fl_gr_update'][j]:
                ekf.model.parameters['sigma_fl_gr'] = 1e-2
                ekf.model.parameters['gr_res_sigma'] = 0
                ekf.model.parameters['fl_gr_temp2_prox_sigma_max'] = 0.0
                ekf.model.parameters['fl_gr_temp_prox_sigma_max'] = 0.0
        
        # Initialize arrays for logging
        len = int(timeEnd_m)
        time_h_arr = np.empty(len)
        fl_arr = np.empty(len)
        fl_est_arr = np.empty(len)
        temp_arr = np.empty(len)
        temp_target_arr = np.empty(len)
        od_arr = np.empty(len)
        od_est_arr = np.empty(len)
        pp_rel_arr = np.empty(len)
        pp_rel_est_arr = np.empty(len)
        pp_rel_target_arr = np.full(len, -1, dtype=float)
        pp_rel_od_arr = np.empty(len)
        pp_rel_fl_arr = np.empty(len)
        while time_m < timeEnd_m:
            ### Simulate
            time_h = time_m/60
            u = np.array([temp, dil])
            dt = time_s - time_s_prev
            
            # Add deviations, noise to the system model dynamics
            if noise:
                model_sim.parameters['gr_e'][-1] = model_est.parameters['gr_e'][-1] - 0.05*model_est.getSteadyStateGrowthRates(temp)[0]
                # model_sim.parameters['gr_p'][-1] = model_est.parameters['gr_p'][-1] - 0.05*model_est.getSteadyStateGrowthRates(temp)[1]
                model_sim.parameters['gr_fp'][-1] = model_est.parameters['gr_fp'][-1] + np.sin(time_h/2*2*np.pi)*400*min(1,1-(temp-29)/6)
            
            # Get simulated states and measurements
            x, p = model_sim.predict(x, p, u, dt)
            od = x['e'] + x['p']
            fl = (x['fp'] + model_sim.parameters['od_fac']*od) * model_sim.parameters['e_fac']['Faith'] + model_sim.parameters['e_ofs']['Faith']
            
            # Add measurement noise
            od_meas = od
            if noise:
                od_meas = od + np.random.normal(0, model_sim.parameters['sigma_od_mod'])
                fl = fl + np.random.normal(0, 30)

            y = np.array([od_meas, fl])
            pp_rel = x['p']/(x['e'] + x['p'])
            if dithered:
                if x['e'] + x['p'] > od_th:
                    dil = True
                if x['e'] + x['p'] < od_l_th:
                    dil = False
            else:
                dil = 0
            if dt > 0:
                temp = alpha_temp * temp + (1 - alpha_temp) * temp_target
            # Log
            time_h_arr[time_m] = time_h
            od_arr[time_m] = od_meas
            fl_arr[time_m] = fl
            temp_arr[time_m] = temp
            pp_rel_arr[time_m] = pp_rel

            ### State Estimation
            u_ekf = np.array([temp, dil])
            ekf.estimate(time_s, u_ekf, y)
            x_est = ekf.est
            p_est = ekf.var
            od_est = x_est['e'] + x_est['p']
            fl_est = (x_est['fp'] + model_est.parameters['od_fac']*od_est) * model_est.parameters['e_fac']['Faith'] + model_est.parameters['e_ofs']['Faith']
            pp_rel_est = x_est['p']/od_est
            # Log
            od_est_arr[time_m] = od_est
            fl_est_arr[time_m] = fl_est
            pp_rel_od_arr[time_m] = ekf.p_est_od/y[0]
            pp_rel_fl_arr[time_m] = ekf.p_est_fl/y[0]
            pp_rel_est_arr[time_m] = pp_rel_est

            if time_h < 4:
                temp_target = 36
            elif time_h < 8:
                temp_target = 29
            elif time_h < 12:
                temp_target = 31
            else:
                temp_target = model_est.parameters['crit_temp']
                
            #log
            temp_target_arr[time_m] = temp_target
                    
            # Get data for next iteration
            time_s_prev = time_s
            time_m += 1
            time_s = time_m*60

        ### Plot Results
        max_fl = max(fl_arr)

        ax_m = ax[3]
        if j == 0:
            ax_m.plot(time_h_arr,pp_rel_arr*100, color = pp_color, lw = 1.5, label = 'Ground Truth, $p$')
            ax_m.plot(time_h_arr,pp_rel_est_arr*100, '--', color = pp_color,lw = 1, alpha = 0.7, label = 'Without Measurement Update')
        elif j == 1:
            ax_m.plot(time_h_arr,pp_rel_est_arr*100, '--', color = '#0000FF',lw = 1, alpha = 0.7, label = 'Only fl Update')
        elif j == 2:
            ax_m.plot(time_h_arr,pp_rel_est_arr*100, '-.', color = 'k',lw = 1, alpha = 0.7, label = 'Only od Curvature Update')
            # ax_m.plot(time_h_arr,pp_rel_od_arr*100, 'x', color = 'k',markersize = 1, label = 'od est')
        elif j == 3:
            ax_m.plot(time_h_arr,pp_rel_est_arr*100, '-.', color = '#0000FF',lw = 1, alpha = 0.7, label = 'Only fl Curvature Update')
            # ax_m.plot(time_h_arr,pp_rel_fl_arr*100, 'x', color = '#0000FF',markersize = 1, label = 'fl est')
        else:
            ax_m.plot(time_h_arr,pp_rel_est_arr*100, ':', color = pp_color,lw = 1.5, label = 'With all Measurements, $\hat{p}$')

    ax_t = ax[0]
    ax_t.hlines(model_sim.parameters['crit_temp'],time_h_arr[0]-0.5,time_h_arr[-1]+0.5,'r','--',lw=1, label = '$T_{crit}$')
    ax_t.plot(time_h_arr,temp_arr,'r',lw=1,alpha=1, label = '$T_{meas}$')
    tticks = np.array([29, model_sim.parameters['crit_temp'], 36])
    ax_t.set_yticks(tticks, labels=tticks)
    ax_t.set_ylim([28,37])
    ax_t.set_ylabel("Temp. ($\degree$C)")
    ax_t.legend(loc='lower right')

    ax_od = ax[1]
    ax_od.plot(time_h_arr,od_arr, 'k', lw = 1, label = '$od_{meas}$')
    ax_od.set_ylabel("OD")
    ax_od.legend(loc='lower right')

    ax_fl = ax[2]
    ax_fl.plot(time_h_arr,fl_arr, 'b', lw = 1, label = '$fl_{meas}$')
    ax_fl.set_ylabel("Fluorescence (a.u.)")
    ax_fl.legend(loc='lower right')

    ax_m.set_ylabel("Relative $P. putida$ Abundance (%)")
    ax_m.set_xlabel("Time (h)")
    ax_m.set_ylim([0,100])
    ax_m.set_xlim([-0.5,timeEnd_m/60+0.5])
    handles, labels = ax_m.get_legend_handles_labels()
    first_legend = ax_m.legend(handles[:1],labels[:1],loc='upper right')
    ax_m.add_artist(first_legend)
    ax_m.legend(handles[1:],labels[1:],loc='upper left', title = 'EKF Estimate', ncol = 2)

    fig.tight_layout()
    fig.align_ylabels()
        
    # Save figures
    dataName = "simulation"
    results_dir = "Images/{}".format(dataName)
    if noise:
        fig.suptitle('Simulation with noise and mode mismatch', fontsize=14)
    else:
        fig.suptitle('Simulation without noise and perfect knowledge of the model', fontsize=14)
    fig.tight_layout()
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    else:
        fig.savefig(results_dir + "/sim{}{}.svg".format('_' + test_var if test_var else '', '_noise' if noise else ''), transparent=True)
