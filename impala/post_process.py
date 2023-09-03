
import numpy as np
from impala import superCal as sc
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import seaborn as sns
from matplotlib.lines import Line2D
import pandas as pd
import scipy.stats as ss

def func_prediction_plot(
            actual_x, actual_y, # obs
            pred_y_lb, pred_y_ub, # prediction (using thetai in hierarchical)
            theta0_y_lb = None, theta0_y_ub = None, # hierarchical mean
            parent_y_lb = None, parent_y_ub = None, # hierarchical parent distribution
            text = None, pdf = None, ylim=None, sigma=None
            ):
    """ PTW Prediction Single Plot """
    fig = plt.figure(figsize = (4,3))
    if parent_y_lb is not None:
        plt.fill_between(actual_x, parent_y_lb, parent_y_ub, color = 'lightgrey', label = r'$\theta^*$')
    if theta0_y_lb is not None:
        plt.fill_between(actual_x, theta0_y_lb, theta0_y_ub, color = 'lightblue', label = r'$\theta_0$')
    plt.fill_between(actual_x, pred_y_lb, pred_y_ub, color = 'lightgreen', label = r'$\theta_i$')
    if sigma is not None:
        for i in range(len(actual_x)):
            plt.plot([actual_x[i], actual_x[i]],[actual_y[i]-sigma, actual_y[i]+sigma], color='blue', linewidth=0.1)
    plt.scatter(actual_x, actual_y, color = 'blue', s = 0.5, label = 'y')
    if ylim is not None:
        plt.ylim(ylim)
    if text is not None:
        #plt.text(*text_coords, text)
        plt.text(.1, 0.85, text, transform=plt.gca().transAxes)
    if pdf:
        pdf.savefig(fig)
    return




def ptw_prediction_plots_pool(setup, calib_out, path, mcmc_use, ylim=None, alpha=0.05):
    """ PTW Prediction Hierarchical Plots (no input) """
    pred_theta_raw = [np.empty([mcmc_use.shape[0], setup.ys[i].shape[0]]) for i in range(setup.nexp)]
    for i in range(setup.nexp):
        pred_theta_raw[i] = setup.models[i].eval(
            sc.tran_unif(
                calib_out.theta[mcmc_use,0],
                setup.bounds_mat, setup.bounds.keys(),
                ),
            )
    
    obs_strain = []
    obs_stress = []
    pred_theta = []
    pred_theta_quant_lb = []
    pred_theta_quant_ub = []
    text = []
    sigma = []
    ntot = 0
    for i in range(setup.nexp):
        for j in range(setup.ns2[i]):
            if type(setup.models[i]).__name__ == 'ModelMaterialStrength':
                ntot += 1
                pred_theta.append(pred_theta_raw[i].T[setup.s2_ind[i] == j].T)
                pred_theta_quant_lb.append(np.percentile(pred_theta[-1], alpha/2, 0))
                pred_theta_quant_ub.append(np.percentile(pred_theta[-1], 1-alpha/2, 0))
                text.append('edot: '+str(round(setup.models[i].edots[j]*1e6))+'/s\ntemp: '+str(round(setup.models[i].temps[j]))+'K')
                obs_strain.append(setup.models[i].meas_strain_histories[j])
                obs_stress.append(setup.ys[i][setup.s2_ind[i] == j])
                sigma.append(np.sqrt(calib_out.s2[i][mcmc_use, 0, j]).mean())

    if ylim == 'constant':
        m1 = min([pp.min() for pp in pred_theta_quant_lb] + [pp.min() for pp in obs_stress])
        m2 = max([pp.max() for pp in pred_theta_quant_ub] + [pp.max() for pp in obs_stress])
        ylim = [m1, m2]

    pdf = PdfPages(path)
    for k in range(ntot):
        func_prediction_plot(actual_x=obs_strain[k], actual_y=obs_stress[k],
            pred_y_lb=pred_theta_quant_lb[k], pred_y_ub=pred_theta_quant_ub[k],
            text=text[k], pdf=pdf, ylim=ylim, sigma=sigma[k])
    pdf.close()
    return


def ptw_prediction_plots_hier(setup, calib_out, path, mcmc_use, ylim=None, alpha=0.05):
    """ PTW Prediction Hierarchical Plots (no input) """
    pred_theta_raw  = [np.empty([mcmc_use.shape[0], setup.ys[i].shape[0]]) for i in range(setup.nexp)]
    pred_theta0_raw = [np.empty([mcmc_use.shape[0], setup.ys[i].shape[0]]) for i in range(setup.nexp)]
    pred_thetap_raw = [np.empty([mcmc_use.shape[0], setup.ys[i].shape[0]]) for i in range(setup.nexp)]

    theta_parent = sc.chol_sample_1per_constraints(
            calib_out.theta0[mcmc_use, 0], calib_out.Sigma0[mcmc_use, 0], setup.checkConstraints,
            setup.bounds_mat, setup.bounds.keys(), setup.bounds,
            )
    
    for i in range(setup.nexp):
        setup.models[i].pool = False
        pred_theta_raw[i] = setup.models[i].eval(
            sc.tran_unif(calib_out.theta[i][mcmc_use,0,:,:].reshape([-1, setup.p]), setup.bounds_mat, setup.bounds.keys()), pool=False, nugget=True
            )
        pred_theta0_raw[i] = setup.models[i].eval(
            sc.tran_unif(calib_out.theta0[mcmc_use,0], setup.bounds_mat, setup.bounds.keys()), pool=True, nugget=True
            )
        pred_thetap_raw[i] = setup.models[i].eval(
            sc.tran_unif(theta_parent, setup.bounds_mat, setup.bounds.keys()), pool=True, nugget=True
            )
        setup.models[i].pool = True
    obs_strain = []
    obs_stress = []
    pred_theta = []
    pred_theta0 = []
    pred_thetap = []
    pred_theta_quant_lb = []
    pred_theta_quant_ub = []
    pred_theta0_quant_lb = []
    pred_theta0_quant_ub = []
    pred_thetap_quant_lb = []
    pred_thetap_quant_ub = []
    text = []
    sigma = []

    ntot = 0
    for i in range(setup.nexp):
        for j in range(setup.ntheta[i]):
            if type(setup.models[i]).__name__ == 'ModelMaterialStrength':
                ntot += 1
                pred_theta.append(pred_theta_raw[i].T[setup.s2_ind[i] == j].T)
                pred_theta_quant_lb.append(np.percentile(pred_theta[-1], alpha/2, 0))
                pred_theta_quant_ub.append(np.percentile(pred_theta[-1], 1-alpha/2, 0))
                pred_theta0.append(pred_theta0_raw[i].T[setup.s2_ind[i] == j].T)
                pred_theta0_quant_lb.append(np.percentile(pred_theta0[-1], alpha/2, 0))
                pred_theta0_quant_ub.append(np.percentile(pred_theta0[-1], 1-alpha/2, 0))
                pred_thetap.append(pred_thetap_raw[i].T[setup.s2_ind[i] == j].T)
                pred_thetap_quant_lb.append(np.percentile(pred_thetap[-1], alpha/2, 0))
                pred_thetap_quant_ub.append(np.percentile(pred_thetap[-1], 1-alpha/2, 0))
                text.append('edot: '+str(round(setup.models[i].edots[j]*1e6))+'/s\ntemp: '+str(round(setup.models[i].temps[j]))+'K')
                obs_strain.append(setup.models[i].meas_strain_histories[j])
                obs_stress.append(setup.ys[i][setup.s2_ind[i] == j])
                sigma.append(np.sqrt(calib_out.s2[i][mcmc_use, 0, j]).mean())

    if ylim == 'constant':
        m1 = min([pp.min() for pp in pred_thetap_quant_lb] + [pp.min() for pp in obs_stress])
        m2 = max([pp.max() for pp in pred_thetap_quant_ub] + [pp.max() for pp in obs_stress])
        ylim = [m1, m2]

    pdf = PdfPages(path)
    for k in range(ntot):
        func_prediction_plot(actual_x=obs_strain[k], actual_y=obs_stress[k],
            pred_y_lb=pred_theta_quant_lb[k], pred_y_ub=pred_theta_quant_ub[k],
            text=text[k], pdf=pdf, ylim=ylim, sigma=sigma[k],
            theta0_y_lb=pred_theta0_quant_lb[k], theta0_y_ub=pred_theta0_quant_ub[k], # hierarchical mean
            parent_y_lb=pred_thetap_quant_lb[k], parent_y_ub=pred_thetap_quant_ub[k])
    pdf.close()
    return


def ptw_prediction_plots_cluster(setup, calib_out, path, mcmc_use, ylim=None, alpha=0.05):
    pred_theta_raw  = [np.empty([mcmc_use.shape[0], setup.ys[i].shape[0]]) for i in range(setup.nexp)]
    pred_theta0_raw = [np.empty([mcmc_use.shape[0], setup.ys[i].shape[0]]) for i in range(setup.nexp)]
    pred_thetap_raw = [np.empty([mcmc_use.shape[0], setup.ys[i].shape[0]]) for i in range(setup.nexp)]

    thetas = calib_out.theta[mcmc_use, 0]
    deltas = [calib_out.delta[i][mcmc_use] for i in range(setup.nexp)]
    nclustmax = max(calib_out.out.delta[i].max() for i in range(setup.nexp)) + 1
    dcounts = np.zeros((mcmc_use.shape[0], nclustmax))
    for it, s in enumerate(mcmc_use):
        for i in range(setup.nexp):
            dcounts[it] += np.bincount(calib_out.delta[i][s,0], minlength = nclustmax)
    etas = calib_out.eta[mcmc_use,0]
    nclust = (dcounts > 0).sum(axis = 1)
    prob   = dcounts + (dcounts == 0) * (etas / (nclustmax - nclust + 1e-9)).reshape(-1,1)
    prob[:] /= prob.sum(axis = 1).reshape(-1,1)
    cumprob = np.cumsum(prob, axis = 1)
    unif = np.random.uniform(size = (mcmc_use.shape[0],1))
    dnew = (unif > cumprob).sum(axis = 1)
    theta_parent = thetas[np.arange(mcmc_use.shape[0]), dnew]
    for i in range(setup.nexp):
        for j in range(mcmc_use.shape[0]):
            pred_theta_raw[i][j] = setup.models[i].eval(
                sc.tran_unif(
                    calib_out.theta_hist[i][mcmc_use[j],0], 
                    setup.bounds_mat, setup.bounds.keys(),
                    ),
                )
            pred_theta0_raw[i][j] = setup.models[i].eval(
                sc.tran_unif(
                    np.repeat(calib_out.theta0[mcmc_use[j],0].reshape(1,-1), setup.ns2[i], axis = 0), 
                    setup.bounds_mat, setup.bounds.keys(),
                    ),
                )
            pred_thetap_raw[i][j] = setup.models[i].eval(
                sc.tran_unif(
                    np.repeat(theta_parent[j].reshape(1,-1), setup.ns2[i], axis = 0), 
                    setup.bounds_mat, setup.bounds.keys(),
                    ),
                )
    
    obs_strain = []
    obs_stress = []
    pred_theta = []
    pred_theta0 = []
    pred_thetap = []
    pred_theta_quant_lb = []
    pred_theta_quant_ub = []
    pred_theta0_quant_lb = []
    pred_theta0_quant_ub = []
    pred_thetap_quant_lb = []
    pred_thetap_quant_ub = []
    text = []
    sigma = []
    ntot = 0
    for i in range(setup.nexp):
        for j in range(setup.ntheta[i]):
            if type(setup.models[i]).__name__ == 'ModelMaterialStrength':
                ntot += 1
                pred_theta.append(pred_theta_raw[i].T[setup.s2_ind[i] == j].T)
                pred_theta_quant_lb.append(np.percentile(pred_theta[-1], alpha/2, 0))
                pred_theta_quant_ub.append(np.percentile(pred_theta[-1], 1-alpha/2, 0))
                pred_theta0.append(pred_theta0_raw[i].T[setup.s2_ind[i] == j].T)
                pred_theta0_quant_lb.append(np.percentile(pred_theta0[-1], alpha/2, 0))
                pred_theta0_quant_ub.append(np.percentile(pred_theta0[-1], 1-alpha/2, 0))
                pred_thetap.append(pred_thetap_raw[i].T[setup.s2_ind[i] == j].T)
                pred_thetap_quant_lb.append(np.percentile(pred_thetap[-1], alpha/2, 0))
                pred_thetap_quant_ub.append(np.percentile(pred_thetap[-1], 1-alpha/2, 0))
                text.append('edot: '+str(round(setup.models[i].edots[j]*1e6))+'/s\ntemp: '+str(round(setup.models[i].temps[j]))+'K')
                obs_strain.append(setup.models[i].meas_strain_histories[j])
                obs_stress.append(setup.ys[i][setup.s2_ind[i] == j])
                sigma.append(np.sqrt(calib_out.s2[i][mcmc_use, 0, j]).mean())

    if ylim == 'constant':
        m1 = min([pp.min() for pp in pred_thetap_quant_lb] + [pp.min() for pp in obs_stress])
        m2 = max([pp.max() for pp in pred_thetap_quant_ub] + [pp.max() for pp in obs_stress])
        ylim = [m1, m2]

    pdf = PdfPages(path)
    for k in range(ntot):
        func_prediction_plot(actual_x=obs_strain[k], actual_y=obs_stress[k],
            pred_y_lb=pred_theta_quant_lb[k], pred_y_ub=pred_theta_quant_ub[k],
            text=text[k], pdf=pdf, ylim=ylim, sigma=sigma[k],
            theta0_y_lb=pred_theta0_quant_lb[k], theta0_y_ub=pred_theta0_quant_ub[k], # hierarchical mean
            parent_y_lb=pred_thetap_quant_lb[k], parent_y_ub=pred_thetap_quant_ub[k])
    pdf.close()
    return

def kde_contour(x1, x2, percentile):
    density = gaussian_kde([x1,x2], bw_method = 'silverman')
    X, Y = np.mgrid[min(x1):max(x1):100j, min(x2):max(x2):100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = density(positions)
    Z /= Z.sum()
    t = np.linspace(0, Z.max(), 1000)
    integral = ((Z >= t[:, None, None]) * Z).sum(axis = (1,2))
    f = interp1d(integral, t)
    t_contours = f(np.array([percentile]))
    return {'X' : X, 'Y' : Y, 'Z' : Z.reshape([100,100]), 'conts' : t_contours}


def pairwise_theta_plot_hier(setup, calib_out, path, mcmc_use, alpha=0.05, highlight=None):
    """ Pairwise Theta scatterplot """
    lty = ['solid', 'dotted', 'dashed', 'dashdot']
    if highlight is None:
        highlight = [range(setup.ntheta[k]) for k in range(setup.nexp)]
    theta_parent = sc.chol_sample_1per_constraints(
        calib_out.theta0[mcmc_use,0], calib_out.Sigma0[mcmc_use,0], setup.checkConstraints,
        setup.bounds_mat, setup.bounds.keys(), setup.bounds,
        )
    theta_names = list(setup.bounds.keys())
    theta0_unst = sc.unnormalize(calib_out.theta0[mcmc_use, 0, :], setup.bounds_mat)
    theta_parent_unst = sc.unnormalize(theta_parent, setup.bounds_mat)
    theta_unst = [calib_out.theta[k][mcmc_use,0,:,:] for k in range(setup.nexp)]
    for k in range(setup.nexp):
        for s in range(setup.ntheta[k]):
            theta_unst[k][:,s,:] = sc.unnormalize(calib_out.theta[k][mcmc_use,0,s,:], setup.bounds_mat)
    plt.figure(figsize = (15,15))
    for i in range(setup.p):
        for j in range(setup.p):
            if i == j:
                plt.subplot2grid((setup.p, setup.p), (i,j))
                for k in range(setup.nexp):
                    #for s in range(self.setup.ntheta[k]):
                    if highlight[k] is not None:
                        for s in highlight[k]:
                            sns.kdeplot(theta_unst[k][:,s,i], color = 'lightgreen')
                sns.kdeplot(theta0_unst[:, i], color = 'blue')
                sns.kdeplot(theta_parent_unst[:, i], color = 'grey')
                #plt.xlim(0,1)

                plt.xlim(setup.bounds_mat[i, 0], setup.bounds_mat[i, 1])

                ax = plt.gca()
                ax.axes.yaxis.set_visible(False)
                plt.xlabel(theta_names[i])
                ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
                plt.setp(ax.get_xticklabels(), rotation = 30, horizontalalignment = 'right')
            elif i < j:
                plt.subplot2grid((setup.p, setup.p), (i,j))
                cnt = 0
                for k in range(setup.nexp):
                    #for s in range(self.setup.ntheta[k]):
                    if highlight[k] is not None:
                        for s in highlight[k]:
                            cnt += 1
                            contour = kde_contour(theta_unst[k][:, s, j], theta_unst[k][:, s, i], 1-alpha)
                            plt.contour(contour['X'], contour['Y'], contour['Z'], contour['conts'], colors = 'lightgreen', linestyles=lty[cnt % len(lty)])
                contour = kde_contour(theta0_unst[:, j], theta0_unst[:, i], 1-alpha)
                plt.contour(contour['X'], contour['Y'], contour['Z'], contour['conts'], colors = 'blue')

                contour = kde_contour(theta_parent_unst[:, j], theta_parent_unst[:, i], 1-alpha)
                plt.contour(contour['X'], contour['Y'], contour['Z'], contour['conts'], colors = 'grey')
                #plt.xlim(0,1)
                #plt.ylim(0,1)

                plt.xlim(setup.bounds_mat[j, 0], setup.bounds_mat[j, 1])
                plt.ylim(setup.bounds_mat[i, 0], setup.bounds_mat[i, 1])

                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
            else:
                pass
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.subplot2grid((setup.p, setup.p), (2, 0))
    colors = ['lightgreen','blue','grey']
    lines = [Line2D([0],[0],color = c, linewidth = 2) for c in colors]
    labels = [r'$\theta_i$',r'$\theta_0$',r'$\theta^*$']
    plt.legend(lines,labels)
    plt.axis('off')
    if path:
        plt.savefig(path, bbox_inches = 'tight')
    else:
        plt.show()
    return


def pairwise_theta_plot_pool(setup, calib_out, path, mcmc_use, alpha=0.05):
    """ Pairwise Theta scatterplot """
    theta_names = list(setup.bounds.keys())
    theta0_unst = sc.unnormalize(calib_out.theta[mcmc_use,0,:], setup.bounds_mat)
    plt.figure(figsize = (15,15))
    for i in range(setup.p):
        for j in range(setup.p):
            if i == j:
                plt.subplot2grid((setup.p, setup.p), (i,j))
                sns.kdeplot(theta0_unst[:, i], color = 'blue')
                #plt.xlim(0,1)
                plt.xlim(setup.bounds_mat[i, 0], setup.bounds_mat[i, 1])
                ax = plt.gca()
                ax.axes.yaxis.set_visible(False)
                plt.xlabel(theta_names[i])
                ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
                plt.setp(ax.get_xticklabels(), rotation = 30, horizontalalignment = 'right')
            elif i < j:
                plt.subplot2grid((setup.p, setup.p), (i,j))
                contour = kde_contour(theta0_unst[:, j], theta0_unst[:, i], 1-alpha)
                plt.contour(contour['X'], contour['Y'], contour['Z'], contour['conts'], colors = 'blue')
                #plt.xlim(0,1)
                #plt.ylim(0,1)
                plt.xlim(setup.bounds_mat[j, 0], setup.bounds_mat[j, 1])
                plt.ylim(setup.bounds_mat[i, 0], setup.bounds_mat[i, 1])
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
            else:
                pass
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.subplot2grid((setup.p, setup.p), (2, 0))
    plt.axis('off')
    if path:
        plt.savefig(path, bbox_inches = 'tight')
    else:
        plt.show()
    return

def pairwise_theta_plot_pool_compare(setup, calib_out_list, cols, path, mcmc_use, alpha=0.05):
    """ Pairwise Theta scatterplot """
    theta_names = list(setup.bounds.keys())
    theta0_unst_list = [sc.unnormalize(x.theta[mcmc_use,0,:], setup.bounds_mat) for x in calib_out_list]
    n = len(calib_out_list)
    plt.figure(figsize = (15,15))
    for i in range(setup.p):
        for j in range(setup.p):
            if i == j:
                plt.subplot2grid((setup.p, setup.p), (i,j))
                for k in range(n):
                    sns.kdeplot(theta0_unst_list[k][:, i], color = cols[k])
                #plt.xlim(0,1)
                plt.xlim(setup.bounds_mat[i, 0], setup.bounds_mat[i, 1])
                ax = plt.gca()
                ax.axes.yaxis.set_visible(False)
                plt.xlabel(theta_names[i])
                ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
                plt.setp(ax.get_xticklabels(), rotation = 30, horizontalalignment = 'right')
            elif i < j:
                plt.subplot2grid((setup.p, setup.p), (i,j))
                contour_list = [kde_contour(x[:, j], x[:, i], 1-alpha) for x in theta0_unst_list]
                for k in range(n):
                    plt.contour(contour_list[k]['X'], contour_list[k]['Y'], contour_list[k]['Z'], contour_list[k]['conts'], colors = cols[k])
                #plt.xlim(0,1)
                #plt.ylim(0,1)
                plt.xlim(setup.bounds_mat[j, 0], setup.bounds_mat[j, 1])
                plt.ylim(setup.bounds_mat[i, 0], setup.bounds_mat[i, 1])
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
            else:
                pass
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.subplot2grid((setup.p, setup.p), (2, 0))
    plt.axis('off')
    if path:
        plt.savefig(path, bbox_inches = 'tight')
    else:
        plt.show()
    return



def pairwise_theta_plot_cluster(setup, calib_out, path, mcmc_use, alpha=0.05, highlight=None):
    thetas = calib_out.theta[mcmc_use,0]
    deltas = [calib_out.delta[i][mcmc_use] for i in range(setup.nexp)]
    nclustmax = max(calib_out.delta[i].max() for i in range(setup.nexp)) + 1
    dcounts = np.zeros((mcmc_use.shape[0], nclustmax))
    for it, s in enumerate(mcmc_use):
        for i in range(setup.nexp):
            dcounts[it] += np.bincount(calib_out.delta[i][s,0], minlength = nclustmax)
    etas = calib_out.eta[mcmc_use, 0]
    nclust = (dcounts > 0).sum(axis = 1)
    prob   = dcounts + (dcounts == 0) * (etas / (nclustmax - nclust + 1e-9)).reshape(-1,1)
    prob  /= prob.sum(axis = 1).reshape(-1,1)
    cumprob = np.cumsum(prob, axis = 1)
    unif = np.random.uniform(size = (mcmc_use.shape[0],1))
    dnew = (unif > cumprob).sum(axis = 1)
    theta_parent = thetas[np.arange(mcmc_use.shape[0]), dnew]


    lty = ['solid', 'dotted', 'dashed', 'dashdot']
    if highlight is None:
        highlight = [range(setup.ntheta[k]) for k in range(setup.nexp)]
    theta_names = list(setup.bounds.keys())
    theta0_unst = sc.unnormalize(calib_out.theta0[mcmc_use, 0, :], setup.bounds_mat)
    theta_parent_unst = sc.unnormalize(theta_parent, setup.bounds_mat)
    theta_unst = [calib_out.theta_hist[k][mcmc_use,0,:,:] for k in range(setup.nexp)]
    for k in range(setup.nexp):
        for s in range(setup.ntheta[k]):
            theta_unst[k][:,s,:] = sc.unnormalize(calib_out.theta[k][mcmc_use,0,s,:], setup.bounds_mat)
    plt.figure(figsize = (15,15))
    for i in range(setup.p):
        for j in range(setup.p):
            if i == j:
                plt.subplot2grid((setup.p, setup.p), (i,j))
                for k in range(setup.nexp):
                    #for s in range(self.setup.ntheta[k]):
                    if highlight[k] is not None:
                        for s in highlight[k]:
                            sns.kdeplot(theta_unst[k][:,s,i], color = 'lightgreen')
                sns.kdeplot(theta0_unst[:, i], color = 'blue')
                sns.kdeplot(theta_parent_unst[:, i], color = 'grey')
                #plt.xlim(0,1)

                plt.xlim(setup.bounds_mat[i, 0], setup.bounds_mat[i, 1])

                ax = plt.gca()
                ax.axes.yaxis.set_visible(False)
                plt.xlabel(theta_names[i])
                ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
                plt.setp(ax.get_xticklabels(), rotation = 30, horizontalalignment = 'right')
            elif i < j:
                plt.subplot2grid((setup.p, setup.p), (i,j))
                cnt = 0
                for k in range(setup.nexp):
                    #for s in range(self.setup.ntheta[k]):
                    if highlight[k] is not None:
                        for s in highlight[k]:
                            cnt += 1
                            contour = kde_contour(theta_unst[k][:, s, j], theta_unst[k][:, s, i], 1-alpha)
                            plt.contour(contour['X'], contour['Y'], contour['Z'], contour['conts'], colors = 'lightgreen', linestyles=lty[cnt % len(lty)])
                contour = kde_contour(theta0_unst[:, j], theta0_unst[:, i], 1-alpha)
                plt.contour(contour['X'], contour['Y'], contour['Z'], contour['conts'], colors = 'blue')

                contour = kde_contour(theta_parent_unst[:, j], theta_parent_unst[:, i], 1-alpha)
                plt.contour(contour['X'], contour['Y'], contour['Z'], contour['conts'], colors = 'grey')
                #plt.xlim(0,1)
                #plt.ylim(0,1)

                plt.xlim(setup.bounds_mat[j, 0], setup.bounds_mat[j, 1])
                plt.ylim(setup.bounds_mat[i, 0], setup.bounds_mat[i, 1])

                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
            else:
                pass
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.subplot2grid((setup.p, setup.p), (2, 0))
    colors = ['lightgreen','blue','grey']
    lines = [Line2D([0],[0],color = c, linewidth = 2) for c in colors]
    labels = [r'$\theta_i$',r'$\theta_0$',r'$\theta^*$']
    plt.legend(lines,labels)
    plt.axis('off')
    try:
        plt.subplot2grid((setup.p, setup.p), (4, 0))
        sns.distplot(nclust, kde = True, color = 'blue')
        plt.xlim(0,nclustmax)
    except IndexError:
        pass
    if path:
        plt.savefig(path, bbox_inches = 'tight')
    else:
        plt.show()
    return


def cluster_matrix(delta_list, ns2, nclustmax, nburn = 20000, nthin = 10):
    # subset delta to post burn-in
    delta_relist = [d[nburn::nthin] for d in delta_list]
    # Declare constants 
    nsamp = delta_relist[0].shape[0]
    nexp  = len(delta_relist)
    # create a combined delta array (for all experiments/vectorized experiments)
    # Boolean array, so (True iff member of cluster)
    breaks = np.hstack((0,np.cumsum(ns2)))
    bounds = [(breaks[i],breaks[i+1]) for i in range(breaks.shape[0] - 1)]
    delta_mat = np.empty((delta_relist[0].shape[0], breaks[-1], nclustmax), dtype = bool)
    # Fill the combined delta array
    for i in range(breaks.shape[0] - 1):
        delta_mat[:,bounds[i][0]:bounds[i][1]] = delta_relist[i][:,0,:,None] == np.arange(nclustmax)
    # X = matrix of incidence (nsamp x nclust) ->  XX^t matrix of coincidence (nsamp x nsamp)
    # Average over all iterations -> matrix of average coincidence / shared cluster membership
    out = np.einsum('icp,iqp->icq', delta_mat, delta_mat).mean(axis = 0)
    return out, breaks

def cluster_matrix_plot(setup, calib_out, path = None, **kwargs):
    cmat, breaks = cluster_matrix(calib_out.delta, setup.ns2, calib_out.nclustmax, **kwargs)
    plt.matshow(cmat)
    if breaks.shape[0] > 1:
        for breakpoint in breaks[1:-1] - 0.5:
            plt.axhline(breakpoint, color = 'red', linestyle = '--')
            plt.axvline(breakpoint, color = 'green', linestyle = '--')
    plt.legend()
    if path:
        plt.savefig(path, bbox_inches = 'tight')
    else:
        plt.show()
    return

def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)

def pairs(setup, mat_st, col=None, s=None):
    dat = pd.DataFrame(sc.tran_unif(mat_st, setup.bounds_mat, setup.bounds.keys()))
    if col is None:
        col = ['blue']*mat_st.shape[0]
    if s is None:
        s = [3]*mat_st.shape[0]
    dat['col'] = col
    #g = sns.pairplot(dat, plot_kws={"s": s}, corner=True, diag_kind='hist', hue='col')
    g = sns.pairplot(dat, plot_kws={"s": s}, diag_kind='hist', hue='col')
    g.map_upper(hide_current_axis) # to make compatible with early version of seaborn
    for i in range(mat_st.shape[1]):
        g.axes[i,i].set_xlim(setup.bounds[dat.keys()[i]])
        g.axes[i,i].set_ylim(setup.bounds[dat.keys()[i]])
    g.fig.set_size_inches(10,10)
    g
#    plt.show()
    return

def parameter_trace_plot(sample_parameters,ylim=None):
    palette = plt.get_cmap('Set1')
    if len(sample_parameters.shape) == 1:
        n = sample_parameters.shape[0]
        plt.plot(range(n), sample_parameters, marker = '', linewidth = 1)
    else:
        # df = pd.DataFrame(sample_parameters, self.parameter_order)
        n, d = sample_parameters.shape
        for i in range(d):
            plt.subplot(d, 1, i+1)
            plt.plot(range(n), sample_parameters[:,i], marker = '',
                    color = palette(i), linewidth = 1)
            ax = plt.gca()
            if ylim is not None:
                ax.set_ylim(ylim)
#    plt.show()
    return



def save_parent_strength(setup, ptw_mod, calib_out, mcmc_use, path):

    theta_parent = sc.chol_sample_1per_constraints(
        calib_out.theta0[mcmc_use,0], calib_out.Sigma0[mcmc_use,0], setup.checkConstraints,
        setup.bounds_mat, setup.bounds.keys(), setup.bounds,
        )
    theta_parent_native = sc.unnormalize(theta_parent, setup.bounds_mat)

    #sub_dict = {k: calib_out.theta_parent_native[k][mcmc_use] for k in calib_out.theta_parent_native.keys()}
    #params = pd.DataFrame(sub_dict)
    #params = pd.DataFrame(theta_parent_native)

    theta_parent_native_dict = sc.tran_unif(theta_parent, setup.bounds_mat, setup.bounds.keys())

    params = pd.DataFrame(theta_parent_native_dict)

    pred = [setup.models[i].eval(theta_parent_native_dict, pool=True) for i in range(setup.nexp)]
    llik = sum([((pred[i]-setup.ys[i])**2).mean(axis=1) for i in range(setup.nexp)])

    #params['sse'] = calib_out.llik[mcmc_use]
    params['sse'] = llik
    consts = pd.DataFrame(ptw_mod.constants, index=[0])
    bounds = pd.DataFrame(setup.bounds)
    mods = ptw_mod.model_info
    # should save other stuff, but this is enough to do ranking
    with open(path, 'w') as fd:
        fd.write(','.join(mods) + '\n')
    with open(path, 'a') as fd:
        fd.write(consts.to_csv(index=False))
        fd.write(bounds.to_csv(index=False))
        fd.write('===\n')
        fd.write(params.to_csv(index=False))
    return

def get_bounds(edot, strain, temp, results_csv, write_path, percentiles=[0.05, 0.5, 0.95]):
    # rank parent distribution samples by stress at particular strain, strain rate, temperature, save to file
    edot_star = edot * 1e-6 # first term is per second

    df = pd.read_csv(results_csv, nrows=1, header=None)
    mods = df.loc[0, :].values.tolist()

    df = pd.read_csv(results_csv, skiprows=1, nrows=1)
    consts = df.to_dict('records')[0]

    df = pd.read_csv(results_csv, skiprows=7)
    theta_parent_native = dict(zip(df.T.index,df.values.T))


    model_ptw_star = sc.ModelMaterialStrength(
        temps=np.array(temp), 
        edots=np.array(edot*1e-6), 
        consts=consts, 
        strain_histories=[np.arange(0, strain, .01)], 
        flow_stress_model=mods[0],
        shear_model=mods[1],
        specific_heat_model=mods[2], 
        melt_model=mods[3],  
        density_model=mods[4],
        pool=True)
    
    stress_star = model_ptw_star.eval(theta_parent_native)[:,-1]
    rank = np.argsort(stress_star)
    rank_sse = np.argsort(theta_parent_native['sse'])
    idx_rank = (np.array(percentiles) * stress_star.shape[0]).astype(int)
    sub_dat = pd.DataFrame({k: theta_parent_native[k][rank][idx_rank] for k in theta_parent_native.keys()})
    sub_dat['perc'] = percentiles
    sub_dat['stress'] = stress_star[rank][idx_rank]
    sub_dat['sse_rank'] = rank_sse[rank][idx_rank]

    template = "edot(1/s)=" + str(edot) + ", strain=" + str(strain) + ", temp(K)=" + str(temp) + "\n{}"

    with open(write_path, 'w') as fp:
        fp.write(template.format(sub_dat.to_csv(index=False)))

    return




def get_samples_rank(edot, strain, temp, results_csv, write_path):
    # rank parent distribution samples by stress at particular strain, strain rate, temperature, save all samples to file, for sky
    edot_star = edot * 1e-6 # first term is per second

    df = pd.read_csv(results_csv, nrows=1, header=None)
    mods = df.loc[0, :].values.tolist()

    df = pd.read_csv(results_csv, skiprows=1, nrows=1)
    consts = df.to_dict('records')[0]

    df = pd.read_csv(results_csv, skiprows=7)
    theta_parent_native = dict(zip(df.T.index,df.values.T))


    model_ptw_star = sc.ModelMaterialStrength(
        temps=np.array(temp), 
        edots=np.array(edot*1e-6), 
        consts=consts, 
        strain_histories=[np.arange(0, strain, .01)], 
        flow_stress_model=mods[0],
        shear_model=mods[1],
        specific_heat_model=mods[2], 
        melt_model=mods[3],  
        density_model=mods[4],
        pool=True)
    
    stress_star = model_ptw_star.eval(theta_parent_native)[:,-1]
    ranked_post = pd.DataFrame(theta_parent_native)
    ranked_post['stress'] = stress_star
    ranked_post['rank'] = ss.rankdata(stress_star) # append

    template = "edot(1/s)=" + str(edot) + ", strain=" + str(strain) + ", temp(K)=" + str(temp) + "\n{}"

    with open(write_path, 'w') as fp:
        fp.write(template.format(ranked_post.to_csv(index=False)))

    return


def get_best_sse(results_csv, write_path):

    df = pd.read_csv(results_csv, skiprows=7)
    theta_parent_native = dict(zip(df.T.index,df.values.T))
    rank_sse = np.argsort(theta_parent_native['sse'])
    sub_dat = pd.DataFrame({k: theta_parent_native[k][rank_sse][0] for k in theta_parent_native.keys()}, index=[0])
    
    with open(write_path, 'w') as fp:
        fp.write(sub_dat.to_csv(index=False))

    return

