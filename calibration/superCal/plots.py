from math import ceil, sqrt
from models import interpolate_experiment
import seaborn as sns
import numpy as np
import impala as impala
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from numpy.random import uniform

class PTW_Plotter(object):
    """ PTW Prediction and Pairwise Plots """
    pooled = None

    def ptw_prediction_plot_single(
                self,
                actual_x, actual_y, 
                pred_y_lb, pred_y_ub,
                hier_y_lb = None, hier_y_ub = None,
                pnew_y_lb = None, pnew_y_ub = None,
                edot = None, temp = None, pdf = None,
                ):
        """ PTW Prediction Single Plot """
        ymaxs = [actual_y.max(), pred_y_ub.max()]
        if hier_y_ub is not None:
            ymaxs.append(hier_y_ub.max())
        if pnew_y_ub is not None:
            ymaxs.append(pnew_y_ub.max())
        ymax = max(ymaxs)
        xmax = actual_x.max()
        text_coords = (0.1 * xmax, 0.85 * ymax)

        fig = plt.figure(figsize = (4,3))
        if pnew_y_lb is not None:
            plt.fill_between(actual_x, pnew_y_lb, pnew_y_ub, color = 'lightgrey', label = r'$\theta^*$')
        if hier_y_lb is not None:
            plt.fill_between(actual_x, hier_y_lb, hier_y_ub, color = 'lightblue', label = r'$\theta_0$')
        plt.fill_between(actual_x, pred_y_lb, pred_y_ub, color = 'lightgreen', label = r'$\theta_i$')
        plt.scatter(actual_x, actual_y, color = 'blue', s = 0.5, label = 'y')
        if edot is not None and temp is not None:
            plt.text(*text_coords, 'edot: {:.3E}\n temp: {}'.format(edot, temp))
        if pdf:
            pdf.savefig(fig)
        return

    def ptw_prediction_plots_hier(self, path):
        """ PTW Prediction Hierarchical Plots (no input) """
        sel = np.arange(20000, self.setup.nmcmc, 10) # need to script this in
        pred_theta_raw  = [np.empty([sel.shape[0], self.setup.ys[i].shape[0]]) for i in range(self.setup.nexp)]
        pred_theta0_raw = [np.empty([sel.shape[0], self.setup.ys[i].shape[0]]) for i in range(self.setup.nexp)]
        pred_thetap_raw = [np.empty([sel.shape[0], self.setup.ys[i].shape[0]]) for i in range(self.setup.nexp)]

        theta_parent = impala.chol_sample_1per_constraints(
                self.out.theta0[sel, 0], self.out.Sigma0[sel, 0], self.setup.checkConstraints,
                self.setup.bounds_mat, self.setup.bounds.keys(), self.setup.bounds,
                )
        
        for i in range(self.setup.nexp):
            for j in range(sel.shape[0]):
                pred_theta_raw[i][j] = self.setup.models[i].eval(
                    impala.tran(self.out.theta[i][sel[j],0], self.setup.bounds_mat,self. setup.bounds.keys())
                    )
                pred_theta0_raw[i][j] = self.setup.models[i].eval(
                    impala.tran(
                        np.repeat(self.out.theta0[sel[j],0].reshape(1,-1), self.setup.ntheta[i], axis = 0), 
                        self.setup.bounds_mat, self.setup.bounds.keys(),
                        )
                    )
                pred_thetap_raw[i][j] = self.setup.models[i].eval(
                    impala.tran(np.repeat(theta_parent[j].reshape(1,-1), self.setup.ntheta[i], axis = 0), 
                    self.setup.bounds_mat, self.setup.bounds.keys()),
                    )
        
        real_strain = []
        real_stress = []
        pred_theta = []
        pred_theta0 = []
        pred_thetap = []
        pred_theta_quant_lb = []
        pred_theta_quant_ub = []
        pred_theta0_quant_lb = []
        pred_theta0_quant_ub = []
        pred_thetap_quant_lb = []
        pred_thetap_quant_ub = []
        edots = []
        temps = []

        for i in range(self.setup.nexp):
            for j in range(self.setup.ntheta[i]):
                pred_theta.append(pred_theta_raw[i].T[self.setup.s2_ind[i] == j].T)
                pred_theta_quant_lb.append(np.quantile(pred_theta[-1], 0.025, 0))
                pred_theta_quant_ub.append(np.quantile(pred_theta[-1], 0.975, 0))
                pred_theta0.append(pred_theta0_raw[i].T[self.setup.s2_ind[i] == j].T)
                pred_theta0_quant_lb.append(np.quantile(pred_theta0[-1], 0.025, 0))
                pred_theta0_quant_ub.append(np.quantile(pred_theta0[-1], 0.975, 0))
                pred_thetap.append(pred_thetap_raw[i].T[self.setup.s2_ind[i] == j].T)
                pred_thetap_quant_lb.append(np.quantile(pred_thetap[-1], 0.025, 0))
                pred_thetap_quant_ub.append(np.quantile(pred_thetap[-1], 0.975, 0))
                edots.append(self.setup.models[i].edots[j])
                temps.append(self.setup.models[i].temps[j])
                real_strain.append(self.setup.models[i].meas_strain_histories[j])
                real_stress.append(self.setup.ys[i][self.setup.s2_ind[i] == j])

        rows = zip(
            real_strain, real_stress, pred_theta_quant_lb, pred_theta_quant_ub,
            pred_theta0_quant_lb, pred_theta0_quant_ub, pred_thetap_quant_lb, pred_thetap_quant_ub,
            edots, temps,
            )
        keys = [
            'actual_x','actual_y','pred_y_lb', 'pred_y_ub','hier_y_lb', 
            'hier_y_ub','pnew_y_lb','pnew_y_ub','edot','temp',
            ]
        plot_param_list = [dict(zip(keys,row)) for row in rows]

        pdf = PdfPages(path)
        for plot_params in plot_param_list:
            self.ptw_prediction_plot_single(**plot_params, pdf = pdf)
        pdf.close()
        return

    def ptw_prediction_plots_pool(self, path):
        """ PTW Prediction Hierarchical Plots (no input) """
        sel = np.arange(20000, self.setup.nmcmc, 10) # need to script this in
        pred_theta_raw  = [
            np.empty([sel.shape[0], self.setup.ys[i].shape[0]]) 
            for i in range(self.setup.nexp)
            ]
        for i in range(self.setup.nexp):
            pred_theta_raw[i] = self.setup.models[i].eval(
                impala.tran(self.out.theta[sel,0], self.setup.bounds_mat, self.setup.bounds.keys()),
                )
        
        real_strain = []
        real_stress = []
        pred_theta = []
        pred_theta_quant_lb = []
        pred_theta_quant_ub = []
        edots = []
        temps = []

        for i in range(self.setup.nexp):
            for j in range(self.setup.ns2[i]):
                pred_theta.append(pred_theta_raw[i].T[self.setup.s2_ind[i] == j].T)
                pred_theta_quant_lb.append(np.quantile(pred_theta[-1], 0.025, 0))
                pred_theta_quant_ub.append(np.quantile(pred_theta[-1], 0.975, 0))
                edots.append(self.setup.models[i].edots[j])
                temps.append(self.setup.models[i].temps[j])
                real_strain.append(self.setup.models[i].meas_strain_histories[j])
                real_stress.append(self.setup.ys[i][self.setup.s2_ind[i] == j])

        rows = zip(
            real_strain,real_stress,pred_theta_quant_lb,pred_theta_quant_ub,edots,temps,
            )
        keys = ['actual_x','actual_y','pred_y_lb', 'pred_y_ub','edot','temp']
        plot_param_list = [dict(zip(keys,row)) for row in rows]

        pdf = PdfPages(path)
        for plot_params in plot_param_list:
            self.ptw_prediction_plot_single(**plot_params, pdf = pdf)
        pdf.close()
        return

    def ptw_prediction_plots_cluster(self, path):
        sel = np.arange(20000, self.setup.nmcmc, 10) # need to script this in
        pred_theta_raw  = [np.empty([sel.shape[0], self.setup.ys[i].shape[0]]) for i in range(self.setup.nexp)]
        pred_theta0_raw = [np.empty([sel.shape[0], self.setup.ys[i].shape[0]]) for i in range(self.setup.nexp)]
        pred_thetap_raw = [np.empty([sel.shape[0], self.setup.ys[i].shape[0]]) for i in range(self.setup.nexp)]

        thetas = self.out.theta[sel,0]
        deltas = [self.out.delta[i][sel] for i in range(self.setup.nexp)]
        nclustmax = max(self.out.delta[i].max() for i in range(self.setup.nexp)) + 1
        dcounts = np.zeros((sel.shape[0], nclustmax))
        for it, s in enumerate(sel):
            for i in range(self.setup.nexp):
                dcounts[it] += np.bincount(self.out.delta[i][s,0], minlength = nclustmax)
        etas = self.out.eta[sel,0]
        nclust = (dcounts > 0).sum(axis = 1)
        prob   = dcounts + (dcounts == 0) * (etas / (nclustmax - nclust + 1e-9)).reshape(-1,1)
        prob[:] /= prob.sum(axis = 1).reshape(-1,1)
        cumprob = np.cumsum(prob, axis = 1)
        unif = uniform(size = (sel.shape[0],1))
        dnew = (unif > cumprob).sum(axis = 1)
        theta_parent = thetas[np.arange(sel.shape[0]), dnew]
        for i in range(self.setup.nexp):
            for j in range(sel.shape[0]):
                pred_theta_raw[i][j] = self.setup.models[i].eval(
                    impala.tran(
                        self.out.theta_hist[i][sel[j],0], 
                        self.setup.bounds_mat,self. setup.bounds.keys(),
                        ),
                    )
                pred_theta0_raw[i][j] = self.setup.models[i].eval(
                    impala.tran(
                        np.repeat(self.out.theta0[sel[j],0].reshape(1,-1), self.setup.ns2[i], axis = 0), 
                        self.setup.bounds_mat, self.setup.bounds.keys(),
                        ),
                    )
                pred_thetap_raw[i][j] = self.setup.models[i].eval(
                    impala.tran(
                        np.repeat(theta_parent[j].reshape(1,-1), self.setup.ns2[i], axis = 0), 
                        self.setup.bounds_mat, self.setup.bounds.keys(),
                        ),
                    )
        
        real_strain = []
        real_stress = []
        pred_theta = []
        pred_theta0 = []
        pred_thetap = []
        pred_theta_quant_lb = []
        pred_theta_quant_ub = []
        pred_theta0_quant_lb = []
        pred_theta0_quant_ub = []
        pred_thetap_quant_lb = []
        pred_thetap_quant_ub = []
        edots = []
        temps = []

        for i in range(self.setup.nexp):
            for j in range(self.setup.ntheta[i]):
                pred_theta.append(pred_theta_raw[i].T[self.setup.s2_ind[i] == j].T)
                pred_theta_quant_lb.append(np.quantile(pred_theta[-1], 0.025, 0))
                pred_theta_quant_ub.append(np.quantile(pred_theta[-1], 0.975, 0))
                pred_theta0.append(pred_theta0_raw[i].T[self.setup.s2_ind[i] == j].T)
                pred_theta0_quant_lb.append(np.quantile(pred_theta0[-1], 0.025, 0))
                pred_theta0_quant_ub.append(np.quantile(pred_theta0[-1], 0.975, 0))
                pred_thetap.append(pred_thetap_raw[i].T[self.setup.s2_ind[i] == j].T)
                pred_thetap_quant_lb.append(np.quantile(pred_thetap[-1], 0.025, 0))
                pred_thetap_quant_ub.append(np.quantile(pred_thetap[-1], 0.975, 0))
                edots.append(self.setup.models[i].edots[j])
                temps.append(self.setup.models[i].temps[j])
                real_strain.append(self.setup.models[i].meas_strain_histories[j])
                real_stress.append(self.setup.ys[i][self.setup.s2_ind[i] == j])

        rows = zip(
            real_strain, real_stress, pred_theta_quant_lb, pred_theta_quant_ub,
            pred_theta0_quant_lb, pred_theta0_quant_ub, pred_thetap_quant_lb, pred_thetap_quant_ub,
            edots, temps,
            )
        keys = [
            'actual_x','actual_y','pred_y_lb', 'pred_y_ub','hier_y_lb', 
            'hier_y_ub','pnew_y_lb','pnew_y_ub','edot','temp',
            ]
        plot_param_list = [dict(zip(keys,row)) for row in rows]

        pdf = PdfPages(path)
        for plot_params in plot_param_list:
            self.ptw_prediction_plot_single(**plot_params, pdf = pdf)
        pdf.close()
        return

    def ptw_prediction_plots(self, path):
        """ PTW Prediction Plots against model """
        if (type(self.out) is impala.OutCalibPool):
            return self.ptw_prediction_plots_pool(path)
        elif (type(self.out) is impala.OutCalibHier):
            return self.ptw_prediction_plots_hier(path)
        elif (type(self.out) is impala.OutCalibClust):
            return self.ptw_prediction_plots_cluster(path)
        else:
            raise ValueError('Improper out type')
        return

    @staticmethod
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

    def pairwise_theta_plot_hier(self, path = None):
        """ Pairwise Theta scatterplot """
        sel = np.arange(20000, self.setup.nmcmc, 10)
        theta_parent = impala.chol_sample_1per_constraints(
            self.out.theta0[sel,0], self.out.Sigma0[sel,0], self.setup.checkConstraints,
            self.setup.bounds_mat, self.setup.bounds.keys(), self.setup.bounds,
            )
        plt.figure(figsize = (15,15))
        for i in range(self.setup.p):
            for j in range(self.setup.p):
                if i == j:
                    plt.subplot2grid((self.setup.p, self.setup.p), (i,j))
                    for k in range(self.setup.nexp):
                        for s in range(self.setup.ntheta[k]):
                            sns.distplot(impala.invprobit(self.out.theta[k][sel,0,s,i]), 
                                            hist = False, kde = True, color = 'lightgreen')
                    sns.distplot(impala.invprobit(self.out.theta0[sel, 0, i]), hist = False, kde = True, color = 'blue')
                    sns.distplot(impala.invprobit(theta_parent[:,i]), hist = False, kde = True, color = 'grey')
                    plt.xlim(0,1)
                    ax = plt.gca()
                    ax.axes.yaxis.set_visible(False)
                    ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
                    plt.setp(ax.get_xticklabels(), rotation = 30, horizontalalignment = 'right')
                elif i < j:
                    plt.subplot2grid((self.setup.p, self.setup.p), (i,j))
                    for k in range(self.setup.nexp):
                        for s in range(self.setup.ntheta[k]):
                            contour = self.kde_contour(impala.invprobit(self.out.theta[k][sel, 0, s, j]),
                                                        impala.invprobit(self.out.theta[k][sel, 0, s, i]), 0.9)
                            plt.contour(contour['X'], contour['Y'], contour['Z'], contour['conts'], colors = 'lightgreen')
                    contour = self.kde_contour(impala.invprobit(self.out.theta0[sel, 0, j]), 
                                                impala.invprobit(self.out.theta0[sel, 0, i]), 0.9)
                    plt.contour(contour['X'], contour['Y'], contour['Z'], contour['conts'], colors = 'blue')

                    contour = self.kde_contour(impala.invprobit(theta_parent[:, j]), 
                                                impala.invprobit(theta_parent[:, i]), 0.9)
                    plt.contour(contour['X'], contour['Y'], contour['Z'], contour['conts'], colors = 'grey')
                    plt.xlim(0,1)
                    plt.ylim(0,1)
                    ax = plt.gca()
                    ax.axes.xaxis.set_visible(False)
                    ax.axes.yaxis.set_visible(False)
                else:
                    pass
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.subplot2grid((self.setup.p, self.setup.p), (2, 0))
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

    def pairwise_theta_plot_pool(self, path = None):
        sel = np.arange(20000, self.setup.nmcmc, 10)
        plt.figure(figsize = (15,15))
        for i in range(self.setup.p):
            for j in range(self.setup.p):
                if i == j:
                    plt.subplot2grid((self.setup.p, self.setup.p), (i,j))
                    sns.distplot(impala.invprobit(self.out.theta[sel,0,i]), hist = False, kde = True, color = 'blue')
                    plt.xlim(0,1)
                    ax = plt.gca()
                    ax.axes.yaxis.set_visible(False)
                    ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
                    plt.setp(ax.get_xticklabels(), rotation = 30, horizontalalignment = 'right')
                elif i < j:
                    plt.subplot2grid((self.setup.p, self.setup.p), (i,j))
                    contour = self.kde_contour(impala.invprobit(self.out.theta[sel,0,j]),
                                                impala.invprobit(self.out.theta[sel,0,i]), 0.9)
                    plt.contour(contour['X'], contour['Y'], contour['Z'], contour['conts'], colors = 'blue')
                    plt.xlim(0,1)
                    plt.ylim(0,1)
                    ax = plt.gca()
                    ax.axes.xaxis.set_visible(False)
                    ax.axes.yaxis.set_visible(False)
                else:
                    pass
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.axis('off')
        if path:
            plt.savefig(path, bbox_inches = 'tight')
        else:
            plt.show()
        return

    def pairwise_theta_plot_cluster(self, path = None):
        sel = np.arange(20000, self.setup.nmcmc, 10)
        thetas = self.out.theta[sel,0]
        deltas = [self.out.delta[i][sel] for i in range(self.setup.nexp)]
        nclustmax = max(self.out.delta[i].max() for i in range(self.setup.nexp)) + 1
        dcounts = np.zeros((sel.shape[0], nclustmax))
        for it, s in enumerate(sel):
            for i in range(self.setup.nexp):
                dcounts[it] += np.bincount(self.out.delta[i][s,0], minlength = nclustmax)
        etas = self.out.eta[sel, 0]
        nclust = (dcounts > 0).sum(axis = 1)
        prob   = dcounts + (dcounts == 0) * (etas / (nclustmax - nclust + 1e-9)).reshape(-1,1)
        prob  /= prob.sum(axis = 1).reshape(-1,1)
        cumprob = np.cumsum(prob, axis = 1)
        unif = uniform(size = (sel.shape[0],1))
        dnew = (unif > cumprob).sum(axis = 1)
        theta_parent = thetas[np.arange(sel.shape[0]), dnew]
        plt.figure(figsize = (15,15))
        for i in range(self.setup.p):
            for j in range(self.setup.p):
                if i == j:
                    plt.subplot2grid((self.setup.p, self.setup.p), (i,j))
                    for k in range(self.setup.nexp):
                        for s in range(self.setup.ns2[k]):
                            sns.distplot(impala.invprobit(self.out.theta_hist[k][sel,0,s,i]), 
                                            hist = False, kde = True, color = 'lightgreen')
                    sns.distplot(impala.invprobit(self.out.theta0[sel, 0, i]), hist = False, kde = True, color = 'blue')
                    sns.distplot(impala.invprobit(theta_parent[:,i]), hist = False, kde = True, color = 'grey')
                    plt.xlim(0,1)
                    ax = plt.gca()
                    ax.axes.yaxis.set_visible(False)
                    ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
                    plt.setp(ax.get_xticklabels(), rotation = 30, horizontalalignment = 'right')
                elif i < j:
                    plt.subplot2grid((self.setup.p, self.setup.p), (i,j))
                    for k in range(self.setup.nexp):
                        for s in range(self.setup.ns2[k]):
                            contour = self.kde_contour(impala.invprobit(self.out.theta_hist[k][sel, 0, s, j]),
                                                        impala.invprobit(self.out.theta_hist[k][sel, 0, s, i]), 0.9)
                            plt.contour(contour['X'], contour['Y'], contour['Z'], contour['conts'], colors = 'lightgreen')
                    contour = self.kde_contour(impala.invprobit(self.out.theta0[sel, 0, j]), 
                                                impala.invprobit(self.out.theta0[sel, 0, i]), 0.9)
                    plt.contour(contour['X'], contour['Y'], contour['Z'], contour['conts'], colors = 'blue')

                    contour = self.kde_contour(impala.invprobit(theta_parent[:, j]), 
                                                impala.invprobit(theta_parent[:, i]), 0.9)
                    plt.contour(contour['X'], contour['Y'], contour['Z'], contour['conts'], colors = 'grey')
                    plt.xlim(0,1)
                    plt.ylim(0,1)
                    ax = plt.gca()
                    ax.axes.xaxis.set_visible(False)
                    ax.axes.yaxis.set_visible(False)
                else:
                    pass
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.subplot2grid((self.setup.p, self.setup.p), (2, 0))
        colors = ['lightgreen','blue','grey']
        lines = [Line2D([0],[0],color = c, linewidth = 2) for c in colors]
        labels = [r'$\theta_i$',r'$\theta_0$',r'$\theta^*$']
        plt.legend(lines,labels)
        plt.axis('off')
        try:
            plt.subplot2grid((self.setup.p, self.setup.p), (4, 0))
            sns.distplot(nclust, kde = True, color = 'blue')
            plt.xlim(0,nclustmax)
        except IndexError:
            pass
        if path:
            plt.savefig(path, bbox_inches = 'tight')
        else:
            plt.show()
        pass

    def pairwise_theta_plot(self, path = None):
        if (type(self.out) is impala.OutCalibPool):
            return self.pairwise_theta_plot_pool(path)
        elif (type(self.out) is impala.OutCalibHier):
            return self.pairwise_theta_plot_hier(path)
        elif (type(self.out) is impala.OutCalibClust):
            return self.pairwise_theta_plot_cluster(path)
        else:
            raise ValueError('Improper out type')
        return
    
    @staticmethod
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
    
    def cluster_matrix_plot(self, path = None, **kwargs):
        cmat, breaks = self.cluster_matrix(self.out.delta, self.setup.ns2, self.out.nclustmax, **kwargs)
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
    
    def __init__(self, setup, out):
        """  """
        self.setup = setup
        self.out   = out
        return

    pass

if __name__ == '__main__':
    pass

# EOF
