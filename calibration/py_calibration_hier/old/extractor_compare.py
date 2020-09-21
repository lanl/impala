from physical_models_c import MaterialModel
from multiprocessing import Pool
from itertools import repeat
import sqlite3 as sql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def compute_curve_summary(theta_indep, theta_shrink, theta0, field_names,
                        const, temp, emax, edot, nhist, models):
    model = MaterialModel(**models)
    model.set_history_variables(emax, edot, nhist)
    param = {x:y for x,y in zip(field_names, list(theta0[0]))}
    model.initialize(param, const, T = temp)

    stress_indep = np.empty((theta_indep.shape[0], nhist))
    for i in range(theta_indep.shape[0]):
        model.initialize_state(T = temp)
        model.update_parameters(theta_indep[i])
        res = model.compute_state_history()
        stress_indep[i] = res[:,2]
    strain = res[:,1]
    stress_indep_summary = np.quantile(stress_indep, (0.1, 0.5, 0.9), axis = 0)

    stress_shrink = np.empty((theta_shrink.shape[0], nhist))
    for i in range(theta_shrink.shape[0]):
        model.initialize_state(T = temp)
        model.update_parameters(theta_shrink[i])
        res = model.compute_state_history()
        stress_shrink[i] = res[:,2]
    stress_shrink_summary = np.quantile(stress_shrink, (0.1, 0.5, 0.9), axis = 0)

    stress0 = np.empty((theta0.shape[0], nhist))
    for i in range(theta0.shape[0]):
        model.initialize_state(T = temp)
        model.update_parameters(theta0[i])
        res = model.compute_state_history()
        stress0[i] = res[:,2]
    stress0_summary = np.quantile(stress0, (0.1, 0.5, 0.9), axis = 0)
    results = pd.DataFrame(
        np.vstack((strain, stress_indep_summary, stress_shrink_summary, stress0_summary)).T,
        columns = (
            'strain',
            'stress_indep_10','stress_indep_50','stress_indep_90',
            'stress_shrink_10','stress_shrink_50','stress_shrink_90',
            'stress0_10','stress0_50','stress0_90'
            ),
        )
    return results

def compute_curve_summary_wrapper(args):
    return compute_curve_summary(*args)

class HierarchicalExtractor(object):
    tbl_name_query = "SELECT tbl_name FROM sqlite_master WHERE type = 'table';"
    param_query = "PRAGMA TABLE_INFO({});"
    data_query = "SELECT {} FROM {};"
    meta_query = "SELECT table_name, temperature, edot, emax FROM meta;"
    model_query = "SELECT model_type, model_name FROM models;"

    def set_models(self):
        cursor = self.conn_o.cursor()
        self.models_used = {x : y for x, y in cursor.execute(self.model_query)}
        return

    def compute_curves(self, nburn, thin):
        self.theta0 = self.get_theta_shrink_table('theta0', nburn, thin)
        self.thetas_indep = self.get_thetas_indep(nburn, thin)
        self.thetas_shrink = self.get_thetas_shrink(nburn, thin)
        self.source = self.get_source_name()
        metas = self.get_meta()
        # Finding relevant meta-information for a given calibration (in order)
        self.metas = [
            metas[self.source[table]]
            for table
            in self.theta_tables
            ]
        temps = [row[0] for row in self.metas]
        edots = [row[1] for row in self.metas]
        emaxs = [row[2] for row in self.metas]
        # Building the iterator
        iterator = zip(
                self.thetas_indep, self.thetas_shrink, repeat(self.theta0), repeat(self.theta_fields),
                repeat(self.constants), temps, emaxs, edots, repeat(100),
                repeat(self.models_used)
                )
        pool = Pool(8)
        results = pool.map(compute_curve_summary_wrapper, iterator)
        pool.close()
        return results

    def set_table_list(self):
        cursor = self.conn_o.cursor()
        tables = [x[0] for x in cursor.execute(self.tbl_name_query)]
        self.theta_tables = [
            table
            for table in tables
            if table.startswith('theta_')
            ]
        self.sigma2_tables = [
            table
            for table in tables
            if table.startswith('sigma2_')
            ]
        return

    def parameter_list(self, table):
        cursor = self.conn_o.cursor()
        pragma = cursor.execute(self.param_query.format(table))
        fields = [x[1] for x in pragma]
        return fields

    def get_theta_shrink_table(self, table, nburn, thin):
        cursor = self.conn_o.cursor()
        result = cursor.execute(
            self.data_query.format(','.join(self.theta_fields), table)
            )
        return np.array(list(result))[nburn::thin]

    def get_theta_indep_table(self, table, nburn, thin):
        cursor = self.conn_o2.cursor()
        result = cursor.execute(
            self.data_query.format(','.join(self.theta_fields), table)
            )
        return np.array(list(result))[nburn::thin]

    def get_thetas_shrink(self, nburn, thin):
        result = [
            self.get_theta_shrink_table(table, nburn, thin)
            for table in self.theta_tables
            ]
        return result

    def get_thetas_indep(self, nburn, thin):
        result = [
            self.get_theta_indep_table(table, nburn, thin)
            for table in self.theta_tables
            ]
        return result

    def get_sigma2s(self, nburn, thin):
        cursor = self.conn_o.cursor()
        def f(table):
            res = list(cursor.execute(self.data_query.format('sigma2',table)))
            return (np.array(res)[nburn::thin]).flatten()

        result = np.vstack([f(table) for table in self.sigma2_tables]).T
        return result

    def get_data(self, nburn = 0, thin = 1):
        theta0         = self.get_theta_table('theta0', nburn, thin)
        thetas_shrink  = self.get_thetas_shrink(nburn, thin)
        thetas_indep   = self.get_thetas_indep(nburn, thin)
        sigma2s        = self.get_sigma2s(nburn, thin)
        return (theta0, thetas_shrink, thetas_indep, sigma2s)

    def get_meta(self):
        cursor = self.conn_i.cursor()
        res = {x[0] : (x[1],x[2],x[3]) for x in cursor.execute(self.meta_query)}
        return res

    def get_source_name(self):
        query = self.data_query.format('source_name, calib_name', 'meta')
        cursor = self.conn_o.cursor()
        table_dict = {y : x for x , y in cursor.execute(query)}
        return table_dict

    def set_constants(self):
        cursor = self.conn_o.cursor()
        self.constants = {
            x : y
            for x, y
            in cursor.execute("SELECT constant, value FROM constants;")
            }
        return

    def __init__(self, path_inputs, path_results_indep, path_results_shrink):
        self.conn_i = sql.connect(path_inputs)
        self.conn_o = sql.connect(path_results_shrink)
        self.conn_o2 = sql.connect(path_results_indep)
        self.set_table_list()
        self.set_constants()
        self.set_models()
        self.theta_fields = self.parameter_list('theta0')
        return

class Visualizer(object):
    curves = None
    thetas = None
    curve_query = "SELECT strain, stress FROM {};"

    def compute_curves(self, nburn, thin):
        self.calibrated_curves = self.extractor.compute_curves(nburn, thin)
        return

    def get_plot_names(self):
        cursor = self.extractor.conn_i.cursor()
        fname = {x : y for x, y in cursor.execute("SELECT table_name, fname from META;")}
        self.fnames = [fname[table] for table in self.table_order]
        return

    def import_curves(self):
        self.table_order = [
            self.extractor.source[table]
            for table in self.extractor.theta_tables
            ]
        cursor = self.extractor.conn_i.cursor()
        self.empirical_curves = [
            pd.DataFrame(
                np.array(list(cursor.execute(self.curve_query.format(table)))),
                columns = ('strain','stress'),
                )
            for table in self.table_order
            ]
        return

    def make_plots(self):
        b_alpha = 0.7; m_alpha = 0.9; linewidth = 0.5
        for i in range(len(self.calibrated_curves)):
            # plot predicted curves
            plt.plot(
                self.calibrated_curves[i].strain,
                self.calibrated_curves[i].stress_indep_10,
                alpha = b_alpha,
                color = 'blue',
                linestyle = '-',
                linewidth = linewidth
                )
            # plt.plot(
            #     self.calibrated_curves[i].strain,
            #     self.calibrated_curves[i].stress_indep_50,
            #     alpha = m_alpha,
            #     color = 'blue',
            #     linestyle = '-',
            #     linewidth = linewidth
            #     )
            plt.plot(
                self.calibrated_curves[i].strain,
                self.calibrated_curves[i].stress_indep_90,
                alpha = b_alpha,
                color = 'blue',
                linestyle = '-',
                linewidth = linewidth
                )

            plt.plot(
                self.calibrated_curves[i].strain,
                self.calibrated_curves[i].stress_shrink_10,
                alpha = b_alpha,
                color = 'green',
                linestyle = '-',
                linewidth = linewidth
                )
            # plt.plot(
            #     self.calibrated_curves[i].strain,
            #     self.calibrated_curves[i].stress_shrink_50,
            #     alpha = m_alpha,
            #     color = 'green',
            #     linestyle = '-',
            #     linewidth = linewidth
            #     )
            plt.plot(
                self.calibrated_curves[i].strain,
                self.calibrated_curves[i].stress_shrink_90,
                alpha = b_alpha,
                color = 'green',
                linestyle = '-',
                linewidth = linewidth
                )

            # plot hierarchical predicted curves
            plt.plot(
                self.calibrated_curves[i].strain,
                self.calibrated_curves[i].stress0_10,
                alpha = b_alpha,
                color = 'red',
                linestyle = '-',
                linewidth = linewidth
                )
            plt.plot(
                self.calibrated_curves[i].strain,
                self.calibrated_curves[i].stress0_50,
                alpha = m_alpha,
                color = 'red',
                linestyle = '-',
                linewidth = linewidth
                )
            plt.plot(
                self.calibrated_curves[i].strain,
                self.calibrated_curves[i].stress0_90,
                alpha = b_alpha,
                color = 'red',
                linestyle = '-',
                linewidth = linewidth
                )
            # plot empirical data
            plt.plot(
                self.empirical_curves[i].strain,
                self.empirical_curves[i].stress,
                'bo',
                )
            # compute ylim
            ylim = (
                min(self.empirical_curves[i].stress.min(),
                    self.calibrated_curves[i].values[:,1:].min()),
                max(self.empirical_curves[i].stress.max(),
                    self.calibrated_curves[i].values[:,1:].max())
                )
            plt.ylim(ylim)
            path = './output/{}.png'.format(self.fnames[i])
            if os.path.exists(path):
                os.remove(path)
            plt.savefig(path)
            plt.clf()

        return

    def __init__(self, extractor):
        self.extractor = extractor
        return

# EOF
