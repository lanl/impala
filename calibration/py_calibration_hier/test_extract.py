from extractor import *
import sqlite3 as sql


copper_dbs = ('data_copper.db', 'copper_results.db')
Al5083_dbs = ('data_Al5083.db', 'Al5083_results.db')
Ti64_dbs   = ('data_Ti64.db',   'Ti64_results.db')

if __name__ == '__main__':
    visualizer = Visualizer(HierarchicalExtractor(*Al5083_dbs))
    visualizer.compute_curves(0,1)
    visualizer.import_curves()
    visualizer.get_plot_names()
    visualizer.make_plots()

# EOF
