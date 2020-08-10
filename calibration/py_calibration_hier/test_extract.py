from extractor import *
import sqlite3 as sql


if __name__ == '__main__':
    visualizer = Visualizer(HierarchicalExtractor('data_copper.db', 'copper_results.db'))
    visualizer.compute_curves(0,1)
    visualizer.import_curves()
    visualizer.get_plot_names()
    visualizer.make_plots()

# EOF
