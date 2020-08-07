from extractor import *
import sqlite3 as sql


if __name__ == '__main__':
    visualizer = Visualizer(HierarchicalExtractor('data_Ti64.db', 'Ti64_results.db'))
    visualizer.compute_curves(50000,20)
    visualizer.import_curves()
    visualizer.get_plot_names()
    visualizer.make_plots()

# EOF
