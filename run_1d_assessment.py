# run_1d_assessment.py
from config import BASIN_NAMES, BASIN_COLORS, SECTORS_DIR
from syn_smb import RACMOCatalog, multi_basin_run, plot_multibasin_psd, plot_multibasin_metrics, multibasin_metrics_table, plot_multibasin_split_test, plot_multibasin_smb

FIG_DIR = './figures/1D/'
catalog = RACMOCatalog(SECTORS_DIR)
results = multi_basin_run(catalog.paths(keys=BASIN_NAMES))
# print(results[BASIN_NAMES[0]][''])
plot_multibasin_smb(results, save_path=FIG_DIR + "fig_multibasin_smb.png")
# plot_multibasin_psd(results, save_path=FIG_DIR + "fig08_multibasin_psd.png")

# table = multibasin_metrics_table(results, experiment="baseline")
# plot_multibasin_metrics(results, save_path=FIG_DIR + "fig09_multibasin_metrics.png")

# plot_multibasin_split_test(results, save_path=FIG_DIR + "fig06_split_test.png")