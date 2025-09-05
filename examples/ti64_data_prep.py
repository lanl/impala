import glob
import os
import sqlite3 as sql

import fdasrsf as fs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBASS as pb
from scipy.interpolate.interpolate import interp1d

#########################################################################
## flyer plates


def movingaverage(interval, window_size=3):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "same")


## read in sims - may be faster ways, but this is general
def read_church_sims(path_sims):
    files_unsorted = os.listdir(path_sims)
    order = [
        int(str.split(ff, "_")[-1][0:4]) for ff in files_unsorted
    ]  # get first four characters after last underscore
    files = [
        x for _, x in sorted(zip(order, files_unsorted))
    ]  # sorted list of files
    nsims = len(files)
    files_full = [path_sims + "/" + files[i] for i in range(len(files))]
    dat_sims = [None] * nsims
    for i in range(nsims):
        with open(files_full[i]) as file:
            temp = file.readlines()
            dat_sims[i] = np.vstack([
                np.float_(str.split(temp[j], ",")) for j in range(2, len(temp))
            ])
    return dat_sims


## read in obs
def read_church_obs(path_obs):
    with open(path_obs) as file:
        temp = file.readlines()
        char0 = [
            temp[i][0] for i in range(len(temp))
        ]  # get first character of each line
        start_line = (
            np.where([(char0[i] != "#") for i in range(len(temp))])[0][0] + 1
        )  # find first line of data (last line that starts with # plus 2)
        dat_obs = np.vstack([
            np.float_(str.split(temp[j], ","))
            for j in range(start_line, len(temp))
        ])
    return dat_obs


## interpolate sims and obs to a grid, and possibly shift sims, smooth sims, scale obs
def snap2it(
    time_range,
    dat_sims,
    dat_obs,
    ntimes=200,
    move_start=False,
    start_cutoff=1e-4,
    start_exclude=0,
    smooth_avg_sims=1,
    obs_scale=1.0,
):
    ## new grid of times
    # time_range = [0.0, 0.9]
    # ntimes = 200
    time_grid = np.linspace(time_range[0], time_range[1], ntimes)
    nsims = len(dat_sims)
    if move_start:
        ## get jump off time for each sim (first time with smoothed vel > start_cutoff, excluding first start_exclude points), smoothed to make consistent with smoothing later
        # start_exclude = 200
        # start_cutoff = 1e-4
        sim_start_times = [0] * nsims
        for i in range(nsims):
            xx = movingaverage(dat_sims[i][:, 1], smooth_avg_sims)
            idx = (
                np.where(xx[start_exclude:] > start_cutoff)[0][0]
                + start_exclude
            )
            sim_start_times[i] = dat_sims[i][idx, 0]
    ## interpolate sims to be on new grid of times, and smooth
    sims_grid = np.empty([nsims, ntimes])
    for i in range(nsims):
        ifunc = interp1d(
            dat_sims[i][:, 0] - sim_start_times[i],
            movingaverage(dat_sims[i][:, 1], smooth_avg_sims),
            kind="cubic",
        )
        sims_grid[i, :] = ifunc(time_grid)
    ## interpolate obs to be on new grid of times, transform units, and possibly smooth
    ifunc = interp1d(
        dat_obs[:, 0],
        movingaverage(dat_obs[:, 1] / obs_scale, 1),
        kind="cubic",
        bounds_error=False,
        fill_value=0.0,
    )
    obs_grid = ifunc(time_grid)
    return {"sims": sims_grid, "obs": obs_grid, "time": time_grid}


## do warping
def warp(grid_dat, lam=0.01):
    out = fs.fdawarp(grid_dat["sims"].T, grid_dat["time"])
    out.multiple_align_functions(grid_dat["sims"][0], parallel=True, lam=lam)
    # out.multiple_align_functions(obs_all_list[j], parallel=True, lam=.001)
    gam_sim = out.gam
    vv_sim = fs.geometry.gam_to_v(out.gam)
    ftilde_sim = out.fn
    out2 = fs.pairwise_align_functions(
        grid_dat["sims"][0], grid_dat["obs"], grid_dat["time"], lam=lam
    )
    # out2 = fs.pairwise_align_functions(obs_all_list[j], obs_all_list[j], xx_all_list[j], lam=.01)
    gam_obs = out2[1]
    vv_obs = fs.geometry.gam_to_v(out2[1])
    ftilde_obs = out2[0]
    return {
        "gam_sim": gam_sim,
        "vv_sim": vv_sim,
        "ftilde_sim": ftilde_sim,
        "gam_obs": gam_obs,
        "vv_obs": vv_obs,
        "ftilde_obs": ftilde_obs,
    }


path_sims = os.path.abspath(
    "../../../Desktop/impala_data/Ti64.Flyer.NewAlpha/Ti64_Church2002_Manganin_591_612/Results/"
)
path_obs = os.path.abspath(
    "../../../git/impala/data/ti-6al-4v/Data/FlyerPlate/Church2002/Church2002_Fig2-300k-591m_s-12mm.csv"
)

dat_sims = read_church_sims(path_sims)
dat_obs = read_church_obs(path_obs)

## plot, shows time discrepancy, velocity unit differences
for i in range(len(dat_sims)):
    plt.plot(dat_sims[i][:, 0], dat_sims[i][:, 1])
plt.plot(dat_obs[:, 0], dat_obs[:, 1] / 100)
plt.show()

# put on same grid
dat_grid = snap2it(
    [0.0, 0.4],
    dat_sims,
    dat_obs,
    move_start=True,
    start_cutoff=1e-4,
    start_exclude=200,
    smooth_avg_sims=3,
    obs_scale=100,
)

plt.plot(dat_grid["time"], dat_grid["sims"].T)
plt.plot(dat_grid["time"], dat_grid["obs"], color="black")
plt.show()

# warp
dat_warp = warp(dat_grid, lam=0.001)

plt.plot(dat_warp["ftilde_sim"], "black")
plt.plot(dat_warp["ftilde_sim"][:, 0], "blue")
plt.plot(dat_warp["ftilde_obs"], "red")
plt.show()

plt.plot(dat_warp["gam_sim"])
plt.plot(dat_warp["gam_obs"], color="black")
plt.show()

plt.plot(dat_warp["vv_sim"])
plt.plot(dat_warp["vv_obs"], color="black")
plt.show()


def emu(inputs, dat_warp, ntest=20):
    ho = np.random.choice(dat_warp["ftilde_sim"].shape[1], ntest, replace=False)
    Xtrain = np.delete(inputs, ho, axis=0)
    Xtest = inputs[ho]

    ftilde_train = np.delete(dat_warp["ftilde_sim"], ho, axis=1)
    ftilde_test = dat_warp["ftilde_sim"][:, ho]
    vv_train = np.delete(dat_warp["vv_sim"], ho, axis=1)
    vv_test = dat_warp["vv_sim"][:, ho]
    gam_train = np.delete(dat_warp["gam_sim"], ho, axis=1)
    gam_test = dat_warp["gam_sim"][:, ho]

    emu_ftilde = pb.bassPCA(
        Xtrain,
        ftilde_train.T,
        ncores=15,
        npc=15,
        nmcmc=50000,
        nburn=40000,
        thin=10,
    )
    emu_vv = pb.bassPCA(
        Xtrain, vv_train.T, ncores=15, npc=9, nmcmc=50000, nburn=40000, thin=10
    )
    # emu_ftilde.plot()
    # emu_vv.plot()

    pred_ftilde = emu_ftilde.predict(Xtrain)
    predtest_ftilde = emu_ftilde.predict(Xtest)
    pred_vv = emu_vv.predict(Xtrain)
    pred_gam = fs.geometry.v_to_gam(pred_vv.mean(0).T)
    predtest_vv = emu_vv.predict(Xtest)
    predtest_gam = fs.geometry.v_to_gam(predtest_vv.mean(0).T)

    ftilde_resids = pred_ftilde.mean(0).T - ftilde_train
    ftilde_resids_test = predtest_ftilde.mean(0).T - ftilde_test

    gam_resids = pred_gam.mean(0).T - gam_train
    gam_resids_test = predtest_gam.mean(0).T - gam_test

    vv_resids = pred_vv.mean(0).T - vv_train
    vv_resids_test = predtest_vv.mean(0).T - vv_test

    return {
        "emu_ftilde": emu_ftilde,
        "emu_vv": emu_vv,
        "pred_ftilde": pred_ftilde,
        "predtest_ftilde": predtest_ftilde,
        "pred_vv": pred_vv,
        "pred_gam": pred_gam,
        "predtest_vv": predtest_vv,
        "predtest_gam": predtest_gam,
        "ftilde_resids": ftilde_resids,
        "ftilde_resids_test": ftilde_resids_test,
        "gam_resids": gam_resids,
        "gam_resids_test": gam_resids_test,
        "vv_resids": vv_resids,
        "vv_resids_test": vv_resids_test,
    }


sim_inputs = np.genfromtxt(
    "../../../Desktop/impala_data/Ti64.Flyer.NewAlpha/Ti64.design.ptw.1000.txt",
    skip_header=1,
)
in2 = pb.normalize(
    sim_inputs, np.array([np.min(sim_inputs, 0), np.max(sim_inputs, 0)]).T
)

emus = emu(in2, dat_warp, 100)

emus["emu_ftilde"].plot()
emus["emu_vv"].plot()
plt.plot(emus["ftilde_resids_test"], "r.")
plt.plot(emus["ftilde_resids"], color="black")
plt.show()
plt.plot(emus["vv_resids_test"], "r.")
plt.plot(emus["vv_resids"], "black")
plt.show()

ntest = 200
inputs = in2
ho = np.random.choice(dat_warp["ftilde_sim"].shape[1], ntest, replace=False)
Xtrain = np.delete(inputs, ho, axis=0)
Xtest = inputs[ho]

ftilde_train = np.delete(dat_warp["ftilde_sim"], ho, axis=1)
ftilde_test = dat_warp["ftilde_sim"][:, ho]
vv_train = np.delete(dat_warp["vv_sim"], ho, axis=1)
vv_test = dat_warp["vv_sim"][:, ho]
gam_train = np.delete(dat_warp["gam_sim"], ho, axis=1)
gam_test = dat_warp["gam_sim"][:, ho]

emu_ftilde = pb.bassPCA(
    in2,
    dat_warp["ftilde_sim"].T,
    ncores=15,
    npc=15,
    nmcmc=50000,
    nburn=40000,
    thin=10,
)

emu_ftilde.bm_list[0].plot()
yy = emu_ftilde.newy[0]
bmod = pb.bass(Xtrain, np.delete(yy, ho), nmcmc=50000, nburn=40000, thin=10)
bmod.plot()
plt.scatter(bmod.predict(Xtrain).mean(0), np.delete(yy, ho))
plt.show()
plt.scatter(bmod.predict(Xtest).mean(0), yy[ho])
plt.show()

bad = np.hstack((
    np.where(bmod.predict(Xtest).mean(0) > 0.1)[0],
    np.where(bmod.predict(Xtest).mean(0) < -0.07)[0],
))

plt.plot(ftilde_test, "black")
plt.plot(ftilde_test[:, bad], "red")
plt.show()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# this produces obs, sims, warped obs, warped sims, warping function obs, warping function sims, aligned emu, vv emu
# model will need obs and emus...do I need to save the rest of this?  If not, lets just dill the emus and obs (pretty sure you cant put them in sql table)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


########################################################################################################################
## Taylor cylinder data


def read_sims(path_sims):
    files_unsorted = os.listdir(path_sims)
    order = [
        int(str.split(ff, "_")[-1][0:4]) for ff in files_unsorted
    ]  # get first four characters after last underscore
    files = [
        x for _, x in sorted(zip(order, files_unsorted))
    ]  # sorted list of files
    nsims = len(files)
    files_full = [path_sims + "/" + files[i] for i in range(len(files))]
    dat_sims = [None] * nsims
    for i in range(nsims):
        with open(files_full[i]) as file:
            temp = file.readlines()
            dat_sims[i] = np.vstack([
                np.float_(str.split(temp[j], ",")) for j in range(2, len(temp))
            ])
    return dat_sims


## read in obs
def read_obs(path_obs):
    with open(path_obs) as file:
        temp = file.readlines()
        char0 = [
            temp[i][0] for i in range(len(temp))
        ]  # get first character of each line
        start_line = (
            np.where([(char0[i] != "#") for i in range(len(temp))])[0][0] + 1
        )  # find first line of data (last line that starts with # plus 2)
        dat_obs = np.vstack([
            np.float_(str.split(temp[j], ","))
            for j in range(start_line, len(temp))
        ])
    return dat_obs


def snap2it(dat_sims, dat_obs, nt=200, mm=0):
    # time_range = [0.0, 0.9]
    # ntimes = 200
    t_grid = np.linspace(0, 1, nt)
    nsims = len(dat_sims)
    height = [dat_sims[i][:, 1].max() for i in range(nsims)]
    ## interpolate sims to be on new grid of times, and smooth
    sims_grid = np.empty([nsims, nt])
    for i in range(nsims):
        idx = np.where(dat_sims[i][:, 0] > mm)[0]
        t_grid = np.linspace(
            dat_sims[i][idx, 1][0], dat_sims[i][idx, 1][-1], nt
        )
        ifunc = interp1d(
            dat_sims[i][idx, 1], dat_sims[i][idx, 0], kind="linear"
        )
        sims_grid[i, :] = ifunc(t_grid)
    ## interpolate obs to be on new grid of times, transform units, and possibly smooth
    ifunc = interp1d(
        dat_obs[:, 0],
        dat_obs[:, 1],
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    t_grid = np.linspace(dat_obs[0, 0], dat_obs[-1, 0], nt)
    obs_grid = ifunc(t_grid)
    obs_height = dat_obs[:, 0].max()
    return {
        "sims": sims_grid,
        "obs": obs_grid,
        "length_sims": height,
        "length_obs": obs_height,
    }


def warp(grid_dat, t_grid=np.linspace(0, 1, 400), lam=0.01):
    out = fs.fdawarp(grid_dat["sims"].T, t_grid)
    out.multiple_align_functions(grid_dat["sims"][0], parallel=True, lam=lam)
    # out.multiple_align_functions(obs_all_list[j], parallel=True, lam=.001)
    gam_sim = out.gam
    vv_sim = fs.geometry.gam_to_v(out.gam)
    ftilde_sim = out.fn
    out2 = fs.pairwise_align_functions(
        grid_dat["sims"][0], grid_dat["obs"], t_grid, lam=lam
    )
    # out2 = fs.pairwise_align_functions(obs_all_list[j], obs_all_list[j], xx_all_list[j], lam=.01)
    gam_obs = out2[1]
    vv_obs = fs.geometry.gam_to_v(out2[1])
    ftilde_obs = out2[0]
    return {
        "gam_sim": gam_sim,
        "vv_sim": vv_sim,
        "ftilde_sim": ftilde_sim,
        "gam_obs": gam_obs,
        "vv_obs": vv_obs,
        "ftilde_obs": ftilde_obs,
    }


sim_inputs = np.genfromtxt(
    "../../../Desktop/impala_data/Ti64.TaylorCyl.NewAlpha/Ti64.design.ptw.1000.txt",
    skip_header=1,
)


path_sims = os.path.abspath(
    "../../../Desktop/impala_data/Ti64.TaylorCyl.NewAlpha/taylor_mcd2007_fig4/Results/"
)
path_obs = os.path.abspath(
    "../../../git/impala/data/ti-6al-4v/Data/TaylorCyl/McDonald2007/McDonald2007_Fig4_Taylor.csv"
)

path_sims = os.path.abspath(
    "../../../Desktop/impala_data/Ti64.TaylorCyl.NewAlpha/taylor_mcd2007_fig5/Results/"
)
path_obs = os.path.abspath(
    "../../../git/impala/data/ti-6al-4v/Data/TaylorCyl/McDonald2007/McDonald2007_Fig5_Taylor.csv"
)

dat_sims = read_sims(path_sims)
dat_obs = read_obs(path_obs) / 10

for i in range(len(dat_sims)):
    plt.plot(dat_sims[i][:, 0], dat_sims[i][:, 1], "grey")
plt.plot(dat_obs[:, 1], dat_obs[:, 0], "black")
plt.show()

dat_snap = snap2it(dat_sims, dat_obs, 400, mm=dat_obs[-1, 1])
plt.plot(dat_snap["sims"].T)
plt.show()

# dat_warp = warp(dat_prep)
# plt.plot(dat_warp['ftilde_sim'])
# plt.show()


files_unsorted = os.listdir(path_sims)
run = [int(str.split(ff, "_")[-1][0:4]) for ff in files_unsorted]
inputs = sim_inputs[np.sort(run) - 1, :]  # exclude failed runs

ntest = 20
ho = np.random.choice(len(run), ntest, replace=False)
Xtrain = np.delete(inputs, ho, axis=0)
Xtest = inputs[ho]

mod_length = pb.bass(Xtrain, np.delete(dat_snap["length_sims"], ho))
mod_length.plot()
plt.scatter(
    mod_length.predict(Xtest).mean(0), np.array(dat_snap["length_sims"])[ho]
)
pb.abline(1, 0)
plt.show()

mod_profile = pb.bassPCA(
    Xtrain, np.delete(dat_snap["sims"], ho, axis=0), npc=15, ncores=10
)
mod_profile.plot()

pred_train = mod_profile.predict(Xtrain)
pred_test = mod_profile.predict(Xtest)
plt.plot(
    pred_train.mean(0).T - np.delete(dat_snap["sims"], ho, axis=0).T, "black"
)
plt.plot(pred_test.mean(0).T - dat_snap["sims"][ho].T, "red")
plt.show()

plt.scatter(mod_profile.predict(Xtest).mean(0), np.array(dat_snap["sims"])[ho])
pb.abline(1, 0)
plt.show()

##########################

path_sims = os.path.abspath(
    "../../../Desktop/impala_data/Ti64.TaylorCyl.NewAlpha/taylor_yu2011_T4/Results/"
)
dat_sims = read_sims(path_sims)
for i in range(len(dat_sims)):
    plt.plot(dat_sims[i][:, 0], dat_sims[i][:, 1])
plt.show()

length_sims = np.array([max(dat_sims[i][:, 1]) for i in range(len(dat_sims))])

files_unsorted = os.listdir(path_sims)
run = [int(str.split(ff, "_")[-1][0:4]) for ff in files_unsorted]
inputs = sim_inputs[np.sort(run) - 1, :]  # exclude failed runs

ntest = 20
ho = np.random.choice(len(run), ntest, replace=False)
Xtrain = np.delete(inputs, ho, axis=0)
Xtest = inputs[ho]

mod_length = pb.bass(Xtrain, np.delete(length_sims, ho))
mod_length.plot()
plt.scatter(mod_length.predict(Xtest).mean(0), length_sims[ho])
pb.abline(1, 0)
plt.show()

# Test             |      Deformed length (y-coordinate, cm)
# Yu2011_T1        |      2.4493
# Yu2011_T2        |      2.4095
# Yu2011_T3        |      2.3901
# Yu2011_T4        |      2.3702


##############################################################################################################################
def add_tc(path_sims, path_obs, xps):
    dat_sims = read_sims(path_sims)
    dat_obs = read_obs(path_obs) / 10
    dat_snap = snap2it(dat_sims, dat_obs, 400, mm=dat_obs[-1, 1])
    files_unsorted = os.listdir(path_sims)
    run = [int(str.split(ff, "_")[-1][0:4]) for ff in files_unsorted]
    inputs = pd.DataFrame(
        sim_inputs[np.sort(run) - 1, :]
    )  # exclude failed runs
    xps.append({
        "obs": pd.DataFrame(np.array(dat_snap["length_obs"]).reshape([1, 1])),
        "sim_inputs": inputs,
        "sim_outputs": pd.DataFrame(dat_snap["length_sims"]),
        "fname": path_sims,
    })
    xps.append({
        "obs": pd.DataFrame(dat_snap["obs"]),
        "sim_inputs": inputs,
        "sim_outputs": pd.DataFrame(dat_snap["sims"]),
        "fname": path_sims,
    })
    # return xps


def add_tc_length(path_sims, obs, xps):
    dat_sims = read_sims(path_sims)
    length_sims = np.array([
        max(dat_sims[i][:, 1]) for i in range(len(dat_sims))
    ])
    files_unsorted = os.listdir(path_sims)
    run = [int(str.split(ff, "_")[-1][0:4]) for ff in files_unsorted]
    inputs = sim_inputs[np.sort(run) - 1, :]  # exclude failed runs
    xps.append({
        "obs": pd.DataFrame(np.array(obs).reshape([1, 1])),
        "sim_inputs": pd.DataFrame(inputs),
        "sim_outputs": pd.DataFrame(length_sims),
        "fname": path_sims,
    })


########################################################################################################################
## SHPB data

# Meta Table Creation String;
#  Add additional meta information necessary to other experiment types to this
#  table.  Also need to add chemistry to this table.
meta_create = """CREATE TABLE meta (
    type TEXT,
    temperature REAL,
    edot REAL,
    emax REAL,
    pname TEXT,
    fname TEXT,
    sim_input REAL,
    sim_output REAL,
    obs REAL,
    table_name TEXT
    );
"""
shpb_data_create = """ CREATE TABLE {}(strain REAL, stress REAL); """
shpb_meta_insert = """ INSERT INTO
        meta(type, temperature, edot, emax, pname, fname, table_name)
        values (?,?,?,?,?,?,?);
        """
shpb_data_insert = """ INSERT INTO {}(strain, stress) values (?,?); """
sims_meta_insert = """ INSERT INTO meta(type, table_name, sim_input, sim_output) values (?,?,?,?); """


def load_Ti64_shpb(curs):
    global ti64_curr_id
    shpb_emax = 0

    base_path = "../../immpala/data/ti-6al-4v/Data/SHPB"
    papers = [os.path.split(f)[1] for f in glob.glob(base_path + "/*")]

    xps = []

    # Load the transports
    for paper in papers:
        files = glob.glob(os.path.join(base_path, paper, "*.Plast.csv"))
        for file in files:
            f = open(file)
            lines = f.readlines()
            temp = float(lines[1].split(" ")[2])  # Kelvin
            edot = float(lines[2].split(" ")[3]) * 1e-6  # 1/s
            pname = os.path.split(os.path.dirname(file))[1]
            fname = os.path.splitext(os.path.basename(file))[0]
            data = pd.read_csv(file, skiprows=range(0, 17)).values
            data[:, 1] = data[:, 1] * 1e-5  # Transform from MPa to Mbar
            xps.append({
                "data": data[1:],
                "temp": temp,
                "edot": edot,
                "pname": pname,
                "fname": fname,
            })
            # shpb_emax = max(shpb_emax, data[:,0].max())

    # truncate to first 40 for runtime
    # xps = xps[:40]

    for xp in xps:
        shpb_emax = max(shpb_emax, xp["data"][:, 0].max())

    # For each experiment:
    for xp in xps:
        # Set the ID
        ti64_curr_id += 1
        table_name = "data_{}".format(ti64_curr_id)
        # Create the relevant data table
        curs.execute(shpb_data_create.format(table_name))
        # Create the corresponding line in the meta table
        curs.execute(
            shpb_meta_insert,
            (
                "shpb",
                xp["temp"],
                xp["edot"],
                shpb_emax,
                xp["pname"],
                xp["fname"],
                table_name,
            ),
        )
        # Fill the data tables
        curs.executemany(
            shpb_data_insert.format(table_name), xp["data"].tolist()
        )
    return


global ti64_curr_id, ti64_shpb_emax
ti64_curr_id = 0
ti64_shpb_emax = 0

# Clear the old database
if os.path.exists("../../../Desktop/data_Ti64.db"):
    os.remove("../../../Desktop/data_Ti64.db")

# Start the SQLite connection
connection = sql.connect("../../../Desktop/data_Ti64.db")
cursor = connection.cursor()
cursor.execute(meta_create)
connection.commit()
# Load the data
current_id = 0
load_Ti64_shpb(cursor)


xps = []

path_sims = os.path.abspath(
    "../../../Desktop/impala_data/Ti64.TaylorCyl.NewAlpha/taylor_mcd2007_fig4/Results/"
)
path_obs = os.path.abspath(
    "../../../git/impala/data/ti-6al-4v/Data/TaylorCyl/McDonald2007/McDonald2007_Fig4_Taylor.csv"
)
add_tc(path_sims, path_obs, xps)

path_sims = os.path.abspath(
    "../../../Desktop/impala_data/Ti64.TaylorCyl.NewAlpha/taylor_mcd2007_fig5/Results/"
)
path_obs = os.path.abspath(
    "../../../git/impala/data/ti-6al-4v/Data/TaylorCyl/McDonald2007/McDonald2007_Fig5_Taylor.csv"
)
add_tc(path_sims, path_obs, xps)

path_sims = os.path.abspath(
    "../../../Desktop/impala_data/Ti64.TaylorCyl.NewAlpha/taylor_yu2011_T1/Results/"
)
add_tc_length(path_sims, 2.4493, xps)

path_sims = os.path.abspath(
    "../../../Desktop/impala_data/Ti64.TaylorCyl.NewAlpha/taylor_yu2011_T2/Results/"
)
add_tc_length(path_sims, 2.4095, xps)

path_sims = os.path.abspath(
    "../../../Desktop/impala_data/Ti64.TaylorCyl.NewAlpha/taylor_yu2011_T3/Results/"
)
add_tc_length(path_sims, 2.3901, xps)

path_sims = os.path.abspath(
    "../../../Desktop/impala_data/Ti64.TaylorCyl.NewAlpha/taylor_yu2011_T4/Results/"
)
add_tc_length(path_sims, 2.3702, xps)


for xp in xps:
    ti64_curr_id += 1
    table_name = "data_{}".format(ti64_curr_id)
    emu_iname = "sims_input_{}".format(ti64_curr_id)
    emu_oname = "sims_output_{}".format(ti64_curr_id)

    xp["obs"].to_sql(table_name, connection, index=False)
    xp["sim_inputs"].to_sql(emu_iname, connection, index=False)
    xp["sim_outputs"].to_sql(emu_oname, connection, index=False)
    cursor.execute(sims_meta_insert, ("tc", table_name, emu_iname, emu_oname))

connection.commit()


# load_Ti64_pca(cursor, connection)
# close the connection
connection.commit()
connection.close()
