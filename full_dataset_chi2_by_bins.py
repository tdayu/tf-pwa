import numpy as np
import ROOT as r
import argparse, os, copy, math
from scipy.spatial import KDTree

BP_MASS = 5.27934
JPSI_MASS = 3.096916
KST_MASS = 0.89555
PI_MASS = 0.13957
K_MASS = 0.493677
RHO_MASS = 0.77526
RHO_WIDTH = 0.1474

class Masses:
    def __init__(self, mpipi, mKpi, mKpipi, mJpsipi, mJpsipipi, mJpsiK, mJpsiKpi, mJpsipiminus):
        self.mpipi = mpipi
        self.mKpi = mKpi
        self.mKpipi = mKpipi
        self.mJpsipi = mJpsipi
        self.mJpsipipi = mJpsipipi
        self.mJpsiK = mJpsiK
        self.mJpsiKpi = mJpsiKpi
        self.mJpsipiminus = mJpsipiminus

class Bin2D:
    def __init__(self, xlow, ylow, xhigh, yhigh):
        self.xlow = xlow
        self.ylow = ylow
        self.xhigh = xhigh
        self.yhigh = yhigh

    def __repr__(self):
        return f"x : [{self.xlow}, {self.xhigh}] y : [{self.ylow}, {self.yhigh}]"

def get_invariant_masses(momenta):
    jpsi_PE = momenta[..., 0] + momenta[..., 4]
    jpsi_PX = momenta[..., 1] + momenta[..., 5]
    jpsi_PY = momenta[..., 2] + momenta[..., 6]
    jpsi_PZ = momenta[..., 3] + momenta[..., 7]

    K_PE = momenta[...,8]
    K_PX = momenta[...,9]
    K_PY = momenta[...,10]
    K_PZ = momenta[...,11]

    Pim_PE = momenta[...,12]
    Pim_PX = momenta[...,13]
    Pim_PY = momenta[...,14]
    Pim_PZ = momenta[...,15]

    Pip_PE = momenta[...,16]
    Pip_PX = momenta[...,17]
    Pip_PY = momenta[...,18]
    Pip_PZ = momenta[...,19]

    # Calculate the invariant masses
    mKpipi = np.sqrt(np.square(K_PE + Pim_PE + Pip_PE)
                     - np.square(K_PX + Pim_PX + Pip_PX)
                     - np.square(K_PY + Pim_PY + Pip_PY)
                     - np.square(K_PZ + Pim_PZ + Pip_PZ))


    mKpi = np.sqrt(np.square(K_PE + Pim_PE)
                   - np.square(K_PX + Pim_PX)
                   - np.square(K_PY + Pim_PY)
                   - np.square(K_PZ + Pim_PZ))


    mpipi = np.sqrt(np.square(Pim_PE + Pip_PE)
                    - np.square(Pim_PX + Pip_PX)
                    - np.square(Pim_PY + Pip_PY)
                    - np.square(Pim_PZ + Pip_PZ))

    mJpsipi = np.sqrt(np.square(jpsi_PE + Pip_PE)
                     - np.square(jpsi_PX + Pip_PX)
                     - np.square(jpsi_PY + Pip_PY)
                     - np.square(jpsi_PZ + Pip_PZ))

    mJpsipipi = np.sqrt(np.square(jpsi_PE + Pip_PE + Pim_PE)
                        - np.square(jpsi_PX + Pip_PX + Pim_PX)
                        - np.square(jpsi_PY + Pip_PY + Pim_PY)
                        - np.square(jpsi_PZ + Pip_PZ + Pim_PZ))

    mJpsiK = np.sqrt(np.square(jpsi_PE + K_PE)
                     - np.square(jpsi_PX + K_PX)
                     - np.square(jpsi_PY + K_PY)
                     - np.square(jpsi_PZ + K_PZ))

    mJpsiKpi = np.sqrt(np.square(jpsi_PE + K_PE + Pim_PE)
                       - np.square(jpsi_PX + K_PX + Pim_PX)
                       - np.square(jpsi_PY + K_PY + Pim_PY)
                       - np.square(jpsi_PZ + K_PZ + Pim_PZ))

    mJpsipiminus = np.sqrt(np.square(jpsi_PE + Pim_PE)
                           - np.square(jpsi_PX + Pim_PX)
                           - np.square(jpsi_PY + Pim_PY)
                           - np.square(jpsi_PZ + Pim_PZ))

    return Masses(mpipi, mKpi, mKpipi, mJpsipi, mJpsipipi, mJpsiK, mJpsiKpi, mJpsipiminus)

def find_isobin_boundaries(x : np.array, lower : float, upper : float, nbins : int):
    assert(np.amin(x) > lower)
    assert(np.amax(x) < upper)

    events_per_bin = x.shape[0] // nbins
    remainder = x.shape[0] % nbins
    events_in_bin = [events_per_bin + 1 if i < remainder else events_per_bin for i in range(nbins - 1)]
    x = np.sort(x)
    boundary_indices = np.cumsum(events_in_bin)

    boundaries = [x[index] for index in boundary_indices]
    boundaries.insert(0, lower)
    boundaries.append(upper)

    return np.array(boundaries)

def select_and_isobin(selection_variable : np.array, isobin_variable : np.array, 
                      selection_low : float, selection_high : float, 
                      isobin_low : float, isobin_high : float,
                      n_isobins : int,
                      isobin_is_x : bool):
    assert(selection_variable.shape == isobin_variable.shape)

    indices = np.logical_and(np.logical_and(selection_variable >= selection_low, selection_variable < selection_high),
                             np.logical_and(isobin_variable >= isobin_low, isobin_variable < isobin_high))
    isobin_boundaries = find_isobin_boundaries(isobin_variable[indices], isobin_low, isobin_high, n_isobins)

    if isobin_is_x:
        bins = [Bin2D(isobin_boundaries[i], selection_low, isobin_boundaries[i+1], selection_high) for i in range(n_isobins)]
    else:
        bins = [Bin2D(selection_low, isobin_boundaries[i], selection_high, isobin_boundaries[i+1]) for i in range(n_isobins)]

    return bins

def isobin_2D(first_dimension : np.array, second_dimension : np.array,
              first_low : float, first_high : float,
              second_low : float, second_high : float,
              nbins_first : int, nbins_second : int,
              first_is_x : bool):
    assert(first_dimension.shape[0] == second_dimension.shape[0])

    indices = np.logical_and(np.logical_and(first_dimension >= first_low, first_dimension < first_high),
                             np.logical_and(second_dimension >= second_low, second_dimension < second_high))
    first_dimension = first_dimension[indices]
    second_dimension = second_dimension[indices]

    bins = list()

    first_dimension_boundaries = find_isobin_boundaries(first_dimension, first_low, first_high, nbins_first)
    for i in range(nbins_first):
        indices = np.logical_and(first_dimension >= first_dimension_boundaries[i], first_dimension < first_dimension_boundaries[i+1])
        second_dimension_boundaries = find_isobin_boundaries(second_dimension[indices], second_low, second_high, nbins_second)

        for j in range(nbins_second):
            x_low, x_high, y_low, y_high = (first_dimension_boundaries[i], first_dimension_boundaries[i+1], second_dimension_boundaries[j], second_dimension_boundaries[j+1]) \
                                           if first_is_x else \
                                           (second_dimension_boundaries[j], second_dimension_boundaries[j+1], first_dimension_boundaries[i], first_dimension_boundaries[i+1])
            bins.append(Bin2D(x_low, y_low, x_high, y_high))

    return bins

def get_kdtree_binnings(first_dimension : np.array, second_dimension : np.array,
                        first_low : float, first_high : float,
                        second_low : float, second_high : float,
                        leafsize : int):

    def recurse(node, node_binning):
        if hasattr(node, 'split'):
            left_tree = node.less
            right_tree = node.greater
            left_binning = copy.deepcopy(node_binning)
            right_binning = copy.deepcopy(node_binning)
            if node.split_dim == 0:
                left_binning.xhigh = node.split
                right_binning.xlow = node.split
            else:
                left_binning.yhigh = node.split
                right_binning.ylow = node.split
            left_binnings = recurse(left_tree, left_binning)
            right_binnings = recurse(right_tree, right_binning)
            binnings = left_binnings + right_binnings
        else:
            binnings = [node_binning]
        return binnings

    indices_first = np.logical_and(first_dimension > first_low, first_dimension < first_high)
    indices_second = np.logical_and(second_dimension > second_low, second_dimension < second_high)
    indices = np.logical_and(indices_first, indices_second)

    kdtree_data = np.stack((first_dimension[indices], second_dimension[indices]), axis=-1)
    kdtree = KDTree(kdtree_data, leafsize)

    source_bin = Bin2D(first_low, second_low, first_high, second_high)
    binnings = recurse(kdtree.tree, source_bin)

    return binnings

def mJpsipipi_mKpipi_bins(masses):
    bins = list()

    bins.extend(select_and_isobin(masses.mKpipi, masses.mJpsipipi, 0.75, 0.95, 3.3, 4.8, 2, False))
    bins.extend(select_and_isobin(masses.mKpipi, masses.mJpsipipi, 0.95, 1.0, 3.3, 4.8, 2, False))
    bins.extend(select_and_isobin(masses.mKpipi, masses.mJpsipipi, 1.0, 1.05, 3.3, 4.8, 4, False))
    bins.extend(select_and_isobin(masses.mKpipi, masses.mJpsipipi, 1.05, 1.1, 3.3, 4.8, 6, False))

    bins.extend(select_and_isobin(masses.mJpsipipi, masses.mKpipi, 3.3, 3.7, 1.1, 2.2, 3, True))
    bins.extend(select_and_isobin(masses.mJpsipipi, masses.mKpipi, 3.7, 3.823, 1.1, 2.2, 8, True))
    bins.extend(select_and_isobin(masses.mJpsipipi, masses.mKpipi, 3.823, 3.872, 1.1, 2.2, 4, True))
    bins.extend(get_kdtree_binnings(masses.mKpipi, masses.mJpsipipi, 1.1, 2.2, 3.872, 4.8, 1000))
    # bins.extend(isobin_2D(masses.mJpsipipi, masses.mKpipi, 3.872, 4.8, 1.1, 2.2, 40, 40, False))

    return bins

def mJpsipipi_mpipi_bins(masses):
    bins = list()

    bins.extend(select_and_isobin(masses.mJpsipipi, masses.mpipi, 3.3, 3.7, 0.25, 1.7, 3, True))
    bins.extend(select_and_isobin(masses.mJpsipipi, masses.mpipi, 3.7, 3.823, 0.25, 1.7, 8, True))
    bins.extend(select_and_isobin(masses.mJpsipipi, masses.mpipi, 3.823, 3.872, 0.25, 1.7, 4, True))
    bins.extend(get_kdtree_binnings(masses.mpipi, masses.mJpsipipi, 0.25, 1.7, 3.872, 4.8, 1000))
    # bins.extend(isobin_2D(masses.mJpsipipi, masses.mpipi, 3.872, 4.8, 0.25, 1.7, 40, 40, False))

    return bins

def mKpipi_mpipi_bins(masses):
    bins = list()

    bins.extend(select_and_isobin(masses.mKpipi, masses.mpipi, 0.75, 0.95, 0.25, 1.7, 2, True))
    bins.extend(select_and_isobin(masses.mKpipi, masses.mpipi, 0.95, 1.0, 0.25, 1.7, 2, True))
    bins.extend(select_and_isobin(masses.mKpipi, masses.mpipi, 1.0, 1.05, 0.25, 1.7, 4, True))
    bins.extend(select_and_isobin(masses.mKpipi, masses.mpipi, 1.05, 1.1, 0.25, 1.7, 6, True))
    bins.extend(get_kdtree_binnings(masses.mpipi, masses.mKpipi, 0.25, 1.7, 1.1, 2.2, 1000))
    # bins.extend(isobin_2D(masses.mKpipi, masses.mpipi, 1.1, 2.2, 0.25, 1.7, 40, 40, False))

    return bins

def mKpipi_mKpi_bins(masses):
    bins = list()

    bins.extend(select_and_isobin(masses.mKpipi, masses.mKpi, 0.75, 0.95, 0.65, 2.1, 2, True))
    bins.extend(select_and_isobin(masses.mKpipi, masses.mKpi, 0.95, 1.0, 0.65, 2.1, 2, True))
    bins.extend(select_and_isobin(masses.mKpipi, masses.mKpi, 1.0, 1.05, 0.65, 2.1, 4, True))
    bins.extend(select_and_isobin(masses.mKpipi, masses.mKpi, 1.05, 1.1, 0.65, 2.1, 6, True))
    bins.extend(get_kdtree_binnings(masses.mKpi, masses.mKpipi, 0.65, 2.1, 1.1, 2.2, 1000))
    # bins.extend(isobin_2D(masses.mKpipi, masses.mKpi, 1.1, 2.2, 0.65, 2.1, 40, 40, False))

    return bins

def mJpsiK_mpipi_bins(masses):
    bins = get_kdtree_binnings(masses.mpipi, masses.mJpsiK, 0.25, 1.7, 3.5, 5.0, 1000)
    # bins = isobin_2D(masses.mJpsiK, masses.mpipi, 3.5, 5.0, 0.25, 1.7, 40, 40, False)
    return bins

def mJpsipi_mKpi_bins(masses):
    bins = get_kdtree_binnings(masses.mKpi, masses.mJpsipi, 0.65, 2.1, 3.2, 4.7, 1000)
    # bins = isobin_2D(masses.mJpsipi, masses.mKpi, 3.2, 4.7, 0.65, 2.1, 40, 40, False)
    return bins

def mJpsiKpi_mKpi_bins(masses):
    bins = get_kdtree_binnings(masses.mKpi, masses.mJpsiKpi, 0.65, 2.1, 3.7, 5.2, 1000)
    # bins = isobin_2D(masses.mJpsiKpi, masses.mKpi, 3.7, 5.2, 0.65, 2.1, 40, 40, False)
    return bins

def find_indices(x, y, binning):
    indices_x = np.logical_and(x > binning.xlow, x < binning.xhigh)
    indices_y = np.logical_and(y > binning.ylow, y < binning.yhigh)
    indices = np.logical_and(indices_x, indices_y)

    return indices

def plot_chi2(output_path, xlabel, ylabel, 
              title, bins, 
              x_data, y_data, weights_data,
              x_phase_space, y_phase_space, weights_phase_space,
              xlow, xhigh, ylow, yhigh):

    # Change the boundaries of the bins to reflect data
    for binning in bins:
        x_band_data = np.logical_and(y_data > binning.ylow, y_data < binning.yhigh)
        x_band_phsp = np.logical_and(y_phase_space > binning.ylow, y_phase_space < binning.yhigh)
        y_band_data = np.logical_and(x_data > binning.xlow, x_data < binning.xhigh)
        y_band_phsp = np.logical_and(x_phase_space > binning.xlow, x_phase_space < binning.xhigh)

        x_band_points = np.concatenate((x_data[x_band_data], x_phase_space[x_band_phsp]))
        y_band_points = np.concatenate((y_data[y_band_data], y_phase_space[y_band_phsp]))

        if not np.any(x_band_points > binning.xhigh):
            indices_data = find_indices(x_data, y_data, binning)
            indices_phsp = find_indices(x_phase_space, y_phase_space, binning)
            xmax = max(np.amax(x_data[indices_data]), np.amax(x_phase_space[indices_phsp]))
            binning.xhigh = min(xhigh, xmax + 0.01 * (xmax - binning.xlow))
        elif not np.any(x_band_points < binning.xlow):
            indices_data = find_indices(x_data, y_data, binning)
            indices_phsp = find_indices(x_phase_space, y_phase_space, binning)
            xmin = min(np.amin(x_data[indices_data]), np.amin(x_phase_space[indices_phsp]))
            binning.xlow = max(xlow, xmin - 0.01 * (binning.xhigh - xmin))

        if not np.any(y_band_points < binning.ylow):
            indices_data = find_indices(x_data, y_data, binning)
            indices_phsp = find_indices(x_phase_space, y_phase_space, binning)
            ymin = min(np.amin(y_data[indices_data]), np.amin(y_phase_space[indices_phsp]))
            binning.ylow = max(ylow, ymin - 0.01 * (binning.yhigh - ymin))
        elif not np.any(y_band_points > binning.yhigh):
            indices_data = find_indices(x_data, y_data, binning)
            indices_phsp = find_indices(x_phase_space, y_phase_space, binning)
            ymax = max(np.amax(y_data[indices_data]), np.amax(y_phase_space[indices_phsp]))
            binning.yhigh = min(yhigh, ymax + 0.01 * (ymax - binning.ylow))

    histogram = r.TH2Poly("hist_data", "", xlow, xhigh, ylow, yhigh)
    model_histogram =  r.TH2Poly("hist_model", "", xlow, xhigh, ylow, yhigh)

    for binning in bins:
        histogram.AddBin(binning.xlow, binning.ylow, binning.xhigh, binning.yhigh)
        model_histogram.AddBin(binning.xlow, binning.ylow, binning.xhigh, binning.yhigh)

    histogram.Sumw2()
    model_histogram.Sumw2()

    histogram.FillN(weights.shape[0], x_data.data, y_data.data, weights.data)
    model_histogram.FillN(weights_phase_space.shape[0], x_phase_space.data, y_phase_space.data, weights_phase_space.data)
    histogram.Add(model_histogram, -1)
    chi2 = 0
    for i in range(1, histogram.GetNumberOfBins()+1):
        # chi2 = np.sign(histogram.GetBinContent(i))*histogram.GetBinContent(i)*histogram.GetBinContent(i)/(histogram.GetBinError(i)*histogram.GetBinError(i))
        content = histogram.GetBinContent(i)/histogram.GetBinError(i)
        chi2 += content * content
        histogram.SetBinContent(i, max(min(content, 5), -5))
        # histogram.SetBinContent(i, min(chi2, 10))

    histogram.SetMaximum(5)
    histogram.SetMinimum(-5)
    histogram.SetStats(False)
    histogram.GetXaxis().SetTitle(xlabel)
    histogram.GetYaxis().SetTitle(ylabel)
    histogram.SetMarkerSize(0)
    histogram.SetLineWidth(1)
    histogram.SetLineColorAlpha(r.kBlack, 0.3)
    canvas = r.TCanvas("canvas", "canvas", 1200, 1200)
    histogram.Draw("colz")

    indices = np.random.uniform(0, 1, size=weights_data.size) < 0.04
    x_scatter = x_data[indices].astype(np.float64)
    y_scatter = y_data[indices].astype(np.float64)

    scatter_plot = r.TGraph(int(np.sum(indices)), x_scatter.data, y_scatter.data)
    scatter_plot.Draw("P;same")
    histogram.Draw("colz;L;same")

    canvas.Update()
    canvas.SaveAs(output_path)
    canvas.Clear()
    canvas.Close()

    print(histogram.Integral(), chi2, histogram.GetNumberOfBins(), histogram.GetNumberOfBins() - 318)

    del histogram
    del model_histogram
    del scatter_plot
    del canvas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to plot 2D chi2 of fit.")
    parser.add_argument("phase_space_weights", type=str, help="Path to phase space weights npy file.")
    parser.add_argument("output_dir", type=str, help="Output directory to dump plots.")
    arguments = parser.parse_args()

    os.makedirs(arguments.output_dir, exist_ok=True)

    momenta = np.load("/home/dtou/Software/tf-pwa/workspace/data/June2022_data/data.npy")
    weights = np.load("/home/dtou/Software/tf-pwa/workspace/data/June2022_data/data_weight.npy")

    # mc_momenta = np.load("/home/dtou/Software/tf-pwa/workspace/data/June2022_data/mc.npy")
    mc_momenta = np.load("/home/dtou/Software/tf-pwa/workspace/data/June2022_data/B_JpsiK1.npy")
    mc_weights = np.load(arguments.phase_space_weights)

    masses = get_invariant_masses(momenta)
    mc_masses = get_invariant_masses(mc_momenta)

    r.gStyle.SetPalette(r.kRainBow, 0, 0.5)

    # split into 4 bins of mKpipi, mKpi vs mpipi
    mKpipi_boundaries = [0., 1.35, 1.55, 1.8, 2.3]
    titles = ["m(K#pi#pi) < 1.35 GeV", "1.35 < m(K#pi#pi) < 1.55", "1.55 < m(K#pi#pi) < 1.8", "m(K#pi#pi) > 1.8"]
    for i in range(4):
        indices = np.logical_and(masses.mKpipi > mKpipi_boundaries[i], masses.mKpipi < mKpipi_boundaries[i+1])
        mc_indices = np.logical_and(mc_masses.mKpipi > mKpipi_boundaries[i], mc_masses.mKpipi < mKpipi_boundaries[i+1])
        mpipi_limit = math.ceil((mKpipi_boundaries[i+1] - K_MASS) * 20) / 20
        mKpi_limit = math.ceil((mKpipi_boundaries[i+1] - PI_MASS) * 20) / 20

        binnings = get_kdtree_binnings(masses.mpipi[indices], masses.mKpi[indices], 0.25, mpipi_limit, 0.6, mKpi_limit, 500)

        plot_chi2(os.path.join(arguments.output_dir, f"mKpipi_bin{i}_mKpi_mpipi.pdf"), "m(#pi#pi)", "m(K#pi)", 
                  titles[i], binnings, 
                  masses.mpipi[indices], masses.mKpi[indices], weights[indices],
                  mc_masses.mpipi[mc_indices], mc_masses.mKpi[mc_indices], mc_weights[mc_indices],
                  0.25, mpipi_limit, 0.6, mKpi_limit)

    # split into 3-4 bins of mKpi, mKpipi vs mJpsipi, mKpipi vs mJpsiKpi
    mKpi_boundaries = [K_MASS + PI_MASS, 0.84555, 0.94555, 1.2, 2.1]
    titles = ["m(K#pi) < m_{K^{*}} - 0.05 GeV", "|m(K#pi) - m_{K^{*}}| < 0.05 GeV", "m_{K^{*}} + 0.05 < m(K#pi) < 1.2", "m(K#pi) > 1.2"]
    rho_veto = np.logical_or(masses.mpipi < RHO_MASS - RHO_WIDTH, masses.mpipi > RHO_MASS + RHO_WIDTH)
    mc_rho_veto = np.logical_or(mc_masses.mpipi < RHO_MASS - RHO_WIDTH, mc_masses.mpipi > RHO_MASS + RHO_WIDTH)
    for i in range(4):
        indices = np.logical_and(masses.mKpi > mKpi_boundaries[i], masses.mKpi < mKpi_boundaries[i+1])
        indices = np.logical_and(indices, rho_veto)
        mc_indices = np.logical_and(mc_masses.mKpi > mKpi_boundaries[i], mc_masses.mKpi < mKpi_boundaries[i+1])
        mc_indices = np.logical_and(mc_indices, mc_rho_veto)
        mKpipi_limit = math.floor((mKpi_boundaries[i] + PI_MASS) * 20) / 20
        mJpsipi_limit = math.ceil((BP_MASS - mKpi_boundaries[i]) * 20) / 20

        binnings = get_kdtree_binnings(masses.mKpipi[indices], masses.mJpsipi[indices], mKpipi_limit, 2.2, 3.2, mJpsipi_limit, 250)

        plot_chi2(os.path.join(arguments.output_dir, f"mKpi_bin{i}_mJpsipi_mKpipi.pdf"), "m(K#pi#pi)", "m(J/#psi#pi)", 
                  titles[i], binnings, 
                  masses.mKpipi[indices], masses.mJpsipi[indices], weights[indices],
                  mc_masses.mKpipi[mc_indices], mc_masses.mJpsipi[mc_indices], mc_weights[mc_indices],
                  mKpipi_limit, 2.2, 3.2, mJpsipi_limit)

    # split into 3-4 bins of mpipi, mKpipi vs mJpsipipi, mKpipi vs mJpsiK
    mpipi_boundaries = [2 * PI_MASS, 0.62786, 0.75662, 0.92266, 2.0]
    titles = ["m(#pi#pi) < m_{#rho} - #Gamma_{#rho}", "m_{#rho} - #Gamma_{#rho} < m(#pi#pi) < m_{#omega} - 3#Gamma_{#omega}", "m_{#omega} - 3#Gamma_{#omega} < m(#pi#pi) < m_{#rho} + #Gamma_{#rho}", "m(#pi#pi) > m_{#rho} + #Gamma_{#rho}"]
    Kst_veto = np.logical_or(masses.mKpi < KST_MASS - 0.1, masses.mKpi > KST_MASS + 0.1)
    mc_Kst_veto = np.logical_or(mc_masses.mKpi < KST_MASS - 0.1, mc_masses.mKpi > KST_MASS + 0.1)
    for i in range(4):
        indices = np.logical_and(masses.mpipi > mpipi_boundaries[i], masses.mpipi < mpipi_boundaries[i+1])
        indices = np.logical_and(indices, Kst_veto)
        mc_indices = np.logical_and(mc_masses.mpipi > mpipi_boundaries[i], mc_masses.mpipi < mpipi_boundaries[i+1])
        mc_indices = np.logical_and(mc_indices, mc_Kst_veto)
        mKpipi_limit = math.floor((mpipi_boundaries[i] + K_MASS) * 20) / 20
        mJpsipipi_limit = math.floor((mpipi_boundaries[i] + JPSI_MASS) * 20) / 20

        binnings = get_kdtree_binnings(masses.mKpipi[indices], masses.mJpsipipi[indices], mKpipi_limit, 2.2, mJpsipipi_limit, 4.8, 250)

        plot_chi2(os.path.join(arguments.output_dir, f"mpipi_bin{i}_mJpsipipi_mKpipi.pdf"), "m(K#pi#pi)", "m(J/#psi#pi)", 
                  titles[i], binnings, 
                  masses.mKpipi[indices], masses.mJpsipipi[indices], weights[indices],
                  mc_masses.mKpipi[mc_indices], mc_masses.mJpsipipi[mc_indices], mc_weights[mc_indices],
                  mKpipi_limit, 2.2, mJpsipipi_limit, 4.8)

    # f_0_980  special_bin = f_0_980 +/- 60
    indices = np.logical_and(masses.mpipi > 0.92, masses.mpipi < 1.04)
    mc_indices = np.logical_and(mc_masses.mpipi > 0.92, mc_masses.mpipi < 1.04)
    mKpipi_limit = math.floor((0.92 + K_MASS) * 20) / 20
    mJpsipipi_limit = math.floor((0.92 + JPSI_MASS) * 20) / 20

    binnings = get_kdtree_binnings(masses.mKpipi[indices], masses.mJpsipipi[indices], mKpipi_limit, 2.2, mJpsipipi_limit, 4.8, 250)

    plot_chi2(os.path.join(arguments.output_dir, f"f_0_980_mJpsipipi_mKpipi.pdf"), "m(K#pi#pi)", "m(J/#psi#pi)", 
              titles[i], binnings, 
              masses.mKpipi[indices], masses.mJpsipipi[indices], weights[indices],
              mc_masses.mKpipi[mc_indices], mc_masses.mJpsipipi[mc_indices], mc_weights[mc_indices],
              mKpipi_limit, 2.2, mJpsipipi_limit, 4.8)

    # plot_chi2(os.path.join(arguments.output_dir, "mJpsipipi_mKpipi.pdf"), "m(K#pi#pi)", "m(J/#psi#pi#pi)", 
    #           mJpsipipi_mKpipi_bins(masses), 
    #           masses.mKpipi, masses.mJpsipipi, weights,
    #           mc_masses.mKpipi, mc_masses.mJpsipipi, mc_weights,
    #           0.75, 2.2, 3.3, 4.8)

    # plot_chi2(os.path.join(arguments.output_dir, "mJpsipipi_mpipi.pdf"), "m(#pi#pi)", "m(J/#psi#pi#pi)", 
    #           mJpsipipi_mpipi_bins(masses), 
    #           masses.mpipi, masses.mJpsipipi, weights,
    #           mc_masses.mpipi, mc_masses.mJpsipipi, mc_weights,
    #           0.25, 1.7, 3.3, 4.8)

    # plot_chi2(os.path.join(arguments.output_dir, "mKpipi_mpipi.pdf"), "m(#pi#pi)", "m(K#pi#pi)", 
    #           mKpipi_mpipi_bins(masses), 
    #           masses.mpipi, masses.mKpipi, weights,
    #           mc_masses.mpipi, mc_masses.mKpipi, mc_weights,
    #           0.25, 1.7, 0.75, 2.2,)

    # plot_chi2(os.path.join(arguments.output_dir, "mKpipi_mKpi.pdf"), "m(K#pi)", "m(K#pi#pi)", 
    #           mKpipi_mKpi_bins(masses), 
    #           masses.mKpi, masses.mKpipi, weights,
    #           mc_masses.mKpi, mc_masses.mKpipi, mc_weights,
    #           0.65, 2.1, 0.75, 2.2)

    # plot_chi2(os.path.join(arguments.output_dir, "mJpsiK_mpipi.pdf"), "m(#pi#pi)", "m(J/#psiK)", 
    #           mJpsiK_mpipi_bins(masses), 
    #           masses.mpipi, masses.mJpsiK, weights,
    #           mc_masses.mpipi, mc_masses.mJpsiK, mc_weights,
    #           0.25, 1.7, 3.5, 5.0)

    # plot_chi2(os.path.join(arguments.output_dir, "mJpsipi_mKpi.pdf"), "m(K#pi)", "m(J/#psi#pi)", 
    #           mJpsipi_mKpi_bins(masses), 
    #           masses.mKpi, masses.mJpsipi, weights,
    #           mc_masses.mKpi, mc_masses.mJpsipi, mc_weights,
    #           0.65, 2.1, 3.2, 4.7)

    # plot_chi2(os.path.join(arguments.output_dir, "mJpsiKpi_mKpi.pdf"), "m(K#pi)", "m(J/#psiK#pi)", 
    #           mJpsiKpi_mKpi_bins(masses), 
    #           masses.mKpi, masses.mJpsiKpi, weights,
    #           mc_masses.mKpi, mc_masses.mJpsiKpi, mc_weights,
    #           0.65, 2.1, 3.7, 5.2)
