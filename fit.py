#!/usr/bin/env python3

import csv
import json
import time
from pprint import pprint

# avoid using Xwindow
import matplotlib
matplotlib.use("agg")

import tensorflow as tf
import numpy as np
import scipy.interpolate as interpolate

# examples of custom particle model
from tf_pwa.breit_wigner import BW, Gamma
from tf_pwa.amp import simple_resonance, register_particle, Particle
from tf_pwa.config_loader import ConfigLoader, MultiConfig
from tf_pwa.experimental import extra_amp, extra_data
from tf_pwa.utils import error_print, tuple_table

@simple_resonance("New", params=["alpha", "beta"])
def New_Particle(m, alpha, beta=0):
    """example Particle model define, can be used in config.yml as `model: New`"""
    zeros = tf.zeros_like(m)
    r = -tf.complex(alpha, beta) * tf.complex(m, zeros)
    return tf.exp(r)


# BW resonance where the total width is the sum of its partial widths
# The widths is given as a x : y in a npy file, which is used in a cubic spline interpolation
# Do not multiply the width of the resonance into the value of y, this can be treated as a floating parameter in the fit
@register_particle("BW_SPW")
class BW_SPW(Particle):
    def __init__(self, *args, **kwargs):
        super(BW_SPW, self).__init__(*args, **kwargs)
        self.nbins, self.range, self.bin_mins, self.bin_coefficients = self.init_interpolation(kwargs["interpolate_knots_file"])

        print(self.nbins, self.range, self.get_mass(), self.get_width(), [c[0] for c in self.bin_coefficients])

    def init_interpolation(self, filepath):
        xy = np.load(filepath)
        # x is the location of knots
        # y the value of the function at each knot
        x = xy[0]
        y = xy[1]

        # Number of bins is number of knots - 1
        nbins = tf.convert_to_tensor(x.shape[0] - 1, dtype=tf.int32)
        x_range = tf.convert_to_tensor([x[0], x[-1]])
        bin_mins = tf.convert_to_tensor(x[:-1])

        # Get the coefficient of the splines
        cubic_spline = interpolate.CubicSpline(x, y)
        bin_coefficients = [tf.convert_to_tensor(cubic_spline.c[i]) for i in range(4)]

        return nbins, x_range, bin_mins, bin_coefficients

    def get_amp(self, data, data_c, **kwargs):
        m = data["m"]
        mass = tf.cast(self.get_mass(), m.dtype)
        # width = tf.cast(self.get_width(), m.dtype)

        # Calculate the mass dependent width
        bin_indices = tf.histogram_fixed_width_bins(m, self.range, self.nbins)
        coefficients = [tf.gather(self.bin_coefficients[i], bin_indices) for i in range(4)]
        bin_mins = tf.gather(self.bin_mins, bin_indices)
        running_width = tf.math.polyval(coefficients, m - bin_mins)
        running_width = tf.stop_gradient(running_width)

        # Calculate the normalisation such that gamma(m0) = gamma0
        # bin_index = tf.histogram_fixed_width_bins(mass, self.range, self.nbins)
        # m0_coefficients = [tf.gather(self.bin_coefficients[i], bin_index) for i in range(4)]
        # m0_bin_min = tf.gather(self.bin_mins, bin_index)
        # normalisation = tf.math.polyval(m0_coefficients, mass - m0_bin_min)

        # gamma = width * running_width / normalisation

        # Use the mass dependent width to calculate the Breit-Wigner amplitude
        return BW(m, mass, running_width)

# BW resonance for the f_0_980
# The widths are functions of pi and K masses
# The values of g_pipi and g_KK couplings are determined by BES https://arxiv.org/abs/hep-ex/0411001
@register_particle("Flatte_f_0_980")
class Flatte_f_0_980(Particle):
    def __init__(self, *args, **kwargs):
        super(Flatte_f_0_980, self).__init__(*args, **kwargs)
        self.g_pipi = 0.165
        self.g_KK = 4.21 * self.g_pipi

        neutral_pi_mass = 0.1349768
        charged_pi_mass = 0.13957039
        neutral_K_mass = 0.497611
        charged_K_mass = 0.493677

        self.neutral_pi_threshold = 4 * neutral_pi_mass * neutral_pi_mass
        self.charged_pi_threshold = 4 * charged_pi_mass * charged_pi_mass
        self.neutral_K_threshold = 4 * neutral_K_mass * neutral_K_mass
        self.charged_K_threshold = 4 * charged_K_mass * charged_K_mass

    def get_amp(self, data, data_c, **kwargs):
        m = data["m"]
        m0 = tf.cast(self.get_mass(), m.dtype)
        s = m * m

        # Calculate the mass dependent width for pipi
        neutral_pi_width = tf.sqrt(1 - self.neutral_pi_threshold/s)
        charged_pi_width = tf.sqrt(1 - self.charged_pi_threshold/s)
        pipi_width = self.g_pipi * (neutral_pi_width + 2 * charged_pi_width) / 3

        # Calculate the mass dependent width for KK below threshold
        # Essentially, the minimum function ensures that for s < threshold, the width is 0
        real_neutral_K_width = tf.sqrt(tf.maximum(self.neutral_K_threshold/s, 1) - 1)
        real_charged_K_width = tf.sqrt(tf.maximum(self.charged_K_threshold/s, 1) - 1)
        real_KK_width = self.g_KK * (real_neutral_K_width + real_charged_K_width) / 2

        # Calculate the mass dependent width for KK above threshold
        # Essentially, the minimum function ensures that for s < threshold, the width is 0
        imag_neutral_K_width = tf.sqrt(1 - tf.minimum(self.neutral_K_threshold/s, 1))
        imag_charged_K_width = tf.sqrt(1 - tf.minimum(self.charged_K_threshold/s, 1))
        imag_KK_width = self.g_KK * (imag_neutral_K_width + imag_charged_K_width) / 2

        # The Flatte parameterisation
        real_term = m0 * m0 - s + m0 * real_KK_width
        imag_term = m0 * (pipi_width + imag_KK_width)
        complex_norm = real_term * real_term + imag_term * imag_term
        amplitude = tf.complex(real_term / complex_norm, imag_term / complex_norm)

        return amplitude

# BW resonance where the total width is the sum of its partial widths
# The widths is given as a x : y in a npy file, which is used in a cubic spline interpolation
# Do not multiply the width of the resonance into the value of y, this can be treated as a floating parameter in the fit
@register_particle("BW_sigma")
class BW_sigma(Particle):
    def __init__(self, *args, **kwargs):
        super(BW_sigma, self).__init__(*args, **kwargs)
        self.pion_threshold = 4 * np.square(0.13957039)

    def get_amp(self, data, data_c, **kwargs):
        m = data["m"]
        mass = tf.cast(self.get_mass(), m.dtype)
        width = tf.cast(self.get_width(), m.dtype)

        # Calculate the mass dependent width
        gamma = tf.sqrt(1 - self.pion_threshold/tf.square(m)) * width

        # Use the mass dependent width to calculate the Breit-Wigner amplitude
        return BW(m, mass, gamma)

@register_particle("Gounaris_Sakurai")
class Gounaris_Sakurai(Particle):
    def __init__(self, *args, **kwargs):
        super(Gounaris_Sakurai, self).__init__(*args, **kwargs)
        self.pion_mass = 0.13957039

    def get_amp(self, data, data_c, **kwargs):
        def D_function(m0, q0):
            log_term = np.log((m0 + 2 * q0)/(2 * self.pion_mass))
            first_term = (3 * self.pion_mass * self.pion_mass) / (np.pi * q0 * q0) * log_term
            second_term = m0 / (2 * np.pi * q0)
            third_term = (self.pion_mass * self.pion_mass * m0) / (np.pi * tf.pow(q0, 3))

            D = first_term + second_term - third_term
            return D

        def h_function(m, q):
            log_term = tf.math.log((m + 2*q) / (2 * self.pion_mass))

            h = (2/np.pi) * (q/m) * log_term
            return h

        def dhdm2_function(m0, q0):
            dhdm = h_function(m0, q0) * self.pion_mass * self.pion_mass / (q0 * q0 * m0) + 1./(m0 * np.pi)
            dhdm2 = (1./(2*m0)) * dhdm

            return dhdm2

        def f_function(m, q, m0, q0, gamma0):
            common_factor = gamma0 * m0 * m0 / tf.pow(q0, 3)
            first_term = q * q * (h_function(m, q) - h_function(m0, q0))
            second_term = q0 * q0 * (m*m - m0*m0) * dhdm2_function(m0, q0)

            f = common_factor * (first_term + second_term)
            return f

        m = data["m"]
        mass = tf.cast(self.get_mass(), m.dtype)
        width = tf.cast(self.get_width(), m.dtype)
        q = data_c["|q|"]
        q0 = data_c["|q0|"]

        if self.bw_l is None:
            decay = self.decay[0]
            self.bw_l = min(decay.get_l_list())
        gamma = Gamma(m, width, q, q0, self.bw_l, mass, self.d)

        numerator = 1 + D_function(mass, q0)
        real = mass*mass - m*m + f_function(m, q, mass, q0, width)
        imag = mass*gamma

        denominator = real*real + imag*imag

        amplitude = tf.complex( numerator * real / denominator, numerator * imag / denominator)

        # Use the mass dependent width to calculate the Breit-Wigner amplitude
        return amplitude


def json_print(dic):
    """print parameters as json"""
    s = json.dumps(dic, indent=2)
    print(s, flush=True)


def load_config(config_file="config.yml", total_same=False):
    config_files = config_file.split(",")
    if len(config_files) == 1:
        return ConfigLoader(config_files[0])
    return MultiConfig(config_files, total_same=total_same)


def fit(config, init_params="", method="BFGS", loop=1, maxiter=500, improve=False, check_grad=False):
    """
    simple fit script
    """
    # load config.yml
    # config = ConfigLoader(config_file)

    # load data
    all_data = config.get_all_data()

    fit_results = []
    for i in range(loop):
        # set initial parameters if have
        if config.set_params(init_params):
            print("using {}".format(init_params))
        else:
            print("\nusing RANDOM parameters", flush=True)
        # try to fit
        try:
            fit_result = config.fit(
                batch=90000, method=method, maxiter=maxiter, improve=improve, check_grad=check_grad
            )
        except KeyboardInterrupt:
            config.save_params("break_params.json")
            raise
        except Exception as e:
            print(e)
            config.save_params("break_params.json")
            raise
        fit_results.append(fit_result)
        # reset parameters
        try:
            config.reinit_params()
        except Exception as e:
            print(e)

    fit_result = fit_results.pop()
    for i in fit_results:
        if i.success:
            if not fit_result.success or fit_result.min_nll > i.min_nll:
                fit_result = i

    config.set_params(fit_result.params)
    json_print(fit_result.params)
    fit_result.save_as("final_params.json")

    # calculate parameters error
    if maxiter != 0:
        fit_error = config.get_params_error(fit_result, batch=90000)
        fit_result.set_error(fit_error)
        fit_result.save_as("final_params.json")
        pprint(fit_error)

        print("\n########## fit results:")
        print("Fit status: ", fit_result.success)
        print("Minimal -lnL = ", fit_result.min_nll)
        for k, v in config.get_params().items():
            print(k, error_print(v, fit_error.get(k, None)))

    return fit_result


def write_some_results(config, fit_result, save_root=False):
    # calculate fit fractions
    phsp_noeff = config.get_phsp_noeff()
    fit_frac, err_frac = config.cal_fitfractions({}, phsp_noeff)

    print("########## fit fractions")
    fit_frac_string = ""
    for i in fit_frac:
        if isinstance(i, tuple):
            name = "{}x{}".format(*i)
        else:
            name = i
        fit_frac_string += "{} {}\n".format(
            name, error_print(fit_frac[i], err_frac.get(i, None))
        )
    print(fit_frac_string)
    save_frac_csv("fit_frac.csv", fit_frac)
    save_frac_csv("fit_frac_err.csv", err_frac)
    # from frac_table import frac_table
    # frac_table(fit_frac_string)
    # chi2, ndf = config.cal_chi2(mass=["R_BC", "R_CD"], bins=[[2,2]]*4)

    # plot partial wave distribution
    config.plot_partial_wave(fit_result, plot_pull=True, save_root=save_root)

def write_some_results_combine(config, fit_result, save_root=False):

    from tf_pwa.applications import fit_fractions

    for i, c in enumerate(config.configs):
        c.plot_partial_wave(
            fit_result, prefix="figure/s{}_".format(i), save_root=save_root
        )

    for it, config_i in enumerate(config.configs):
        print("########## fit fractions {}:".format(it))
        mcdata = config_i.get_phsp_noeff()
        fit_frac, err_frac = fit_fractions(
            config_i.get_amplitude(),
            mcdata,
            config.inv_he,
            fit_result.params,
        )
        fit_frac_string = ""
        for i in fit_frac:
            if isinstance(i, tuple):
                name = "{}x{}".format(*i)  # interference term
            else:
                name = i  # fit fraction
            fit_frac_string += "{} {}\n".format(
                name, error_print(fit_frac[i], err_frac.get(i, None))
            )
        print(fit_frac_string)
        save_frac_csv(f"fit_frac{it}.csv", fit_frac)
        save_frac_csv(f"fit_frac{it}_err.csv", err_frac)
    # from frac_table import frac_table
    # frac_table(fit_frac_string)


def save_frac_csv(file_name, fit_frac):
    table = tuple_table(fit_frac)
    with open(file_name, "w") as f:
        f_csv = csv.writer(f)
        f_csv.writerows(table)


def write_run_point():
    """write time as a point of fit start"""
    with open(".run_start", "w") as f:
        localtime = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(time.time())
        )
        f.write(localtime)


def main():
    """entry point of fit. add some arguments in commond line"""
    import argparse

    parser = argparse.ArgumentParser(description="simple fit scripts")
    parser.add_argument(
        "--no-GPU", action="store_false", default=True, dest="has_gpu"
    )
    parser.add_argument("-c", "--config", default="config.yml", dest="config")
    parser.add_argument(
        "-i", "--init_params", default="init_params.json", dest="init"
    )
    parser.add_argument("-m", "--method", default="BFGS", dest="method")
    parser.add_argument("-l", "--loop", type=int, default=1, dest="loop")
    parser.add_argument(
        "-x", "--maxiter", type=int, default=2000, dest="maxiter"
    )
    parser.add_argument("-r", "--save_root", default=False, dest="save_root")
    parser.add_argument(
        "--total-same", action="store_true", default=False, dest="total_same"
    )
    parser.add_argument(
        "--improve", action="store_true", default=False, dest="improve"
    )
    parser.add_argument(
        "--check_grad", action="store_true", default=False, dest="check_grad"
    )
    results = parser.parse_args()
    if results.has_gpu:
        devices = "/device:GPU:0"
    else:
        devices = "/device:CPU:0"
    with tf.device(devices):
        config = load_config(results.config, results.total_same)
        fit_result = fit(
            config, results.init, results.method, results.loop, results.maxiter, results.improve, results.check_grad
        )
        if isinstance(config, ConfigLoader):
            write_some_results(config, fit_result, save_root=results.save_root)
        else:
            write_some_results_combine(
                config, fit_result, save_root=results.save_root
            )


if __name__ == "__main__":
    write_run_point()
    main()
