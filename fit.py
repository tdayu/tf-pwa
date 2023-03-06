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
import itertools, scipy.linalg, scipy.interpolate

# examples of custom particle model
from tf_pwa.breit_wigner import BW, Gamma
from tf_pwa.amp import simple_resonance, register_particle, Particle
from tf_pwa.config_loader import ConfigLoader, MultiConfig
from tf_pwa.experimental import extra_amp, extra_data
from tf_pwa.utils import error_print, tuple_table
from tf_pwa.amp import HelicityDecay, register_decay

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
        self.cubic_spline = self.init_interpolation(kwargs["interpolate_knots_file"])
        # self.nbins, self.range, self.bin_mins, self.bin_coefficients = self.init_interpolation(kwargs["interpolate_knots_file"])

    def init_interpolation(self, filepath):
        xy = np.load(filepath)
        # x is the location of knots
        # y the value of the function at each knot
        x = xy[0]
        y = xy[1]

        # Number of bins is number of knots - 1
        # nbins = tf.convert_to_tensor(x.shape[0] - 1, dtype=tf.int32)
        # x_range = tf.convert_to_tensor([x[0], x[-1]])
        # bin_mins = tf.convert_to_tensor(x[:-1])

        # Get the coefficient of the splines
        cubic_spline = interpolate.CubicSpline(x, y)
        # bin_coefficients = [tf.convert_to_tensor(cubic_spline.c[i]) for i in range(4)]

        # return nbins, x_range, bin_mins, bin_coefficients
        return cubic_spline

    def init_data(self, data, data_c, **kwargs):
        m = data["m"].numpy()
        # mass = tf.cast(self.get_mass(), m.dtype).numpy()
        running_width = self.cubic_spline(m)
        kwargs["all_data"]["BW_SPW_width"] = tf.convert_to_tensor(running_width)
        # bin_indices = tf.histogram_fixed_width_bins(m, self.range, self.nbins)
        # coefficients = [tf.gather(self.bin_coefficients[i], bin_indices) for i in range(4)]
        # bin_mins = tf.gather(self.bin_mins, bin_indices)
        # running_width = tf.math.polyval(coefficients, m - bin_mins)
        # running_width = tf.stop_gradient(running_width)


    @tf.function
    def get_amp(self, data, data_c, **kwargs):
        with tf.name_scope("BW_SPW") as scope:
            m = data["m"]
            mass = tf.cast(self.get_mass(), m.dtype)
            width = tf.cast(self.get_width(), m.dtype)

            # bin_indices = tf.histogram_fixed_width_bins(m, self.range, self.nbins)
            # coefficients = [tf.gather(self.bin_coefficients[i], bin_indices) for i in range(4)]
            # bin_mins = tf.gather(self.bin_mins, bin_indices)
            # running_width = tf.math.polyval(coefficients, m - bin_mins)
            # running_width = tf.stop_gradient(running_width)

            # Calculate the normalisation such that gamma(m0) = gamma0
            # bin_index = tf.histogram_fixed_width_bins(mass, self.range, self.nbins)
            # m0_coefficients = [tf.gather(self.bin_coefficients[i], bin_index) for i in range(4)]
            # m0_bin_min = tf.gather(self.bin_mins, bin_index)
            # normalisation = tf.math.polyval(m0_coefficients, mass - m0_bin_min)

            # gamma = width * running_width / normalisation

            # Use the mass dependent width to calculate the Breit-Wigner amplitude
            return BW(m, mass, width * kwargs["all_data"]["BW_SPW_width"])

# BW resonance where the total width is the sum of its partial widths
# The widths is given as a x : y in a npy file, which is used in a cubic spline interpolation
# Do not multiply the width of the resonance into the value of y, this can be treated as a floating parameter in the fit
@register_particle("BW_SPW_Exp")
class BW_SPW_Exp(Particle):
    def __init__(self, *args, **kwargs):
        super(BW_SPW_Exp, self).__init__(*args, **kwargs)
        self.cubic_spline = self.init_interpolation(kwargs["interpolate_knots_file"])
        self.radius = self.add_var("radius", value=0.1, fix=False)

    def init_interpolation(self, filepath):
        xy = np.load(filepath)
        # x is the location of knots
        # y the value of the function at each knot
        x = xy[0]
        y = xy[1]

        # Get the coefficient of the splines
        cubic_spline = interpolate.CubicSpline(x, y)
        return cubic_spline

    def init_data(self, data, data_c, **kwargs):
        m = data["m"].numpy()
        running_width = self.cubic_spline(m)
        kwargs["all_data"]["BW_SPW_Exp_width"] = tf.convert_to_tensor(running_width)

    @tf.function
    def get_amp(self, data, data_c, **kwargs):
        PI_MASS = 0.13957
        K_MASS = 0.493677
        with tf.name_scope("BW_SPW_Exp") as scope:
            m = data["m"]
            mass = tf.cast(self.get_mass(), m.dtype)
            width = kwargs["all_data"]["BW_SPW_Exp_width"] * tf.exp(-tf.square(self.radius() * (m - 2 * PI_MASS - K_MASS)) / 2)

            return BW(m, mass, width)

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

    # @tf.function
    def get_amp(self, data, data_c, **kwargs):
        with tf.name_scope("Flatte_f_0_980") as scope:
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

    # @tf.function
    def get_amp(self, data, data_c, **kwargs):
        with tf.name_scope("BW_sigma") as scope:
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


# BW resonance where the total width is the sum of its partial widths
# The widths is given as a x : y in a npy file, which is used in a cubic spline interpolation
# Do not multiply the width of the resonance into the value of y, this can be treated as a floating parameter in the fit
@register_particle("BW_Kst_892_plus")
class BW_Kst_892_plus(Particle):
    def __init__(self, *args, **kwargs):
        super(BW_Kst_892_plus, self).__init__(*args, **kwargs)
        self.neutral_kaon_mass = tf.constant(0.497611, dtype=tf.float64)
        self.charged_pion_mass = tf.constant(0.13957039, dtype=tf.float64)

        self.charged_kaon_mass = tf.constant(0.493677, dtype=tf.float64)
        self.neutral_pion_mass = tf.constant(0.1349768, dtype=tf.float64)

    @tf.function
    def get_amp(self, data, data_c, **kwargs):
        # @tf.function
        def centre_of_mass_momentum(m, m_A, m_B):
            q = tf.sqrt((tf.math.square(m) - tf.math.square(m_A + m_B))*(tf.math.square(m) - tf.math.square(m_A - m_B)))/(2 * m)
            return q

        if self.bw_l is None:
            decay = self.decay[0]
            self.bw_l = min(decay.get_l_list())

        with tf.name_scope("BW_Kst_892_plus") as scope:
            m = data["m"]
            mass = tf.cast(self.get_mass(), m.dtype)
            width = tf.cast(self.get_width(), m.dtype)

            q_kplus_pi0 = centre_of_mass_momentum(m, self.charged_kaon_mass, self.neutral_pion_mass)
            q0_kplus_pi0 = centre_of_mass_momentum(mass, self.charged_kaon_mass, self.neutral_pion_mass)

            q_k0_piplus = centre_of_mass_momentum(m, self.neutral_kaon_mass, self.charged_pion_mass)
            q0_k0_piplus = centre_of_mass_momentum(mass, self.neutral_kaon_mass, self.charged_pion_mass)

            kplus_pi0_width = Gamma(m, width, q_kplus_pi0, q0_kplus_pi0, self.bw_l, mass, self.d)
            k0_piplus_width = Gamma(m, width, q_k0_piplus, q0_k0_piplus, self.bw_l, mass, self.d)

            running_width = (kplus_pi0_width + 2 * k0_piplus_width) / 3

            return BW(m, mass, running_width)

## Pre-calculate matrix inversion in 128 bits (with numpy and scipy) for more precision
## Cache the matrix in 64 bits in tensorflow
@register_decay("pipiKMatrix")
class PipiKmatrix(HelicityDecay):
    def __init__(self, *args, **kwargs):
        self.d = 3.0
        self.pole_masses = np.array([0.651, 1.2036, 1.55817, 1.21, 1.82206], dtype=np.float128)
        self.couplings = [np.array([0.22889, -0.55377,  0.     , -0.39899, -0.34639], dtype=np.float128), # f(980)
                          np.array([0.94128,  0.55095,  0.     ,  0.39065,  0.31503], dtype=np.float128), # f(1300)
                          np.array([0.36856,  0.23888,  0.55639,  0.18340,  0.18681], dtype=np.float128), # f(1500)
                          np.array([0.33650,  0.40907,  0.85679,  0.19906, -0.00984], dtype=np.float128), # f(1750)
                          np.array([0.18171, -0.17558, -0.79658, -0.00355,  0.22358], dtype=np.float128)] # f(1200-1600)
        super().__init__(*args, **kwargs)

        self.n_ls = len(self.get_ls_list())
        print(self.n_ls)
        self.n_poles = 5
        self.production_poles = kwargs["production_poles"]
        self.NR_production_channels = kwargs["NR_production_channels"]

        assert(kwargs["pipi_system"] in self.outs[0].name or kwargs["pipi_system"] in self.outs[1].name)
        self.pipi_system = self.outs[0] if kwargs["pipi_system"] in self.outs[0].name else self.outs[1]

    def init_params(self):
        # self.beta = self.add_var(
        #     "beta", is_complex=True, polar=self.params_polar, shape=(1, self.n_ls, len(self.production_poles))
        # )
        # self.f_prod = self.add_var(
        #     "f_prod", is_complex=True, polar=self.params_polar, shape=(1, self.n_ls, len(self.NR_production_channels))
        # )
        # self.beta.set_fix_idx(fix_idx=0, fix_vals=(1.0, 0.0))

        self.beta = []
        self.f_prod = []

        assert self.n_poles > 0

        # Initialize the floating parameters
        for j, (l, s) in enumerate(self.get_ls_list()):
            ls_beta = []
            ls_f_prod = []

            for i in range(len(self.production_poles)):
                ls_beta.append(self.add_var(f"beta{i}_l{l}_s{s}", is_complex=True))
            for i in range(len(self.NR_production_channels)):
                ls_f_prod.append(self.add_var(f"f_prod{i}_l{l}_s{s}", is_complex=True))

            for var in itertools.chain(ls_beta, ls_f_prod):
                var.set_value(np.random.uniform(-0.5, 0.5, size=2))

            if j == 0:
                ls_beta[0].fixed(1.0) # Fix the first beta since this is absorbed by overall decay chain coupling

            self.beta.append(ls_beta)
            self.f_prod.append(ls_f_prod)

        # self.beta = tf.expand_dims(tf.stack(self.beta, axis=0), axis=0) # 1, n_ls, n_pole
        # self.f_prod = tf.expand_dims(tf.stack(self.f_prod, axis=0), axis=0) # 1, n_ls, n_NR

    def get_cached_terms(self, m):
        PI_MASS = 0.13957
        K_MASS = 0.493677
        ETA_MASS = 0.547862
        ETAP_MASS = 0.95778

        def pipi_K_matrix(s, mass_products):
            s0 = -3.92637

            Z = ((1 + 0.15) / (s + 0.15)) * (s - 0.5 * PI_MASS * PI_MASS)

            f1a = np.array([0.23399, 0.15044, -0.20545, 0.32825, 0.35412])
            f1a = np.expand_dims(f1a, axis=0)
            c1a = f1a * np.expand_dims(mass_products[5] * (1 - s0)/(s - s0), axis=-1) # [n, 5]

            K = np.zeros(shape=[s.size, 5, 5], dtype=np.float128)

            for i, j in itertools.product(range(5), range(5)):
                K[:, i, j] = sum([pole_couplings[i] * pole_couplings[j]  * mass_products[k] for k, pole_couplings in enumerate(self.couplings)])

            K[:, 0, :] += c1a
            K = K * np.expand_dims(np.expand_dims(Z, axis=-1), axis=-1)

            return K

        def get_phase_space_matrix(mass):
            def rho4pi_phase_space(mass, filepath):
                data = np.load(filepath)
                x = data["x"]
                y = data["y"]
                normalisation = np.sqrt(1 - 16 * PI_MASS * PI_MASS) / y[-1]
                y = normalisation * y

                function_below_1 = interpolate.CubicSpline(x, y)

                phase_space = np.where(mass < 1, function_below_1(mass), np.sqrt(1 - 16 * PI_MASS * PI_MASS / (mass * mass)))

                return phase_space

            def two_body_phase_space(mass, m1, m2):
                phase_space_squared = np.where(mass > np.abs(m1 - m2), (1 - np.square(m1 + m2)/np.square(mass)) * (1 - np.square(m1 - m2)/np.square(mass)), 0.)
                phase_space = np.sqrt(np.abs(phase_space_squared))
                phase_space = np.where(phase_space_squared >= 0, phase_space + 0j, 0 + phase_space * 1j)

                return phase_space

            matrix = np.zeros(shape=[mass.size, 5, 5], dtype=np.complex256)
            matrix[:, 0, 0] = two_body_phase_space(mass, PI_MASS, PI_MASS)
            matrix[:, 1, 1] = two_body_phase_space(mass, K_MASS, K_MASS)
            matrix[:, 2, 2] = rho4pi_phase_space(mass, "/home/dtou/Software/tf-pwa/workspace/test_K_matrix/threshold0_4pi_phase_space.npz") + 0j
            matrix[:, 3, 3] = two_body_phase_space(mass, ETA_MASS, ETA_MASS)
            matrix[:, 4, 4] = two_body_phase_space(mass, ETA_MASS, ETAP_MASS)

            return matrix

        mass = m.numpy().astype(np.float128)
        s = np.square(mass)

        mass_products = [(np.square(self.pole_masses[1]) - s) * (np.square(self.pole_masses[2]) - s) * (np.square(self.pole_masses[3]) - s) * (np.square(self.pole_masses[4]) - s),
                         (np.square(self.pole_masses[0]) - s) * (np.square(self.pole_masses[2]) - s) * (np.square(self.pole_masses[3]) - s) * (np.square(self.pole_masses[4]) - s),
                         (np.square(self.pole_masses[0]) - s) * (np.square(self.pole_masses[1]) - s) * (np.square(self.pole_masses[3]) - s) * (np.square(self.pole_masses[4]) - s),
                         (np.square(self.pole_masses[0]) - s) * (np.square(self.pole_masses[1]) - s) * (np.square(self.pole_masses[2]) - s) * (np.square(self.pole_masses[4]) - s),
                         (np.square(self.pole_masses[0]) - s) * (np.square(self.pole_masses[1]) - s) * (np.square(self.pole_masses[2]) - s) * (np.square(self.pole_masses[3]) - s),
                         (np.square(self.pole_masses[0]) - s) * (np.square(self.pole_masses[1]) - s) * (np.square(self.pole_masses[2]) - s) * (np.square(self.pole_masses[3]) - s) * (np.square(self.pole_masses[4]) - s)]

        K = pipi_K_matrix(s, mass_products)
        rho = get_phase_space_matrix(mass)
        eye = np.reshape(np.eye(5), [1, 5, 5]) * np.reshape(mass_products[5], [-1, 1, 1])
        K_matrix_term = eye - 1j * np.matmul(K, rho)
        inversed_matrix = np.stack([scipy.linalg.inv(to_inverse) for to_inverse in K_matrix_term], axis=0)

        # cached NR terms
        common_NR_term = (2.0 / (s + 1.0)) * mass_products[5]
        cached_NR_term = np.reshape(common_NR_term, [-1, 1]) * np.stack([inversed_matrix[:, 0, index] for index in self.NR_production_channels], axis=-1)
        cached_NR_term = np.reshape(cached_NR_term, [-1, 1, len(self.NR_production_channels)]) # n, 1, n_pole

        # cached pole terms 
        cached_pole_term = np.stack([mass_products[i] * np.sum(inversed_matrix[:, 0, :] * np.reshape(self.couplings[i], [1, -1]), axis=-1) for i in self.production_poles], axis=-1)
        cached_pole_term = np.reshape(cached_pole_term, [-1, 1, len(self.production_poles)]) # n, 1, n_NR

        return tf.convert_to_tensor(cached_NR_term.astype(np.complex128)), tf.convert_to_tensor(cached_pole_term.astype(np.complex128))

    @tf.function
    def get_ls_amp(self, data, data_p, **kwargs):
        with tf.name_scope("pipi_K_matrix") as scope:
            beta = tf.expand_dims(tf.stack([tf.stack([b() for b in ls_beta]) for ls_beta in self.beta], axis=0), axis=0)
            f_prod = tf.expand_dims(tf.stack([tf.stack([f() for f in ls_f_prod]) for ls_f_prod in self.f_prod], axis=0), axis=0)

            cached_NR = kwargs["all_data"]["pipi_S_wave_cache"]["NR"]
            cached_pole = kwargs["all_data"]["pipi_S_wave_cache"]["pole"]
            amplitude = tf.reduce_sum(beta * cached_pole, axis=-1) + \
                        tf.reduce_sum(f_prod * cached_NR, axis=-1) # n, n_ls

            # data["|q0|2"] = q0
            # if "|q|2" in data:
            #     q = data["|q|2"]
            # else:
            #     data["|q|2"] = q
            if self.has_barrier_factor:
                _, q0 = self.get_relative_momentum(data_p, False)
                _, q = self.get_relative_momentum(data_p, True)

                bf = self.get_barrier_factor2(
                    data_p[self.core]["m"], q, q0, self.d
                )
                amplitude = amplitude * tf.cast(bf, amplitude.dtype)

        return amplitude

    def init_data(self, data_c, data_p, all_data):
        super(PipiKmatrix, self).init_data(data_c, data_p, all_data)

        if "pipi_S_wave_cache" not in all_data:
            pipi_mass = data_p[self.pipi_system]["m"]
            cached_NR, cached_pole = self.get_cached_terms(pipi_mass)
            all_data["pipi_S_wave_cache"] = { "NR" : cached_NR, 
                                              "pole" : cached_pole}

## Pre-calculate matrix inversion in 128 bits (with numpy and scipy) for more precision
## Cache the matrix in 64 bits in tensorflow
@register_decay("KpiKMatrix")
class KpiKmatrix(HelicityDecay):
    def __init__(self, *args, **kwargs):
        self.d = 3.0
        self.pole_mass = np.sqrt(1.7919).astype(np.float128)
        self.couplings = np.array([0.31072, -0.02323], dtype=np.float128)
        self.C11 = np.array([0.00811, -0.15099, 0.79299], np.float128)
        self.C22 = np.array([0.0022596, -0.038266, 0.15040], np.float128)
        self.C12 = np.array([0.00085655, -0.0219, 0.17054], np.float128)
        self.D = np.array([-0.00092057, 0.026637, -0.22147], np.float128)

        super().__init__(*args, **kwargs)

        self.n_ls = len(self.get_ls_list())

        assert(kwargs["Kpi_system"] in self.outs[0].name or kwargs["Kpi_system"] in self.outs[1].name)
        self.Kpi_system = self.outs[0] if kwargs["Kpi_system"] in self.outs[0].name else self.outs[1]

    def init_params(self):
        self.beta = []
        self.c1_coefficients = []
        self.c1_angles = []
        self.c2 = []
        self.c3 = []

        # Initialize the floating parameters
        for j, (l, s) in enumerate(self.get_ls_list()):
            ls_c1_coefficients = []

            self.beta.append(self.add_var(f"beta_l{l}_s{s}", is_complex=True))
            self.c1_angles.append(self.add_var(f"c1_angle_l{l}_s{s}"))
            self.c2.append(self.add_var(f"c20_l{l}_s{s}", is_complex=True))
            self.c3.append(self.add_var(f"c30_l{l}_s{s}", is_complex=True))

            ls_c1_coefficients.append(self.add_var(f"c12_l{l}_s{s}"))
            ls_c1_coefficients.append(self.add_var(f"c11_l{l}_s{s}"))
            ls_c1_coefficients.append(self.add_var(f"c10_l{l}_s{s}"))
            self.c1_coefficients.append(ls_c1_coefficients)

        # Initialize the starting values
        for var in itertools.chain(self.beta, self.c2, self.c3):
            var.set_value(np.random.uniform(-0.5, 0.5, size=2))
        for var in self.c1_angles:
            var.set_value(np.random.uniform(-np.pi, np.pi))
        for coefficients in self.c1_coefficients:
            for coef in coefficients:
                coef.set_value(np.random.uniform(-0.5, 0.5))

        self.beta[0].fixed(1.0) # Fix the first beta since this is absorbed by overall decay chain coupling

    def get_cached_terms(self, m):
        PI_MASS = 0.13957
        K_MASS = 0.493677
        ETAP_MASS = 0.95778
        s_norm = K_MASS * K_MASS + PI_MASS * PI_MASS
        s_0_I1 = 0.23
        s_0_I3 = 0.27

        def Kpi_I1_K_matrix(s):
            common_factor = (s - s_0_I1) / s_norm
            s_tilda = s/s_norm - 1

            K11 = common_factor * (self.couplings[0] * self.couplings[0] / (np.square(self.pole_mass) - s) + np.polyval(self.C11, s_tilda))
            K22 = common_factor * (self.couplings[1] * self.couplings[1] / (np.square(self.pole_mass) - s) + np.polyval(self.C22, s_tilda))
            K12 = common_factor * (self.couplings[0] * self.couplings[1] / (np.square(self.pole_mass) - s) + np.polyval(self.C12, s_tilda))

            K = np.zeros(shape=[s.size, 2, 2], dtype=np.float128)
            K[:, 0, 0] = K11
            K[:, 1, 1] = K22
            K[:, 0, 1] = K12
            K[:, 1, 0] = K12

            return K

        def Kpi_I3_K_matrix(s):
            common_factor = (s - s_0_I3) / s_norm
            s_tilda = s/s_norm - 1
            K_matrix = common_factor * np.polyval(self.D, s_tilda)

            return K_matrix

        def two_body_phase_space(mass, m1, m2):
            phase_space_squared = np.where(mass > np.abs(m1 - m2), (1 - np.square(m1 + m2)/np.square(mass)) * (1 - np.square(m1 - m2)/np.square(mass)), 0.)
            phase_space = np.sqrt(np.abs(phase_space_squared))
            phase_space = np.where(phase_space_squared >= 0, phase_space + 0j, 0 + phase_space * 1j)

            return phase_space

        def get_phase_space_matrix(mass):
            matrix = np.zeros(shape=[mass.size, 2, 2], dtype=np.complex256)
            matrix[:, 0, 0] = two_body_phase_space(mass, K_MASS, PI_MASS)
            matrix[:, 1, 1] = two_body_phase_space(mass, K_MASS, ETAP_MASS)

            return matrix

        mass = m.numpy().astype(np.float128)
        s = np.square(mass)

        K_I1 = Kpi_I1_K_matrix(s)
        rho = get_phase_space_matrix(mass)
        eye = np.reshape(np.eye(2), [1, 2, 2])

        K_matrix_term_I1 = eye - 1j * np.matmul(K_I1, rho)
        inversed_matrix_I1 = np.stack([scipy.linalg.inv(r) for r in K_matrix_term_I1], 0)

        inversed_matrix_I3 = 1./(1. - 1j * two_body_phase_space(mass, K_MASS, PI_MASS) * Kpi_I3_K_matrix(mass))

        cached_pole_term = np.sum(inversed_matrix_I1[:, 0, :] * np.reshape(np.array(self.couplings), [1, -1]), -1) / (np.square(self.pole_mass) - s)

        # cache the NR terms multiplied with the P vector
        cached_NR_Kpi_term = inversed_matrix_I1[:, 0, 0]
        cached_NR_Ketap_term = inversed_matrix_I1[:, 0, 0]

        # cache the I=3/2 term
        cached_I3_term = inversed_matrix_I3

        return tf.convert_to_tensor(cached_pole_term.astype(np.complex128)), tf.convert_to_tensor(cached_NR_Kpi_term.astype(np.complex128)), tf.convert_to_tensor(cached_NR_Ketap_term.astype(np.complex128)), tf.convert_to_tensor(cached_I3_term.astype(np.complex128))

    @tf.function
    def get_ls_amp(self, data, data_p, **kwargs):
        with tf.name_scope("Kpi_K_matrix") as scope:
            mass = data_p[self.Kpi_system]["m"]
            s = tf.square(mass)
            s_hat = s - 2

            # All of them have shape [n, 1]
            cached_pole = kwargs["all_data"]["Kpi_S_wave_cache"]["pole"]
            cached_Kpi_NR = kwargs["all_data"]["Kpi_S_wave_cache"]["Kpi_NR"]
            cached_Ketap_NR = kwargs["all_data"]["Kpi_S_wave_cache"]["Ketap_NR"]
            cached_I3 = kwargs["all_data"]["Kpi_S_wave_cache"]["I3"]

            beta = tf.expand_dims(tf.stack([b() for b in self.beta]), axis=0) # [1, n_ls]
            ls_c1 = [[c1() for c1 in ls_c1_coefficients] for ls_c1_coefficients in self.c1_coefficients] # list of size n_ls, entries shape [3] for 3 c1
            c2 = tf.expand_dims(tf.stack([c() for c in self.c2]), axis=0) # [1, n_ls]
            c3 = tf.expand_dims(tf.stack([c() for c in self.c3]), axis=0) # [1, n_ls]
            ls_c1_angle = tf.expand_dims(tf.stack([angle() for angle in self.c1_angles]), axis=0) # [1, n_ls]

            F1_pole = beta * cached_pole # [n, n_ls]

            Kpi_NR_polynomial = tf.stack([tf.math.polyval(c1, s_hat) for c1 in ls_c1], axis=-1) # [n, n_ls]
            P_vector_Kpi_NR = tf.complex(tf.cos(ls_c1_angle), tf.sin(ls_c1_angle)) * tf.complex(Kpi_NR_polynomial, tf.constant(0., dtype=tf.float64))
            P_vector_Ketap_NR = tf.identity(c2)
            F1_NR = P_vector_Kpi_NR * cached_Kpi_NR + P_vector_Ketap_NR * cached_Ketap_NR

            F3 = c3 * cached_I3

            amplitude = F1_pole + F1_NR + F3

            if self.has_barrier_factor:
                _, q0 = self.get_relative_momentum(data_p, False)
                _, q = self.get_relative_momentum(data_p, True)

                bf = self.get_barrier_factor2(
                    data_p[self.core]["m"], q, q0, self.d
                )
                amplitude = amplitude * tf.cast(bf, amplitude.dtype)

        return amplitude

    def init_data(self, data_c, data_p, all_data):
        super(KpiKmatrix, self).init_data(data_c, data_p, all_data)

        if "Kpi_S_wave_cache" not in all_data:
            Kpi_mass = data_p[self.Kpi_system]["m"]
            cached_pole, cached_Kpi_NR, cached_Ketap_NR, cached_I3 = self.get_cached_terms(Kpi_mass)
            all_data["Kpi_S_wave_cache"] = { "pole" : tf.expand_dims(cached_pole, axis=-1),
                                             "Kpi_NR" : tf.expand_dims(cached_Kpi_NR, axis=-1),
                                             "Ketap_NR" : tf.expand_dims(cached_Ketap_NR, axis=-1),
                                             "I3" : tf.expand_dims(cached_I3, axis=-1)}

# Polynomial lineshape
@register_particle("PolyLineshape")
class PolyLineshape(Particle):
    def __init__(self, *args, **kwargs):
        super(PolyLineshape, self).__init__(*args, **kwargs)
        self.n_degrees = int(kwargs["n_degrees"])

    def init_params(self):
        self.coefficients = [self.add_var(f"poly{i+1}", value=0., fix=False) for i in range(self.n_degrees)]
        for coefficient in self.coefficients:
            coefficient.set_bound((-1, 1))

    @tf.function
    def get_amp(self, data, data_c, **kwargs):
        with tf.name_scope("BW_Kst_892_plus") as scope:
            m = data["m"]
            # The constant is set to 1.0 since this is absorbed by DecayChain amplitude
            polyval_coefficients = [coefficient() for coefficient in reversed(self.coefficients)] + [1.0]
            polyval_coefficients = [tf.cast(coefficient, dtype=m.dtype) for coefficient in polyval_coefficients]
            zeros = tf.constant(0., dtype=m.dtype)

            return tf.complex(tf.math.polyval(polyval_coefficients, m), zeros)

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
                batch=92000, method=method, maxiter=maxiter, improve=improve, check_grad=check_grad
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
        fit_error = config.get_params_error(fit_result, batch=92000)
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
