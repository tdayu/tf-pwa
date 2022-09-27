from tf_pwa.config_loader import ConfigLoader, MultiConfig
import argparse, os, itertools

import numpy as np
import tensorflow as tf
import scipy.interpolate as interpolate
import scipy.interpolate, scipy.linalg
import matplotlib.pyplot as plt

from tf_pwa.histogram import Hist1D, interp_hist
from tf_pwa.breit_wigner import BW, Gamma
from tf_pwa.amp import simple_resonance, register_particle, Particle, HelicityDecay, register_decay
from tf_pwa.data import data_index, data_to_numpy, data_split, data_merge, data_shape

# BW resonance where the total width is the sum of its partial widths
# The widths is given as a x : y in a npy file, which is used in a cubic spline interpolation
# Do not multiply the width of the resonance into the value of y, this can be treated as a floating parameter in the fit
@register_particle("BW_SPW")
class BW_SPW(Particle):
    def __init__(self, *args, **kwargs):
        super(BW_SPW, self).__init__(*args, **kwargs)
        self.nbins, self.range, self.bin_mins, self.bin_coefficients = self.init_interpolation(kwargs["interpolate_knots_file"])

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
        width = tf.cast(self.get_width(), m.dtype)

        # Calculate the mass dependent width
        bin_indices = tf.histogram_fixed_width_bins(m, self.range, self.nbins)
        coefficients = [tf.gather(self.bin_coefficients[i], bin_indices) for i in range(4)]
        bin_mins = tf.gather(self.bin_mins, bin_indices)
        partial_widths = tf.math.polyval(coefficients, m - bin_mins)
        partial_widths = tf.stop_gradient(partial_widths)

        # Calculate the normalisation such that gamma(m0) = gamma0
        bin_index = tf.histogram_fixed_width_bins(mass, self.range, self.nbins)
        m0_coefficients = [tf.gather(self.bin_coefficients[i], bin_index) for i in range(4)]
        m0_bin_min = tf.gather(self.bin_mins, bin_index)
        normalisation = tf.math.polyval(m0_coefficients, mass - m0_bin_min)

        gamma = width * partial_widths / normalisation

        # Use the mass dependent width to calculate the Breit-Wigner amplitude
        return BW(m, mass, gamma)

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

    # @tf.function
    def get_amp(self, data, data_c, **kwargs):
        # @tf.function
        def centre_of_mass_momentum(m, m_A, m_B):
            q = tf.sqrt((tf.math.square(m) - tf.math.square(m_A + m_B))*(tf.math.square(m) - tf.math.square(m_A - m_B)))/(2 * m)
            return q

        if self.bw_l is None:
            decay = self.decay[0]
            self.bw_l = min(decay.get_l_list())

        dummy_m = tf.linspace(tf.constant(0.8,dtype=tf.float64), tf.constant(2.0,dtype=tf.float64), 25)

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

            dummy_width = (Gamma(dummy_m, width, centre_of_mass_momentum(dummy_m, self.charged_kaon_mass, self.neutral_pion_mass), q0_kplus_pi0, self.bw_l, mass, self.d) + 2 * Gamma(dummy_m, width, centre_of_mass_momentum(dummy_m, self.neutral_kaon_mass, self.charged_pion_mass), q0_k0_piplus, self.bw_l, mass, self.d))/3
            dummy = BW(dummy_m, mass, dummy_width)

            print(dummy_m, dummy)

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
        self.production_poles = [0, 1, 2, 3, 4]
        self.NR_production_channels = [0, 1, 2]

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

        def pipi_K_matrix(mass, mass_products):
            s = mass * mass
            s0 = -3.92637

            Z = ((1 + 0.15) / (s + 0.15)) * (s - 0.5 * PI_MASS * PI_MASS)

            f1a = np.array([0.23399, 0.15044, -0.20545, 0.32825, 0.35412])
            f1a = np.expand_dims(f1a, axis=0)
            c1a = f1a * np.expand_dims(mass_products[5] * (1 - s0)/(s - s0), axis=-1) # [n, 5]

            K = np.zeros(shape=[mass.size, 5, 5], dtype=np.float128)

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

    def get_ls_amp(self, data, data_p, **kwargs):
        pipi_mass = data_p[self.pipi_system]["m"]

        if "pipi_S_wave_cache" not in kwargs["all_data"]:
            cached_NR, cached_pole = self.get_cached_terms(pipi_mass)
            kwargs["all_data"]["pipi_S_wave_cache"] = (cached_NR, cached_pole)

        beta = tf.expand_dims(tf.stack([tf.stack([b() for b in ls_beta]) for ls_beta in self.beta], axis=0), axis=0)
        f_prod = tf.expand_dims(tf.stack([tf.stack([f() for f in ls_f_prod]) for ls_f_prod in self.f_prod], axis=0), axis=0)

        (cached_NR, cached_pole) = kwargs["all_data"]["pipi_S_wave_cache"]
        amplitude = tf.reduce_sum(beta * cached_pole, axis=-1) + \
                    tf.reduce_sum(f_prod * cached_NR, axis=-1) # n, n_ls

        q0 = self.get_relative_momentum2(data_p, False)
        data["|q0|2"] = q0
        if "|q|2" in data:
            q = data["|q|2"]
        else:
            q = self.get_relative_momentum2(data_p, True)
            data["|q|2"] = q
        if self.has_barrier_factor:
            bf = self.get_barrier_factor2(
                data_p[self.core]["m"], q, q0, self.d
            )
            amplitude = amplitude * tf.cast(bf, amplitude.dtype)

        return amplitude

@ConfigLoader.register_function()
def get_weights_for_phase_space(
    self,
    params=None,
    data=None,
    phsp=None,
    bg=None,
    batch=65000
):
    if params is None:
        params = {}
    nll = None
    if hasattr(params, "min_nll"):
        nll = float(getattr(params, "min_nll"))
    if hasattr(params, "params"):
        params = getattr(params, "params")

    if data is None:
        data = self.get_data("data")
        bg = self.get_data("bg")
        phsp = self.get_phsp_plot()
    if bg is None:
        if self.config["data"].get("model", "auto") == "cfit":
            bg = _get_cfit_bg(self, data, phsp)
        else:
            bg = [bg] * len(data)
    if self.config["data"].get("model", "auto") == "cfit":
        phsp = _get_cfit_eff_phsp(self, phsp)
    amp = self.get_amplitude()

    self._Ngroup = len(data)
    ws_bkg = [
        None if bg_i is None else bg_i.get("weight", None) for bg_i in bg
    ]

    with amp.temp_params(params):
        weights_i = [amp(i) for i in data_split(phsp[0], batch)]
        weight_phsp = data_merge(*weights_i)  # amp(phsp)
        total_weight = (
            weight_phsp * phsp[0].get("weight", 1.0) * phsp[0].get("eff_value", 1.0)
        )
        data_weight = data[0].get("weight", None)
        if data_weight is None:
            n_data = data_shape(data[0])
        else:
            n_data = np.sum(data_weight)

        norm_frac = n_data / np.sum(total_weight)
        phsp_weights = total_weight * norm_frac

    return phsp_weights

parser = argparse.ArgumentParser(description="Script to plot invariant mass")
parser.add_argument("config", type=str, help="Input config file.")
parser.add_argument("output_path", type=str, help="Output path.")
arguments = parser.parse_args()

os.makedirs(os.path.dirname(arguments.output_path), exist_ok=True)
config = ConfigLoader(arguments.config)
config.set_params("final_params.json")
weights = config.get_weights_for_phase_space()
weights = weights.numpy()

np.save(arguments.output_path, weights)
