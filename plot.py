from tf_pwa.config_loader import ConfigLoader, MultiConfig
import argparse, os, itertools

import numpy as np
import tensorflow as tf
import scipy.interpolate as interpolate
import scipy.interpolate, scipy.linalg
import matplotlib.pyplot as plt

from tf_pwa.histogram import Hist1D, interp_hist
from tf_pwa.breit_wigner import BW, Gamma
from tf_pwa.amp import simple_resonance, register_particle, Particle, register_decay, HelicityDecay
from tf_pwa.data import data_index, data_to_numpy

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
            return BW(m, mass, kwargs["all_data"]["BW_SPW_width"])

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

def json_print(dic):
    """print parameters as json"""
    s = json.dumps(dic, indent=2)
    print(s, flush=True)


@ConfigLoader.register_function()
def plot_grouped_partial_waves(
    self,
    params=None,
    data=None,
    phsp=None,
    bg=None,
    prefix="figure/",
    res=None,
    save_root=False,
    **kwargs
):
    if params is None:
        params = {}
    nll = None
    if hasattr(params, "min_nll"):
        nll = float(getattr(params, "min_nll"))
    if hasattr(params, "params"):
        params = getattr(params, "params")
    # print(nll, params)
    path = os.path.dirname(prefix)
    os.makedirs(path, exist_ok=True)

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
    amp.init_data(data[0])
    amp.init_data(phsp[0])
    self._Ngroup = len(data)
    ws_bkg = [
        None if bg_i is None else bg_i.get("weight", None) for bg_i in bg
    ]
    ## check that group labels and mothers are not repeated
    groups = self.plot_params.config["groups"]
    group_labels = set()
    mothers = set()
    for group in groups:
        label = group["label"]
        if label not in group_labels:
            group_labels.add(label)
        else:
            raise ValueError(f"{label} is repeated in plot group labels")
        for mother in group["mothers"]:
            if mother not in mothers:
                mothers.add(mother)
            else:
                raise ValueError(f"Mother particle <{mother}> is repeated in plot group")


    # ws_bkg, ws_inmc = self._get_bg_weight(data, bg)
    chain_property = []
    for i, group in enumerate(groups):
        name_i = "_".join(group["mothers"])
        display = group["label"]
        chain_property.append([i, name_i, display, None])
    res = [group["mothers"] for group in groups]

    plot_var_dic = {}
    for conf in self.plot_params.get_params():
        name = conf.get("name")
        display = conf.get("display", name)
        upper_ylim = conf.get("upper_ylim", None)
        idx = conf.get("idx")
        trans = conf.get("trans", lambda x: x)
        has_legend = conf.get("legend", False)
        xrange = conf.get("range", None)
        bins = conf.get("bins", None)
        units = conf.get("units", "")
        yscale = conf.get("yscale", "linear")
        plot_var_dic[name] = {
            "display": display,
            "upper_ylim": upper_ylim,
            "legend": has_legend,
            "idx": idx,
            "trans": trans,
            "range": xrange,
            "bins": bins,
            "units": units,
            "yscale": yscale,
        }
    if self._Ngroup == 1:
        data_dict, phsp_dict, bg_dict = self._cal_partial_wave(
            amp,
            params,
            data[0],
            phsp[0],
            bg[0],
            ws_bkg[0],
            prefix,
            plot_var_dic,
            chain_property,
            save_root=save_root,
            res=res,
            **kwargs,
        )
        if "subsamples" in self.plot_params.config:
            print("Plotting in subsamples")
            for subsample in self.plot_params.config["subsamples"]:
                data_indices = np.load(subsample["data"])
                phsp_indices = np.load(subsample["phsp"])
                bins = subsample["bins"]
                additional_prefix = subsample["prefix"]

                assert(data_indices.dtype == bool)
                assert(phsp_indices.dtype == bool)
                assert(all([data_indices.shape == np_array.shape for np_array in data_dict.values()]))
                assert(all([phsp_indices.shape == np_array.shape for np_array in phsp_dict.values()]))

                subsample_data_dict = {key : value[data_indices] for key, value in data_dict.items()}
                subsample_phsp_dict = {key : value[phsp_indices] for key, value in phsp_dict.items()}
                subsample_prefix  = f"{os.path.join(prefix, additional_prefix)}/"
                os.makedirs(subsample_prefix, exist_ok=True)

                subsample_plot_var_dic = {key : value.copy() for key, value in plot_var_dic.items()}
                for key in subsample_plot_var_dic:
                    subsample_plot_var_dic[key]["bins"] = bins
                    subsample_plot_var_dic[key]["legend"] = False

                self._plot_partial_wave(
                    subsample_data_dict,
                    subsample_phsp_dict,
                    bg_dict,
                    subsample_prefix,
                    subsample_plot_var_dic,
                    chain_property,
                    nll=nll,
                    **kwargs,
                )
        os.makedirs(os.path.join(prefix, 'fit_data'), exist_ok=True)
        self._plot_partial_wave(
            data_dict,
            phsp_dict,
            bg_dict,
            f"{os.path.join(prefix, 'fit_data')}/",
            plot_var_dic,
            chain_property,
            single_legend=True,
            nll=nll,
            **kwargs,
        )
    else:
        combine_plot = self.config["plot"].get("combine_plot", True)
        if not combine_plot:
            for dt, mc, sb, w_bkg, i in zip(
                data, phsp, bg, ws_bkg, range(self._Ngroup)
            ):
                data_dict, phsp_dict, bg_dict = self._cal_partial_wave(
                    amp,
                    params,
                    dt,
                    mc,
                    sb,
                    w_bkg,
                    prefix + "d{}_".format(i),
                    plot_var_dic,
                    chain_property,
                    save_root=save_root,
                    **kwargs,
                )
                self._plot_partial_wave(
                    data_dict,
                    phsp_dict,
                    bg_dict,
                    prefix + "d{}_".format(i),
                    plot_var_dic,
                    chain_property,
                    nll=nll,
                    **kwargs,
                )
        else:

            for dt, mc, sb, w_bkg, i in zip(
                data, phsp, bg, ws_bkg, range(self._Ngroup)
            ):
                data_dict, phsp_dict, bg_dict = self._cal_partial_wave(
                    amp,
                    params,
                    dt,
                    mc,
                    sb,
                    w_bkg,
                    prefix + "d{}_".format(i),
                    plot_var_dic,
                    chain_property,
                    save_root=save_root,
                    res=res,
                    **kwargs,
                )
                # self._plot_partial_wave(data_dict, phsp_dict, bg_dict, path+'d{}_'.format(i), plot_var_dic, chain_property, **kwargs)
                if i == 0:
                    datas_dict = {}
                    for ct in data_dict:
                        datas_dict[ct] = [data_dict[ct]]
                    phsps_dict = {}
                    for ct in phsp_dict:
                        phsps_dict[ct] = [phsp_dict[ct]]
                    bgs_dict = {}
                    for ct in bg_dict:
                        bgs_dict[ct] = [bg_dict[ct]]
                else:
                    for ct in data_dict:
                        datas_dict[ct].append(data_dict[ct])
                    for ct in phsp_dict:
                        phsps_dict[ct].append(phsp_dict[ct])
                    for ct in bg_dict:
                        bgs_dict[ct].append(bg_dict[ct])
            for ct in datas_dict:
                datas_dict[ct] = np.concatenate(datas_dict[ct])
            for ct in phsps_dict:
                phsps_dict[ct] = np.concatenate(phsps_dict[ct])
            for ct in bgs_dict:
                bgs_dict[ct] = np.concatenate(bgs_dict[ct])
            self._plot_partial_wave(
                datas_dict,
                phsps_dict,
                bgs_dict,
                prefix + "com_",
                plot_var_dic,
                chain_property,
                nll=nll,
                **kwargs,
            )
            if has_uproot and save_root:
                if bg[0] is None:
                    save_dict_to_root(
                        [datas_dict, phsps_dict],
                        file_name=prefix + "variables_com.root",
                        tree_name=["data", "fitted"],
                    )
                else:
                    save_dict_to_root(
                        [datas_dict, phsps_dict, bgs_dict],
                        file_name=prefix + "variables_com.root",
                        tree_name=["data", "fitted", "sideband"],
                    )
                print("Save root file " + prefix + "com_variables.root")

@ConfigLoader.register_function()
def make_plots(
    self,
    phsp_dict,
    prefix,
    plot_var_dic,
    chain_property,
    save_pdf=False,
    bin_scale=3,
    single_legend=False,
    format="pdf",
    smooth=True,
    color_first=True,
    **kwargs
):

    colors = plt.get_cmap('tab10').colors
    linestyles = ["-", "--", "-.", ":"]

    for name in plot_var_dic:
        phsp_i = phsp_dict[name + "_MC"]

        display = plot_var_dic[name]["display"]
        upper_ylim = plot_var_dic[name]["upper_ylim"]
        has_legend = plot_var_dic[name]["legend"]
        bins = plot_var_dic[name]["bins"]
        units = plot_var_dic[name]["units"]
        xrange = plot_var_dic[name]["range"]
        yscale = plot_var_dic[name].get("yscale", "linear")
        if xrange is None:
            xrange = [np.min(phsp_i) - 0.1, np.max(phsp_i) + 0.1]

        fig = plt.figure()
        ax = plt.subplot()

        legends = []
        legends_label = []

        if color_first:
            style = itertools.product(linestyles, colors)
        else:
            style = itertools.product(colors, linestyles)
        for i, name_i, label, curve_style in chain_property:
            weight_i = phsp_dict["MC_{0}_{1}_fit".format(i, name_i)]
            hist_i = Hist1D.histogram(
                phsp_i,
                weights=weight_i,
                range=xrange,
                bins=bins * bin_scale,
            )
            if smooth:
                if curve_style is None:
                    if color_first:
                        ls, color = next(style)
                    else:
                        color, ls = next(style)
                    le3 = hist_i.draw_kde(
                        ax,
                        label=label,
                        color=color,
                        linestyle=ls,
                        linewidth=1,
                    )
                else:
                    le3 = hist_i.draw_kde(
                        ax, fmt=curve_style, label=label, linewidth=1
                    )
            else:
                if curve_style is None:
                    if color_first:
                        ls, color = next(style)
                    else:
                        color, ls = next(style)
                    le3 = hist_i.draw(
                        ax,
                        label=label,
                        color=color,
                        linestyle=ls,
                        linewidth=1,
                    )
                else:
                    le3 = hist_i.draw(
                        ax,
                        color=curve_style[0],
                        linestyle=curve_style[1:],
                        label=label,
                        linewidth=1,
                    )
            legends.append(le3[0])
            legends_label.append(label)
        if yscale == "log":
            ax.set_ylim((0.1, upper_ylim))
        else:
            ax.set_ylim((0, upper_ylim))
        ax.set_xlim(xrange)
        ax.set_yscale(yscale)
        if has_legend:
            leg = ax.legend(
                legends,
                legends_label,
                frameon=False,
                labelspacing=0.1,
                borderpad=0.0,
            )
        ax.set_xlabel(display + units)

        fig.savefig(prefix + name + "." + format, dpi=300)
        if single_legend:
            export_legend(ax, prefix + "legend.{}".format(format))
        if save_pdf:
            fig.savefig(prefix + name + ".pdf", dpi=300)
            if single_legend:
                export_legend(ax, prefix + "legend.pdf")
        print("Finish plotting " + prefix + name)
        plt.close(fig)

@ConfigLoader.register_function()
def calculate_weights(
    self,
    amp,
    params,
    phsp,
    prefix,
    plot_var_dic,
    chain_property,
    res,
    save_root=False,
    bin_scale=3,
    batch=65000,
    **kwargs
):
    phsp_dict = {}
    with amp.temp_params(params):
        # weights_i = [amp(i) for i in data_split(phsp, batch)]
        # weight_phsp = data_merge(*weights_i)  # amp(phsp)
        # total_weight = (
        #     weight_phsp * phsp.get("weight", 1.0) * phsp.get("eff_value", 1.0)
        # )
        # norm_frac = 1. / np.sum(total_weight)
        # if res is None:
        #     weights = amp.partial_weight(phsp)
        # else:
        weights = []
        used_res = amp.used_res
        for i in res:
            if not isinstance(i, list):
                i = [i]
            amp.set_used_res(i)
            weights.append(amp(phsp))
        # print(weights, amp.decay_group.chains_idx)
        amp.set_used_res(used_res)

        # data_weights = data.get("weight", np.ones((data_shape(data),)))
        # data_dict["data_weights"] = data_weights
        # phsp_weights = total_weight * norm_frac
        # phsp_dict["MC_total_fit"] = phsp_weights  # MC total weight
        for i, name_i, label, _ in chain_property:
            weight_i = (
                weights[i]
                * phsp.get("weight", 1.0)
                * phsp.get("eff_value", 1.0)
            )
            phsp_dict[
                "MC_{0}_{1}_fit".format(i, name_i)
            ] = weight_i  # MC partial weight
        for name in plot_var_dic:
            idx = plot_var_dic[name]["idx"]
            trans = lambda x: np.reshape(plot_var_dic[name]["trans"](x), (-1,))

            phsp_i = trans(data_index(phsp, idx))
            phsp_dict[name + "_MC"] = phsp_i  # MC

    phsp_dict = data_to_numpy(phsp_dict)
    return phsp_dict


@ConfigLoader.register_function()
def plot_resonance(
    self,
    params=None,
    phsp=None,
    prefix="figure/",
    res=None,
    save_root=False,
    **kwargs
):
    if params is None:
        params = {}

    path = os.path.dirname(prefix)
    os.makedirs(path, exist_ok=True)

    amp = self.get_amplitude()
    phsp = phsp if phsp is not None else self.get_phsp_plot()
    self._Ngroup = len(phsp)

    ## check that group labels and mothers are not repeated
    groups = self.plot_params.config["groups"]
    group_labels = set()
    mothers = set()
    for group in groups:
        label = group["label"]
        if label not in group_labels:
            group_labels.add(label)
        else:
            raise ValueError(f"{label} is repeated in plot group labels")
        for mother in group["mothers"]:
            if mother not in mothers:
                mothers.add(mother)
            else:
                raise ValueError(f"Mother particle <{mother}> is repeated in plot group")

    # ws_bkg, ws_inmc = self._get_bg_weight(data, bg)
    chain_property = []
    for i, group in enumerate(groups):
        name_i = "_".join(group["mothers"])
        display = group["label"]
        chain_property.append([i, name_i, display, None])
    res = [group["mothers"] for group in groups]

    plot_var_dic = {}
    for conf in self.plot_params.get_params():
        name = conf.get("name")
        display = conf.get("display", name)
        upper_ylim = conf.get("upper_ylim", None)
        idx = conf.get("idx")
        trans = conf.get("trans", lambda x: x)
        has_legend = conf.get("legend", False)
        xrange = conf.get("range", None)
        bins = conf.get("bins", None)
        units = conf.get("units", "")
        yscale = conf.get("yscale", "linear")
        plot_var_dic[name] = {
            "display": display,
            "upper_ylim": upper_ylim,
            "legend": has_legend,
            "idx": idx,
            "trans": trans,
            "range": xrange,
            "bins": bins,
            "units": units,
            "yscale": yscale,
        }
    if self._Ngroup == 1:
        phsp_dict = self.calculate_weights(
            amp,
            params,
            phsp[0],
            prefix,
            plot_var_dic,
            chain_property,
            res=res,
            save_root=save_root,
            **kwargs,
        )
        self.make_plots(
            phsp_dict,
            prefix,
            plot_var_dic,
            chain_property,
            **kwargs,
        )
    else:
        for dt, mc, sb, w_bkg, i in zip(
            data, phsp, bg, ws_bkg, range(self._Ngroup)
        ):
            phsp_dict = self.calculate_weights(
                amp,
                params,
                mc,
                prefix + "d{}_".format(i),
                plot_var_dic,
                chain_property,
                res=res,
                save_root=save_root,
                **kwargs,
            )
            # self._plot_partial_wave(data_dict, phsp_dict, bg_dict, path+'d{}_'.format(i), plot_var_dic, chain_property, **kwargs)
            if i == 0:
                phsps_dict = {}
                for ct in phsp_dict:
                    phsps_dict[ct] = [phsp_dict[ct]]
            else:
                for ct in phsp_dict:
                    phsps_dict[ct].append(phsp_dict[ct])
        for ct in phsps_dict:
            phsps_dict[ct] = np.concatenate(phsps_dict[ct])
        self.make_plots(
            phsps_dict,
            prefix + "com_",
            plot_var_dic,
            chain_property,
            **kwargs,
        )

parser = argparse.ArgumentParser(description="Script to plot invariant mass")
parser.add_argument("config", type=str, help="Input config file.")
parser.add_argument("prefix", type=str, help="Output prefix.")
parser.add_argument("--single-resonance", action="store_true")
parser.add_argument("--save_root", action="store_true")
arguments = parser.parse_args()

import matplotlib.pyplot as plt
plt.rcParams["xtick.minor.visible"] = True

os.makedirs(arguments.prefix, exist_ok=True)
config = ConfigLoader(arguments.config)
config.set_params("final_params.json")
if arguments.single_resonance:
    config.plot_resonance(plot_pull=True, prefix=arguments.prefix)
else:
    config.plot_grouped_partial_waves(plot_pull=True, prefix=arguments.prefix, save_root=arguments.save_root)
