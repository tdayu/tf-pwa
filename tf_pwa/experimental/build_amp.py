import tensorflow as tf

from tf_pwa.data import data_shape
from tf_pwa.utils import time_print

from .opt_int import split_gls


def build_sum_amplitude(dg, dec_chain, data):
    cached = []
    for i, dc in split_gls(dec_chain):
        amp = dg.get_amp(data)
        int_mc = amp
        m_dep = dg.get_m_dep(data)
        m_dep_all = 1.0
        for i in m_dep:
            for j in i:
                m_dep_all *= tf.reshape(j, (-1,))
        m_dep_all = tf.reshape(m_dep_all, [-1] + [1] * (len(amp.shape) - 1))
        cached.append(amp / m_dep_all)
    return cached


def build_amp_matrix(dec, data, weight=None):
    hij = []
    used_chains = dec.chains_idx
    index = []
    for k, i in enumerate(dec):
        dec.set_used_chains([k])
        tmp = []
        for j, amp in enumerate(build_sum_amplitude(dec, i, data)):
            tmp.append(amp)
        hij.append(tmp)
    dec.set_used_chains(used_chains)
    # print([i.shape for i in hij.values()])
    # print([[j.shape for j in i] for i in hij])
    return index, hij


def _prod(ls):
    ret = 1
    for i in ls:
        ret *= i
    return ret


def build_params_vector(dg, data):
    n_data = data_shape(data)
    m_dep = dg.get_m_dep(data)
    ret = []
    for i in m_dep:
        tmp = i[0]
        if tmp.shape[0] == 1:
            tmp = tf.tile(tmp, [n_data] + [1] * (len(tmp.shape) - 1))
        tmp = tf.reshape(tmp, (-1, _prod(tmp.shape[1:])))
        for j in i[1:]:
            tmp2 = tf.reshape(j, (-1, _prod(j.shape[1:])))
            tmp = tmp[:, :, None] * tmp2[:, None, :]
            tmp = tf.reshape(tmp, (-1, _prod(tmp.shape[1:])))
        ret.append(tmp)
    return ret


def cached_amp2s(dg, data):

    _amp = cached_amp(dg, data)

    @time_print
    # @tf.function
    def _amp2s():  # pragma: no cover # because of tf_funciton
        amp = _amp()
        amp2s = tf.math.real(amp * tf.math.conj(amp))
        return tf.reduce_sum(amp2s, list(range(1, len(amp2s.shape))))

    return _amp2s


def build_amp2s(dg):
    # @tf.function(experimental_relax_shapes=True)
    def _amp2s(data, cached_data):
        n_data = data_shape(data)
        pv = build_params_vector(dg, data)
        ret = []
        for i, j in zip(pv, cached_data):
            # print(j)
            # print(i.shape)
            a = tf.reshape(i, [-1, i.shape[1]] + [1] * (len(j[0].shape) - 1))
            ret.append(tf.reduce_sum(a * tf.stack(j, axis=1), axis=1))
        # print(ret)
        amp = tf.reduce_sum(ret, axis=0)
        amp2s = tf.math.real(amp * tf.math.conj(amp))
        return tf.reduce_sum(amp2s, list(range(1, len(amp2s.shape))))

    return _amp2s


def build_sum_angle_amplitude(dg, dec_chain, data):
    cached = []
    for i, dc in split_gls(dec_chain):
        amp = dg.get_angle_amp(data)
        cached.append(amp)
    return cached


def build_angle_amp_matrix(dec, data, weight=None):
    hij = []
    used_chains = dec.chains_idx
    for k, i in enumerate(dec):
        dec.set_used_chains([k])
        tmp = []
        for j, amp in enumerate(build_sum_angle_amplitude(dec, i, data)):
            tmp.append(amp)
        hij.append(tmp)
    dec.set_used_chains(used_chains)
    return list(dec), hij


def amp_matrix_as_dict(dec, hij):
    ret = {}
    for i, d in zip(hij, dec):
        ret[d] = i
    return ret


def cached_amp(dg, data, matrix_method=build_angle_amp_matrix):

    idx, c_amp = matrix_method(dg, data)
    n_data = data_shape(data)

    # @tf.function
    def _amp():
        pv = build_params_vector(dg, data)
        ret = []
        for i, j in zip(pv, c_amp):
            a = tf.reshape(i, [n_data, -1] + [1] * (len(j[0].shape) - 1))
            ret.append(tf.reduce_sum(a * tf.stack(j, axis=1), axis=1))
        # print(ret)
        amp = tf.reduce_sum(ret, axis=0)
        return amp

    return _amp
