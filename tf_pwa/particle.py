"""
This module implements classes to describe particles and decays.
"""
import functools
from collections import UserList

import numpy as np
import tensorflow as tf

from .breit_wigner import barrier_factor as default_barrier_factor
from .cg import cg_coef
from .config import get_config
from .utils import deep_ordered_iter

# from pysnooper import snoop


DEFAULT_DECAY = "defalut_decay"


def cross_combine(x):
    """
    Combine every two of a list, as well as give every one of them.???
    Can be put to utils.py

    :param x:
    :return:
    """
    if not x:  # if x is []
        return []
    head = x[0]
    tail = x[1:]
    ret = []
    other = cross_combine(tail)
    for i in head:
        if not other:
            ret.append(i)
        else:
            for j in other:
                ret.append(i + j)
    return ret


def _spin_range(a, b):
    while a <= b:
        yield a
        a = a + 1


def _spin_int(x):
    if isinstance(x, int):
        return x
    return int(x + 0.1)


@functools.total_ordering
class BaseParticle(object):
    """
    Base Particle object. Name is "name[:id]".

    :param name: String. Name of the particle
    :param J: Integer or half-integer. The total spin
    :param P: 1 or -1. The parity
    :param spins: List. The spin quantum numbers. If it's not provided, ``spins`` will be ``tuple(range(-J, J + 1))``.
    :param mass: Real variable
    :param width: Real variable
    """

    def __init__(
        self,
        name,
        J=0,
        P=-1,
        C=None,
        spins=None,
        mass=None,
        width=None,
        id_=None,
        disable=False,
        **kwargs
    ):
        self.set_name(name, id_)
        self.decay = []  # list of Decay
        self.creators = []  # list of Decay which creates the particle
        if isinstance(J, str):
            J = eval(J)
        self.J = J
        self.P = P
        self.C = C
        if spins is None:
            spins = tuple(_spin_range(-J, J))
        self.spins = tuple(
            [eval(i) if isinstance(i, str) else i for i in spins]
        )
        self.mass = mass if mass else tf.Variable(0., dtype=tf.float64)
        self.width = width
        self.disable = disable
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set_name(self, name, id_=None):
        if id_ is None:
            names = name.split(":")
            if len(names) > 1:
                self._name = ":".join(names[:-1])
                try:
                    self._id = int(names[-1])
                except ValueError:
                    self._name, self._id = name, 0
            else:
                self._name, self._id = name, 0
        else:
            self._name, self._id = name, id_

    @property
    def name(self):
        return self._name

    def add_decay(self, d):
        """
        :param d: BaseDecay object
        """
        if d not in self.decay:
            self.decay.append(d)

    def remove_decay(self, d):
        """
        :param d: BaseDecay object
        """
        self.decay.remove(d)

    def add_creator(self, d):
        """
        Add a decay reaction where the particle is created.

        :param d: BaseDecay object
        """
        self.creators.append(d)

    def __repr__(self):
        if self._id == 0:
            return self._name
        return "{}:{}".format(self.name, self._id)

    def __hash__(self):
        return hash((self.name, self._id))

    def __eq__(self, other):
        if not isinstance(other, BaseParticle):
            return False
        return (self.name, self._id) == (other.name, other._id)

    def __lt__(self, other):
        if isinstance(other, BaseParticle):
            return (self.name, self._id) < (other.name, other._id)
        return self.__repr__() < other

    def __add__(self, other):
        return ParticleList([self]) + other

    def __lshift__(self, other):
        _BaseDecay, kwargs = get_config(DEFAULT_DECAY, (BaseDecay, {}))
        return _BaseDecay(self, other, **kwargs)

    def chain_decay(self):
        """return all decay chain self decay to"""
        ret = []
        for i in self.decay:
            ret_tmp = [[[i]]]
            for j in i.outs:
                tmp = j.chain_decay()
                if tmp:  # if tmp is not []
                    ret_tmp.append(tmp)
            ret += cross_combine(ret_tmp)
        return ret

    def get_resonances(self):
        """return all resonances self decay to"""
        decay_chain = self.chain_decay()
        chains = [DecayChain(i) for i in decay_chain]
        decaygroup = DecayGroup(chains)
        return decaygroup.resonances


class ParticleList(UserList):
    """List for Particle"""

    def __add__(self, other):
        if isinstance(other, BaseParticle):
            other = [other]
        return super(ParticleList, self).__add__(other)


def GetA2BC_LS_list(
    ja, jb, jc, pa=None, pb=None, pc=None, p_break=False, ca=None
):
    """
    The :math:`L-S` coupling for the decay :math:`A\\rightarrow BC`, where :math:`L` is the orbital
    angular momentum of :math:`B` and :math:`B`, and :math:`S` is the superposition of their spins.
    It's required that :math:`|J_B-J_C|\leq S \leq J_B+J_C` and :math:`|L-S|\leq J_A \leq L+S`. It's also required by the conservation of
    P parity that :math:`L` is keep :math:`P_A = P_B P_C  (-1)^{l}`.

    :param ja: `J` of particle `A`
    :param jb: `J` of particle `B`
    :param jc: `J` of particle `C`
    :param pa: `P` of particle `A`
    :param pb: `P` of particle `B`
    :param pc: `P` of particle `C`
    :param p_break: allow p voilate
    :param ca: enabel c partity select c=(-1)^(l+s)
    :return: List of :math:`(l,s)` pairs.
    """
    if pa is None or pb is None or pc is None:
        p_break = True
    if not p_break:
        dl = 0 if pa * pb * pc == 1 else 1  # pa = pb * pc * (-1)^l
    s_min = abs(jb - jc)
    s_max = jb + jc
    # ns = s_max - s_min + 1
    ret = []
    for s in _spin_range(s_min, s_max):
        for l in _spin_range(abs(ja - s), ja + s):
            if l % 1.0 < 0.7 and l % 1.0 > 0.3:  # half int
                break
            l = _spin_int(l)
            if ca is not None:
                if ca != (-1) ** (l + s):
                    continue
            if not p_break:
                if l % 2 == dl:
                    ret.append((l, s))
            else:
                ret.append((l, s))
    # print(ret, ja, jb, jc, pa, pb, pc)
    return ret


def simple_cache_fun(f):
    """

    :param f:
    :return:
    """
    name = "simple_cached_" + f.__name__

    @functools.wraps(f)
    def g(self):
        if not hasattr(self, name):
            setattr(self, name, f(self))
        return getattr(self, name)

    return g


@functools.total_ordering
class BaseDecay(object):
    """
    Base Decay object

    :param core: Particle. The mother particle
    :param outs: List of Particle. The daughter particles
    :param name: String. Name of the decay
    :param disable: Boolean. If it's True???
    """

    def __init__(
        self,
        core,
        outs,
        name=None,
        disable=False,
        p_break=False,
        c_break=True,
        curve_style=None,
        **kwargs
    ):
        self._name = name
        self.core = core
        self.outs = tuple(outs)
        self.p_break = p_break
        self.c_break = c_break
        if hasattr(self.core, "disable") and self.core.disable:
            disable = True
        for i in self.outs:
            if hasattr(i, "disable") and i.disable:
                disable = True
        if not disable:
            self.core.add_decay(self)
            for i in outs:
                i.add_creator(self)
        self.curve_style = curve_style
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def name(self):
        return self._name

    def __repr__(self):
        ret = str(self.core)
        ret += "->"
        ret += "+".join([str(i) for i in self.outs])
        return ret  # "A->B+C"

    @simple_cache_fun  # @functools.lru_cache()
    def get_id(self):
        return (self.core, tuple(sorted(self.outs)))

    def __hash__(self):
        return hash(self.get_id())

    def __eq__(self, other):
        if not isinstance(other, BaseDecay):
            return False
        return self.get_id() == other.get_id()

    def __lt__(self, other):
        if not isinstance(other, BaseDecay):
            return False
        return self.get_id() < other.get_id()


class Decay(BaseDecay):  # add useful methods to BaseDecay
    """
    General Decay object
    """

    @simple_cache_fun  # functools.lru_cache()
    def get_ls_list(self):
        """
        It has interface to ``tf_pwa.particle.GetA2BC_LS_list(ja, jb, jc, pa, pb, pc)``
        :return: List of (l,s) pairs
        """
        ca = self.core.C if not self.c_break else None
        ja = self.core.J
        jb = self.outs[0].J
        jc = self.outs[1].J
        pa = self.core.P
        pb = self.outs[0].P
        pc = self.outs[1].P
        return tuple(
            GetA2BC_LS_list(
                ja, jb, jc, pa, pb, pc, p_break=self.p_break, ca=ca
            )
        )

    # @functools.lru_cache()
    def get_l_list(self):
        """
        List of l in ``self.get_ls_list()``
        """
        return tuple([l for l, s in self.get_ls_list()])

    @functools.lru_cache()
    def get_min_l(self):
        """
        The minimal l in the LS coupling
        """
        return min(self.get_l_list())

    def generate_params(self, name=None, _ls=True):
        """
        Generate the name of the variable for every (l,s) pair. In PWA, the variable is usually expressed as
        :math:`g_ls`.

        :param name: String. It is the name of the decay by default
        :param _ls: ???
        :return: List of strings
        """
        if name is None:
            name = self.name
        ret = []
        for l, s in self.get_ls_list():
            name_r = "{name}_l{l}_s{s}_r".format(name=name, l=l, s=s)
            name_i = "{name}_l{l}_s{s}_i".format(name=name, l=l, s=s)
            ret.append((name_r, name_i))
        return ret

    @functools.lru_cache()
    def get_cg_matrix(self):  # CG factor inside H
        """
        The matrix indexed by :math:`[(l,s),(\\lambda_b,\\lambda_c)]`. The matrix element is

        .. math::
            \\sqrt{\\frac{ 2 l + 1 }{ 2 j_a + 1 }}
            \\langle j_b, j_c, \\lambda_b, - \\lambda_c | s, \\lambda_b - \\lambda_c \\rangle
            \\langle l, s, 0, \\lambda_b - \\lambda_c | j_a, \\lambda_b - \\lambda_c \\rangle

        This is actually the pre-factor of :math:`g_ls` in the amplitude formula.

        :return: 2-d array of real numbers
        """
        ls = self.get_ls_list()
        m = len(ls)
        ja = self.core.J
        jb = self.outs[0].J
        jc = self.outs[1].J
        n = (2 * jb + 1) * (2 * jc + 1)
        ret = np.zeros(shape=(n, m))
        for i, ls_i in enumerate(ls):
            l, s = ls_i
            j = 0
            for lambda_b in range(-jb, jb + 1):
                for lambda_c in range(-jc, jc + 1):
                    ret[j][i] = (
                        np.sqrt((2 * l + 1) / (2 * ja + 1))
                        * cg_coef(
                            jb, jc, lambda_b, -lambda_c, s, lambda_b - lambda_c
                        )
                        * cg_coef(
                            l,
                            s,
                            0,
                            lambda_b - lambda_c,
                            ja,
                            lambda_b - lambda_c,
                        )
                    )
                    j += 1
        return ret

    def barrier_factor(self, q, q0):  # Barrier factor inside H
        """
        The cached barrier factors with :math:`d=3.0` for all :math:`l`. For barrier factor, refer to
        **tf_pwa.breit_wigner.Bprime(L, q, q0, d)**

        :param q: Data array
        :param q0: Real number
        :return: 1-d array for every data point
        """
        d = 3.0
        ret = default_barrier_factor(self.get_l_list(), q, q0, d)
        return ret


def split_particle_type_list(decays):
    """
    Separate initial particle, intermediate particles, final particles in a decay chain.

    :param decays: DecayChain
    :return: Set of initial Particle, set of intermediate Particle, set of final Particle
    """
    core_particles = []
    out_particles = []

    for i in decays:
        core_particles.append(i.core)
        for j in i.outs:
            out_particles.append(j)

    inner = [
        i for i in core_particles if i in out_particles
    ]  # core_particles & out_particles
    top = [
        i for i in core_particles if i not in inner
    ]  # core_particles - inner
    outs = [
        i for i in out_particles if i not in inner
    ]  # out_particles - inner
    # print(decays, inner, top, outs)
    return top, inner, outs


def split_particle_type(decays):
    top, inner, outs = split_particle_type_list(decays)
    return set(top), set(inner), set(outs)


def split_len(dicts):
    """
    Split a dictionary of lists by their length.
    E.g. {"b":[2],"c":[1,3],"d":[1]} => [None, [('b', [2]), ('d', [1])], [('c', [1, 3])]]

    Put to utils.py or _split_len if not used anymore???

    :param dicts: Dictionary
    :return: List of dictionary
    """
    size_table = []
    for i in dicts:
        tmp = dicts[i]
        size_table.append((len(tmp), i))
    max_l = max([i for i, _ in size_table])
    ret = [None] * (max_l + 1)
    for i, s in size_table:
        if ret[i] is None:
            ret[i] = []
        ret[i].append((s, dicts[s]))
    return ret


class DecayChain(object):
    """
    A decay chain. E.g. :math:`A\\rightarrow BC, B\\rightarrow DE`

    :param chain: ???
    """

    def __init__(self, chain):
        self.chain = chain
        top, inner, outs = split_particle_type_list(chain)
        assert len(top) == 1, "top particles must be only one particle"
        self.top = top.pop()
        self.inner = sorted(list(inner))
        self.outs = sorted(list(outs))

    def __iter__(self):
        return iter(self.chain)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.chain[idx]
        raise NotImplementedError

    def __repr__(self):
        return "{}".format(list(self.chain))

    @simple_cache_fun  # functools.lru_cache()
    def sorted_table(self):
        """
        A topology independent structure.
        E.g. [a->rb,r->cd] => {a:[b,c,d],r:[c,d],b:[b],c:[c],d:[d]}

        :return: Dictionary indexed by Particle
        """
        decay_dict = {}
        for i in self.outs:
            decay_dict[i] = [i]

        chain = self.chain
        while chain:
            tmp_chain = []
            for i in chain:
                if all([j in decay_dict for j in i.outs]):
                    decay_dict[i.core] = []
                    for j in i.outs:
                        decay_dict[i.core] += decay_dict[j]
                    decay_dict[i.core].sort()
                else:
                    tmp_chain.append(i)
            chain = tmp_chain
        decay_dict[self.top] = sorted(list(self.outs))
        return decay_dict

    @simple_cache_fun
    def sorted_table_layers(self):
        """
        Get the layer of decay chain as sorted table.
        Or the list of particle with the same number of final particles.
        So, the first item is always None.
        E.g. [a->rb,r->cd] => [None, [(b, [b]), (c, [c]), (d, [d])], [(r, [c, d])], [(a, [b, c, d])]]

        :return: List of dictionary
        """
        st = self.sorted_table()
        return split_len(st)

    @staticmethod
    def from_sorted_table(decay_dict):
        """
        Create decay chain from a topology independent structure.
        E.g. {a:[b,c,d],r:[c,d],b:[b],c:[c],d:[d]} => [a->rb,r->cd]

        :param decay_dict: Dictionary
        :return: DecayChain
        """

        def sum_list(ls):
            ret = ls[0]
            for i in ls[1:]:
                ret = ret + i
            return ret

        def deep_search(idx, base):
            base_step = 2
            max_step = len(base)
            found = False
            while base_step <= max_step:
                for i in deep_ordered_iter(base, base_step):
                    check = sum_list([base[j] for j in i])
                    if sorted(check) == sorted(idx[1]):
                        found = True
                        return i
                base_step += 1
            if not found:
                raise Exception("not found in searching")

        s_dict = split_len(decay_dict)
        base_dict = dict(s_dict[1])
        ret = []
        for s_dict_i in s_dict[2:]:
            if s_dict_i:
                for j in s_dict_i:
                    found = deep_search(j, base_dict)
                    ret.append(BaseDecay(j[0], found, disable=True))
                    for i in found:
                        del base_dict[i]
                    base_dict[j[0]] = j[1]
        return DecayChain(ret)

    @staticmethod
    def from_particles(top, finals):
        """
        Build possible decay chain Topology.
        E.g. a -> [b,c,d] => [[a->rb,r->cd],[a->rc,r->bd],[a->rd,r->bc]]

        :param top: Particle
        :param finals: List of Particle
        :return: DecayChain
        """
        assert len(finals) > 0, " "

        def get_graphs(g, ps):
            if ps:
                p = ps[0]
                ps = ps[1:]
                ret = []
                for i in g.edges:
                    gi = g.copy()
                    gi.add_node(i, p)
                    ret += get_graphs(gi, ps)
                return ret
            return [g]

        base = _Chain_Graph()
        base.add_edge(top, finals[0])
        gs = get_graphs(base, finals[1:])
        return [
            gi.get_decay_chain(top, head="chain{}_".format(i))
            for i, gi in enumerate(gs)
        ]

    @functools.lru_cache()
    def topology_id(self, identical=True):
        """
        topology identy

        :param identical: allow identical particle in finals
        :return:
        """
        a = self.sorted_table()
        if identical:
            set_a = [[j.name for j in a[i]] for i in a]
        else:
            set_a = [list(a[i]) for i in a]
        return sorted(set_a)

    def __eq__(self, other):
        if not isinstance(other, DecayChain):
            return False
        return self.get_id() == other.get_id()

    def __lt__(self, other):
        if not isinstance(other, DecayChain):
            return False
        return self.get_id() < other.get_id()

    @simple_cache_fun
    def __hash__(self):
        return hash(self.get_id())

    @simple_cache_fun
    def get_id(self):
        """return identity of the decay"""
        decay = sorted(list(self.chain))
        return tuple(decay)

    def standard_topology(self):
        """
        standard topology structure of the decay chain,
        all inner particle will be replace as a tuple of out particles.
        for example [A->R+C, R->B+D], is [A->(B, D)+C, (B, D)->B+D]
        """
        a = self.sorted_table()
        name_map = {}
        for k, v in a.items():
            parts = sorted([str(i) for i in v])
            name_map[k] = "({})".format(", ".join(parts))
        name_map[self.top] = str(self.top)
        for i in self.outs:
            name_map[i] = str(i)
        particle_map = {k: BaseParticle(v) for k, v in name_map.items()}
        ret = []
        for i in self:
            core = particle_map[i.core]
            ret.append(BaseDecay(core, [particle_map[j] for j in i.outs]))
        return DecayChain(ret)

    def topology_map(self, other=None):
        """Mapping relations of the same topology decay
        E.g. [A->R+B,R->C+D],[A->Z+B,Z->C+D] => {A:A,B:B,C:C,D:D,R:Z,A->R+B:A->Z+B,R->C+D:Z->C+D}
        """
        a = self.sorted_table()
        if other is None:
            other = self.standard_topology()
        b = other.sorted_table()
        ret = {}
        for i in a:
            for j in b:
                if a[i] == b[j]:
                    ret[i] = j
                    break
        for i in self:
            test_decay = BaseDecay(
                ret[i.core], [ret[k] for k in i.outs], disable=False
            )
            for j in other:
                if test_decay == j:
                    ret[i] = j
                    break
        return ret

    def topology_same(self, other, identical=True):
        """
        whether self and other is the same topology

        :param other: other decay chains
        :param identical: using identical particles
        :return:
        """
        if not isinstance(other, DecayChain):
            raise TypeError("unsupport type {}".format(type(other)))
        return self.topology_id(identical) == other.topology_id(identical)

    def get_all_particles(self):
        """
        get all particles in the decay chains
        """
        ret = []
        for i in self:
            if i.core not in ret:
                ret.append(i.core)
            for j in i.outs:
                if j not in ret:
                    ret.append(j)
        return ret

    def get_particle_decay(self, particle):
        """
        get particle decay in the chain
        """
        p = BaseParticle(str(particle))
        for i in self:
            if i.core == p:
                return i
        raise ValueError("Not found particle {} decay".format(p))

    def depth_first(self, node_first=True):
        """
        depth first travel for decay
        """
        node_map = {i.core: i for i in self.chain}

        def _dep(t, d):
            dec = node_map.get(t, None)
            if dec is not None:
                if node_first:
                    yield d, dec
                for i in dec.outs:
                    yield from _dep(i, d + 1)
                if not node_first:
                    yield d, dec

        return _dep(self.top, 1)


class _Chain_Graph(object):
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.count = 0

    def add_edge(self, a, b):
        self.edges.append((a, b))

    def add_node(self, e, d):
        self.edges.remove(e)
        count = self.count
        node = "node_{}".format(count)
        self.nodes.append(node)
        self.edges.append((e[0], node))
        self.edges.append((node, e[1]))
        self.edges.append((node, d))
        self.count += 1

    def copy(self):
        ret = _Chain_Graph()
        ret.nodes = self.nodes.copy()
        ret.edges = self.edges.copy()
        ret.count = self.count
        return ret

    def get_decay_chain(self, top, head="tmp_"):
        decay_list = {}
        ret = []
        inner_particle = {}
        for i in self.nodes:
            inner_particle[i] = BaseParticle("{}{}".format(head, i))
        for i, j in self.edges:
            i = inner_particle.get(i, i)
            j = inner_particle.get(j, j)
            if i in decay_list:
                decay_list[i].append(j)
            else:
                decay_list[i] = [j]
        assert len(decay_list[top]) == 1, ""
        tmp = decay_list[top][0]
        decay_list[top] = decay_list[tmp]
        del decay_list[tmp]
        for i in decay_list:
            tmp = BaseDecay(i, decay_list[i], disable=True)
            ret.append(tmp)
        return DecayChain(ret)


class DecayGroup(object):
    """
    A group of two-body decays.

    :param chains: List of DecayChain
    """

    def __init__(self, chains):
        first_chain = chains[0]
        if not isinstance(first_chain, DecayChain):
            chains = [DecayChain(i) for i in chains]
            first_chain = chains[0]
        self.chains = chains
        self.top = first_chain.top
        self.outs = sorted(list(first_chain.outs))
        for i in chains:
            assert i.top == first_chain.top, ""
            assert (
                i.outs == first_chain.outs
            ), "{} and {} particles are different".format(i, first_chain)
        # resonances = set()
        resonances = []
        for i in chains:
            # resonances |= i.inner
            for j in i.inner:
                if j not in resonances:
                    resonances.append(j)
        self.resonances = list(resonances)
        self.identical_particles = []

    def __repr__(self):
        return "{}".format(self.chains)

    def __iter__(self):
        return iter(self.chains)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.chains[idx]
        raise NotImplementedError

    def get_decay_chain(self, id_):
        """get decay chain from identity string"""
        if isinstance(id_, int):
            return self.chains[id_]
        if isinstance(id_, str):
            names = id_.split("/")
        elif isinstance(id_, list):
            names = [str(i) for i in id_]
        elif isinstance(id_, BaseParticle):
            names = [str(id_)]
        else:
            raise NotImplementedError
        return self.get_chain_from_particle(names)

    def topology_structure(self, identical=False, standard=True):
        """

        :param identical:
        :param standard:
        :return:
        """
        ret = []
        for i in self:
            for j in ret:
                if i.topology_same(j, identical):
                    break
            else:
                ret.append(i)
        if standard:
            return [i.standard_topology() for i in ret]
        return ret

    @functools.lru_cache()
    def get_chains_map(self, chains=None):
        """

        :param chains:
        :return:
        """
        if chains is None:
            chains = self.chains
        chain_maps = []
        for decays in self.topology_structure():
            decay_chain = DecayChain(list(decays))
            tmp = {}
            for j in chains:
                if decay_chain.topology_same(j):
                    chain_map = decay_chain.topology_map(j)
                    tmp[j] = chain_map
            chain_maps.append(tmp)
        return chain_maps

    def get_chain_from_particle(self, names):
        """
        get the first decay chain has all particles in names
        """
        particles = [BaseParticle(str(i)) for i in names]
        for decay_chain in self.chains:
            all_particles = list(decay_chain.get_all_particles())
            found = True
            for i in particles:
                if i not in all_particles:
                    found = False
                    break
            if found:
                return decay_chain
        raise ValueError("Not found such decay chain: {}".format(names))

    def get_particle(self, name):
        """
        get particle by name
        """
        for decay_chain in self.chains:
            all_particles = list(decay_chain.get_all_particles())
            for i in all_particles:
                if str(i) == str(name):
                    return i
        raise ValueError("Not found such decay chain: {}".format(name))
