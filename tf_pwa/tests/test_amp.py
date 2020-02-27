import tempfile
import os
import contextlib
import pytest

from tf_pwa.amp import *
from tf_pwa.cal_angle import cal_angle_from_momentum


@contextlib.contextmanager
def write_temp_file(s):
    a = tempfile.mktemp()
    with open(a, "w") as f:
        f.write(s)
    yield a
    os.remove(a)


@regist_particle("Gounaris–Sakurai")
class ParticleGS(Particle):
    def get_amp(self, data, data_c):
        r"""
        Gounaris G.J., Sakurai J.J., Phys. Rev. Lett., 21 (1968), pp. 244-247

        .. math::
          R(m) = \frac{1 + D \Gamma_0 / m_0}{(m_0^2 -m^2) + f(m) - i m_0 \Gamma(m)}

        .. math::
          f(m) = \Gamma_0 \frac{m_0 ^2 }{q_0^3} \left[q^2 [h(m)-h(m_0)] + (m_0^2 - m^2) q_0^2 \frac{d h}{d m}|_{m0} \right]

        .. math::
          h(m) = \frac{2}{\pi} \frac{q}{m} \ln \left(\frac{m+q}{2m_{\pi}} \right)

        .. math::
          \frac{d h}{d m}|_{m0} = h(m_0) [(8q_0^2)^{-1} - (2m_0^2)^{-1}] + (2\pi m_0^2)^{-1}

        .. math::
          D = \frac{f(0)}{\Gamma_0 m_0} = \frac{3}{\pi}\frac{m_\pi^2}{q_0^2} \ln \left(\frac{m_0 + 2q_0}{2 m_\pi }\right)
            + \frac{m_0}{2\pi q_0} - \frac{m_\pi^2 m_0}{\pi q_0^3}
        """
        raise NotImplementedError


def get_test_decay():
    a = Particle("A", J=1, P=-1, spins=(-1, 1))
    b = Particle("B", J=1, P=-1)
    c = Particle("C", J=0, P=-1)
    d = Particle("D", J=1, P=-1)
    bd = ParticleLass("BD", 1, 1, mass=1.0, width=1.0)
    cd = Particle("CD", 1, 1, mass=1.0, width=1.0)
    bc = Particle("BC", 1, 1, mass=1.0, width=1.0)
    R = get_particle("R", 1, 1, mass=1.0, width=1.0)
    HelicityDecay(a, [bc, d])
    HelicityDecay(bc, [b, c])
    HelicityDecay(a, [cd, b])
    HelicityDecay(cd, [c, d])
    HelicityDecay(a, [bd, c])
    get_decay(bd, [b, d])
    HelicityDecayNP(a, [R, c])
    HelicityDecayP(R, [b, d])
    de = DecayGroup(a.chain_decay())
    print(de)
    return de, [b, c, d]


test_data = [
    [
        np.array([[2.0, 0.1, 0.2, 0.3]]),
        np.array([[3.0, 0.2, 0.3, 0.4]]),
        np.array([[4.0, 0.3, 0.4, 0.5]])
    ],
    [
        np.array([[2.0, 0.1, 0.2, 0.3], [2.0, 0.3, 0.2, 0.1]]),
        np.array([[3.0, 0.2, 0.3, 0.4], [3.0, 0.4, 0.3, 0.2]]),
        np.array([[4.0, 0.3, 0.4, 0.5], [4.0, 0.5, 0.4, 0.3]])
    ]
]


def test_amp():
    decs, particle = get_test_decay()
    amp = AmplitudeModel(decs)
    for p_data in test_data: 
        p = dict(zip(particle, p_data))
        data = cal_angle_from_momentum(p, decs)
        amp(data)


def test_particle():
    a = get_particle("ss", model="Gounaris–Sakurai")
    with pytest.raises(NotImplementedError):
        a.get_amp({},{})


def test_dec():
    s = """

Particle D_2*+         2.4654          0.0467
Particle anti-D_2*0    2.4607          0.0475
RUNNINGWIDTH D_2*+

Decay vpho
1 anti-D_2*0  D*0    HELCOV gg6a aa6a gg7a aa7a gg8a aa8a gg9a aa9a  one zero;
1 D_2*+  D*-         HELCOV gg6a aa6a gg7a aa7a gg8a aa8a gg9a aa9a  one zero;
Enddecay

Decay anti-D_2*0
1  D*- pi+ HELCOV one zero;
Enddecay

Decay D_2*+
1  D*0 pi+ HELCOV one zero;
Enddecay

End    
    """

    with write_temp_file(s) as f:
        top, inner, final = load_decfile_particle(f)

    assert top == {Particle("vpho")}
    assert final == {Particle(i) for i in ["D*0", "D*-", "pi+"]}
    assert inner == {Particle(i) for i in ["D_2*+", "anti-D_2*0"]}

    inner = list(inner)
    assert inner[0].decay[0].params == ["one", "zero"]



