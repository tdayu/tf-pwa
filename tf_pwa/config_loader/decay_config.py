import copy
import functools
import random

import yaml

from tf_pwa.amp import (
    DecayChain,
    DecayGroup,
    HelicityDecay,
    get_decay,
    get_decay_chain,
    get_particle,
    split_particle_type,
)

from .base_config import BaseConfig


def set_min_max(dic, name, name_min, name_max):
    if name not in dic and name_min in dic and name_max in dic:
        dic[name] = (
            random.random() * (dic[name_max] - dic[name_min]) + dic[name_min]
        )


def decay_chain_cut_ls(decay):
    for i in decay:
        if isinstance(i, HelicityDecay):
            if len(i.get_ls_list()) == 0:
                return False, f"{i} ls not aviable {i.get_ls_list()}"
    return True, ""


def decay_chain_cut_mass(decay):
    for i in decay:
        if isinstance(i, HelicityDecay):
            if i.core.mass is None or any([j.mass is None for j in i.outs]):
                continue
            # print(i, i.core.mass, [j.mass for j in i.outs])
            if i.core.mass < sum([j.mass for j in i.outs]):
                return (
                    False,
                    f"{i} mass break {i.core.mass} < {[j.mass for j in i.outs]}",
                )
    return True, ""


class DecayConfig(BaseConfig):
    decay_chain_cut_list = {
        "ls_cut": decay_chain_cut_ls,
        "mass_cut": decay_chain_cut_mass,
    }

    def __init__(self, dic, share_dict={}):
        self.config = copy.deepcopy(dic)
        self.decay_chain_config = dic.get("decay_chain", {})
        self.share_dict = share_dict
        self.particle_key_map = {
            "Par": "P",
            "m0": "mass",
            "g0": "width",
            "J": "J",
            "P": "P",
            "spins": "spins",
            "bw": "model",
            "model": "model",
            "bw_l": "bw_l",
            "running_width": "running_width",
        }
        self.cut_list = self.config["data"].get("decay_chain_cut", ["ls_cut"])
        self.decay_key_map = {"model": "model"}
        self.dec = self.decay_item(self.config["decay"])
        (
            self.particle_map,
            self.particle_property,
            self.top,
            self.finals,
        ) = self.particle_item(self.config["particle"], share_dict)
        self.full_decay = DecayGroup(
            self.get_decay_struct(
                self.dec,
                self.particle_map,
                self.particle_property,
                self.top,
                self.finals,
                self.decay_chain_config,
            )
        )
        if self.config["data"].get("cp_trans", True):
            self.disable_allow_cc(self.full_decay)
        self.decay_struct = DecayGroup(
            self.get_decay_struct(self.dec, process_cut=False)
        )
        identical_particles = self.config["data"].get(
            "identical_particles", None
        )
        if identical_particles is not None:
            self.decay_struct.identical_particles = identical_particles
            self.full_decay.identical_particles = identical_particles

    @staticmethod
    def load_config(file_name, share_dict={}):
        if isinstance(file_name, dict):
            return copy.deepcopy(file_name)
        if isinstance(file_name, str):
            if file_name in share_dict:
                return DecayConfig.load_config(share_dict[file_name])
            with open(file_name) as f:
                ret = yaml.safe_load(f)
                if ret is None:
                    ret = {}
            return ret
        raise TypeError("not support config {}".format(type(file_name)))

    def get_decay(self, full=True):
        if full:
            return self.full_decay
        else:
            return self.decay_struct

    @staticmethod
    def _list2decay(core, outs):
        parts = []
        params = {}
        for j in outs:
            if isinstance(j, dict):
                for k, v in j.items():
                    params[k] = v
            else:
                parts.append(j)
        dec = {"core": core, "outs": parts, "params": params}
        return dec

    @staticmethod
    def decay_item(decay_dict):
        decs = []
        for core, outs in decay_dict.items():
            is_list = [isinstance(i, list) for i in outs]
            if all(is_list):
                for i in outs:
                    dec = DecayConfig._list2decay(core, i)
                    decs.append(dec)
            else:
                dec = DecayConfig._list2decay(core, outs)
                decs.append(dec)
        return decs

    @staticmethod
    def _do_include_dict(d, o, share_dict={}):
        s = DecayConfig.load_config(o, share_dict)
        for i in s:
            if i in d:
                if isinstance(d[i], dict):
                    s[i].update(d[i])
                    d[i] = s[i]
            else:
                d[i] = s[i]

    @staticmethod
    def particle_item_list(particle_list):
        particle_map = {}
        particle_property = {}
        for particle, candidate in particle_list.items():
            if isinstance(candidate, list):  # particle map
                if len(candidate) == 0:
                    particle_map[particle] = []
                for i in candidate:
                    if isinstance(i, str):
                        particle_map[particle] = particle_map.get(
                            particle, []
                        ) + [i]
                    elif isinstance(i, dict):
                        map_i, pro_i = DecayConfig.particle_item_list(i)
                        for k, v in map_i.items():
                            particle_map[k] = particle_map.get(k, []) + v
                        particle_property.update(pro_i)
                    else:
                        raise ValueError(
                            "value of particle map {} is {}".format(i, type(i))
                        )
            elif isinstance(candidate, dict):
                particle_property[particle] = candidate
            else:
                raise ValueError(
                    "value of particle {} is {}".format(
                        particle, type(candidate)
                    )
                )
        return particle_map, particle_property

    @staticmethod
    def particle_item(particle_list, share_dict={}):
        top = particle_list.pop("$top", None)
        finals = particle_list.pop("$finals", None)
        includes = particle_list.pop("$include", None)
        if includes:
            if isinstance(includes, list):
                for i in includes:
                    DecayConfig._do_include_dict(
                        particle_list, i, share_dict=share_dict
                    )
            elif isinstance(includes, str):
                DecayConfig._do_include_dict(
                    particle_list, includes, share_dict=share_dict
                )
            else:
                raise ValueError(
                    "$include must be string or list of string not {}".format(
                        type(includes)
                    )
                )
        particle_map, particle_property = DecayConfig.particle_item_list(
            particle_list
        )

        if isinstance(top, dict):
            particle_property.update(top)
        if isinstance(finals, dict):
            particle_property.update(finals)
        return particle_map, particle_property, top, finals

    def rename_params(self, params, is_particle=True):
        ret = {}
        if is_particle:
            key_map = self.particle_key_map
        else:
            key_map = self.decay_key_map
        for k, v in params.items():
            ret[key_map.get(k, k)] = v
        return ret

    def decay_chain_cut(self, decays):
        ret = []
        for i in decays:
            flag = True
            for name in self.cut_list:
                f = DecayConfig.decay_chain_cut_list[name]
                new_flag, msg = f(i)
                flag = flag and new_flag
                if not flag:
                    print(
                        "remove decay chain",
                        i,
                        "by",
                        name,
                        "\n\tbecause of",
                        msg,
                    )
                    break
            if flag:
                ret.append(i)
        return ret

    def get_decay_struct(
        self,
        decay,
        particle_map=None,
        particle_params=None,
        top=None,
        finals=None,
        chain_params={},
        process_cut=True,
    ):
        """get decay structure for decay dict"""
        particle_map = particle_map if particle_map is not None else {}
        particle_params = (
            particle_params if particle_params is not None else {}
        )

        particle_set = {}

        def add_particle(name):
            if name in particle_set:
                return particle_set[name]
            params = particle_params.get(name, {})
            params = self.rename_params(params)
            set_min_max(params, "mass", "m_min", "m_max")
            set_min_max(params, "width", "g_min", "g_max")
            part = get_particle(name, **params)
            particle_set[name] = part
            return part

        def wrap_particle(name):
            name_list = particle_map.get(name, [name])
            return [add_particle(i) for i in name_list]

        def all_combine(out):
            if len(out) < 1:
                yield []
            else:
                for i in out[0]:
                    for j in all_combine(out[1:]):
                        yield [i] + j

        decs = []
        for dec in decay:
            core = wrap_particle(dec["core"])
            outs = [wrap_particle(j) for j in dec["outs"]]
            for i in core:
                for j in all_combine(outs):
                    dec_i = get_decay(i, j, **dec["params"])
                    decs.append(dec_i)

        top_tmp, finals_tmp = set(), set()
        if top is None or finals is None:
            top_tmp, res, finals_tmp = split_particle_type(decs)
        if top is None:
            top_tmp = list(top_tmp)
            ## Needs better error message, possibly print the different decay chains, grouped by top particle?
            assert len(top_tmp) == 1, "not only one top particle"
            top = list(top_tmp)[0]
        else:
            if isinstance(top, str):
                top = particle_set[top]
            elif isinstance(top, dict):
                keys = list(top.keys())
                assert len(keys) == 1
                top = particle_set[keys.pop()]
            else:
                return particle_set[str(top)]
        if finals is None:
            finals = list(finals_tmp)
        elif isinstance(finals, (list, dict)):
            finals = [particle_set[i] for i in finals]
        else:
            raise TypeError("{}: {}".format(finals, type(finals)))

        dec_chain = top.chain_decay()
        ret = []
        for i in dec_chain:
            if sorted(DecayChain(i).outs) == sorted(finals):
                all_params = chain_params.get("$all", {})
                dec_c = get_decay_chain(i, **all_params)
                ret.append(dec_c)
        if process_cut:
            return self.decay_chain_cut(ret)
        return ret

    def disable_allow_cc(self, decay_group):
        for decay_chain in decay_group:
            for decay in decay_chain:
                if hasattr(decay, "allow_cc"):
                    decay.allow_cc = False
