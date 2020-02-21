from opt_einsum import get_symbol, contract_path, contract
from .tensorflow_wrapper import tf

from pysnooper import snoop


class Einsum(object):
    def __init__(self, expr, shapes):
        self.expr = expr
        self.shapes = shapes

    def __call__(self, *args):
        return contract(self.expr, *args, backend="tensorflow")


def symbol_generate(base_map):
    if isinstance(base_map, dict):
        base_map = base_map.values()
    for i in range(100):
        symbol = get_symbol(i)
        if symbol not in base_map:
            yield symbol


def replace_ellipsis(expr, shapes):
    ret = expr
    idx = expr.split("->")[0].split(",")[0]
    if "..." in expr:
        extra_size = len(shapes[0]) - len(idx) + 3
        base_map = set(expr) - {".","-",">",","}
        base_map = dict(enumerate(base_map))
        ig = symbol_generate(base_map)
        extra = []
        for i in range(extra_size):
            extra.append(next(ig))
        ret = expr.replace("...", "".join(extra))
    return ret


def _get_order_bound_list(bd_dict, ord_dict, idx, left=0):
    if idx in ord_dict:
        return [ord_dict[idx]]
    assert idx in bd_dict, "not found"
    bd = bd_dict[idx]
    od = []
    for i in bd[left]:
        if i in ord_dict:
            od.append(ord_dict[i])
        else:
            od += _get_order_bound_list(bd_dict, ord_dict, i, left)
    return od


def ordered_indices(expr, shapes):
    """
    find a better order to reduce transpose.

    """
    ein_s = expr.split("->")
    final_index = ein_s[1]
    idx_input = ein_s[0].split(",")
    base_order = dict(zip(final_index, range(len(final_index))))
    max_i = len(expr)
    base_order["_min"] = -1
    base_order["_max"] = max_i

    combined_index = set("".join(idx_input)) - set(final_index)
    bound_dict = {}
    for i in combined_index:
        bound_dict[i] = ([], [])

    for i in combined_index:
        for j in idx_input:
            if i in j:
                pos = j.index(i)
                if pos-1 >= 0:
                    bound_dict[i][0].append(j[pos-1])
                else:
                    bound_dict[i][0].append("_min")
                if pos+1 < len(j):
                    bound_dict[i][1].append(j[pos+1])
                else:
                    bound_dict[i][1].append("_max")

    for i in bound_dict:
        left = max(_get_order_bound_list(bound_dict, base_order, i, 0))
        right = min(_get_order_bound_list(bound_dict, base_order, i, 1))
        if right > left:
            base_order[i] = left * 0.4 + right * 0.6
        else:
            base_order[i] = left + 0.01

    base_order = dict(sorted(base_order.items(), key=lambda x: x[0]))
    return base_order


def remove_size1(expr, *args):
    sub = expr.replace("->", "").replace(",", "")
    shapes = []
    for i in args:
        shapes += list(i.shape)

    size_map = dict(zip(sub, shapes))
    remove_idx = []
    for i in size_map:
        if size_map[i] == 1:
            remove_idx.append(i)

    for i in remove_idx:
        expr = expr.replace(i, "")

    ret = []
    for i in args:
        shape = list(i.shape)
        if 1 in shape:
            shape.remove(1)
        ret.append(tf.reshape(i, shape))

    return expr, ret, size_map


def einsum(expr, *args, **kwargs):
    shapes = [i.shape for i in args]
    expr = replace_ellipsis(expr, shapes)
    final_idx = expr.split("->")[1]
    expr2, args, size_map = remove_size1(expr, *args)
    final_shape = [size_map[i] for i in final_idx]
    base_order = ordered_indices(expr2, shapes)
    ein_s = expr2.split("->")
    final_index = ein_s[1]
    idxs = ein_s[0].split(",")

    path, path_info = contract_path(expr2, *args, optimize="auto")
    data = list(args)
    in_idx = list(idxs)
    for idx in path:
        part_data = [data[i] for i in idx]
        part_in_idx = [in_idx[i] for i in idx]
        for i in sorted(idx)[::-1]:
            del data[i]
            del in_idx[i]
        out_idx = set("".join(part_in_idx)) & set(final_index + "".join(in_idx))
        out_idx = "".join(sorted(out_idx, key=lambda x: base_order[x]))
        in_idx.append(out_idx)
        expr_i = "{}->{}".format(",".join(part_in_idx), out_idx)
        result = tensor_einsum_reduce_sum(expr_i, *part_data, order=base_order)
        data.append(result)
    return tf.reshape(data[0], final_shape)


def tensor_einsum_reduce_sum(expr, *args, order):
    """
    "abe,bcf->acef"  =reshape=> "ab1e1,1bc1f->acef" =product=> "abcef->acef" =reduce_sum=> "acef"
    """
    ein_s = expr.split("->")
    final_index = ein_s[1]
    idxs = ein_s[0].split(",")
    for i in idxs:
        if len(set(i)) != len(i):  # inner product
            return tf.einsum(expr, *args)

    require_order = sorted(set(ein_s[0]) - {","}, key=lambda x: order[x])

    # transpose
    t_args = []
    for i, j in zip(idxs, args):
        sorted_idx = sorted(i, key=lambda x: order[x])
        if list(i) == sorted_idx:
            t_args.append(j)
        else:
            trans = [i.index(k) for k in sorted_idx]
            t_args.append(tf.transpose(j, trans))

    # reshape
    sum_idx = set(require_order) - set(final_index)
    sum_idx_idx = []
    for i, j in enumerate(require_order):
        if j in sum_idx:
            sum_idx_idx.append(i)
    shapes = [i.shape for i in args]
    expand_shapes = []
    for idx, shape in zip(idxs, shapes):
        ex_shape = []
        shape_dict = dict(zip(idx, shape))
        for i in require_order:
            ex_shape.append(shape_dict.get(i, 1))
        expand_shapes.append(ex_shape)

    s_args = []
    for i, j in zip(expand_shapes, t_args):
        s_args.append(tf.reshape(j, i))

    # product
    ret_1 = s_args[0]
    for i in s_args[1:]:
        ret_1 = ret_1 * i

    # reduce_sum
    ret = tf.reduce_sum(ret_1, axis=sum_idx_idx)
    return ret