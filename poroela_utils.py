import numpy as nm
from sfepy.base.base import Struct, get_default, output
from sfepy.discrete.fem.periodic import match_grid_plane
import sfepy.homogenization.coefs_base as cb

periodic_cache = {}


def match_plane(coor1, coor2, d, di):
    key = '%s_%d' % (d, coor1.shape[0])
    if key not in periodic_cache:
        periodic_cache[key] = match_grid_plane(coor1, coor2, di)

    return periodic_cache[key]


def match_x_plane(coor1, coor2):
    return match_plane(coor1, coor2, 'x', 0)


def match_y_plane(coor1, coor2):
    return match_plane(coor1, coor2, 'y', 1)


def match_z_plane(coor1, coor2):
    return match_plane(coor1, coor2, 'z', 2)


def build_op_pi(val, ir, ic):
    pi = nm.zeros_like(val)
    pi[:, ir] = val[:, ic]
    pi.shape = (pi.shape[0] * pi.shape[1],)

    return pi


def create_pis(val, vname='u'):
    dim = val.shape[1]
    pis = nm.zeros((dim, dim), dtype=nm.object)
    names = []
    for ir in range(dim):
        for ic in range(dim):
            pi = build_op_pi(val, ir, ic)
            pis[ir, ic] = {vname: pi}
            names.append('_%d%d' % (ir, ic))
    return names, pis


def print_safd(key, c0, c1, c2, dt):
    skey = 's' + key

    if key == 'Y':
        c0.update({skey: c0.get('sVol_Y')})
        c1.update({key: c1.get('Vol_Y')['volume_Y']})
        c2.update({key: c2.get('Vol_Y')['volume_Y']})

    fd = (c1.get(key) - c2.get(key)) / dt
    sa = c0.get(skey)
    output('  sa:')
    print(sa)
    output('  fd:')
    print(fd)
    output('  err:')
    print((sa - fd) / nm.linalg.norm(sa))


def get_periodic_bc(var_tab, dim=3, dim_tab=None, regions=None):
    if dim_tab is None:
        dim_tab = {'x': ['left', 'right'],
                   'z': ['bottom', 'top'],
                   'y': ['near', 'far']}

    periodic = {}
    epbcs = {}

    for ivar, reg in var_tab:
        periodic['per_%s' % ivar] = pers = []
        for idim in 'xyz'[0:dim]:
            key = 'per_%s_%s' % (ivar, idim)
            regs = ['%s_%s' % (reg, ii) for ii in dim_tab[idim]]
            if regions is not None:
                if regs[0] not in regions or regs[1] not in regions:
                    regs = []

            if len(regs) > 0:
                epbcs[key] = (regs, {'%s.all' % ivar: '%s.all' % ivar},
                              'match_%s_plane' % idim)
                pers.append(key)

    return epbcs, periodic


def data_to_struct(data):
    out = {}
    for k, v in data.items():
        out[k] = Struct(name='output_data',
                        mode='cell' if v[2] == 'c' else 'vertex',
                        data=v[0],
                        var_name=v[1],
                        dofs=None)

    return out


def coefs2qp(coefs, nqp, ret_others=False):
    out = {}
    others = {}

    for k, v in coefs.items():
        if isinstance(v, float):
            out[k] = nm.broadcast_to(v, (nqp, 1, 1))
            out[k] = nm.array(v, dtype=nm.float64).reshape((1, 1, 1))
        elif type(v) == nm.ndarray:
            aux = nm.atleast_2d(v)
            if aux.shape[-1] > aux.shape[-2]:
                aux = aux.T
            # out[k] = nm.broadcast_to(aux, (nqp,) + aux.shape)
            out[k] = aux.reshape((1,) + aux.shape)
        else:
            others[k] = v

    if ret_others:
        return out, others
    else:
        return out


def get_periodic_regions(reg, label=None, mesh_data=None, eps=1e-9):
    if label is None:
        label = reg

    dim_tab = [['Left', 'Right'], ['Near', 'Far'], ['Bottom', 'Top']]

    if mesh_data is not None:
        mat_id, mesh = mesh_data
        idxs = mesh.cmesh.cell_groups == mat_id
        coors = mesh.coors[mesh.get_conn(mesh.descs[0])[idxs].ravel()]
        bbox = mesh.get_bounding_box()
        cmin, cmax = nm.min(coors, axis=0), nm.max(coors, axis=0)
        cmin, cmax = abs(cmin - bbox[0, :]), abs(cmax - bbox[1, :])
        bbox_list = []
        for dim in range(3):
            if cmin[dim] < eps and cmax[dim] < eps:
                bbox_list += dim_tab[dim]
    else:
        bbox_list = dim_tab[0] + dim_tab[1] + dim_tab[2]

    out = {}
    for k in bbox_list:
        out[f'{label}_{k.lower()}'] = (f'r.{reg} *s r.{k}', 'facet')

    return out
