import sys
import os.path as osp
import numpy as nm
from copy import deepcopy
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.homogenization.utils import coor_to_sym, define_box_regions
from sfepy.homogenization.recovery import compute_p_from_macro
import sfepy.homogenization.coefs_base as cb
from sfepy.discrete.fem.mesh import Mesh
from poroela_utils import match_x_plane, match_y_plane, match_z_plane,\
    data_to_struct, get_periodic_bc, get_periodic_regions

wdir = osp.dirname(__file__)

# Y. Koutsawa et al. / Mechanics Research Communications 37 (2010) 489-494
D_piezo = nm.array([[6.0, 3.72, 3.83, 0, 0, 0],
                    [3.72, 6.0, 3.83, 0, 0, 0],
                    [3.83, 3.83, 20.3, 0, 0, 0],
                    [0, 0, 0, 1.23, 0, 0],
                    [0, 0, 0, 0, 1.23, 0],
                    [0, 0, 0, 0, 0, 1.23]]) * 1e9
g_piezo = nm.array([[0, 0, 0, 0, 0.01, 0],
                    [0, 0, 0, 0, 0, 0.01],
                    [-0.09, -0.09, 5.91, 0, 0, 0]])
d_piezo = nm.array([[18, 0, 0],
                    [0, 18, 0],
                    [0, 0, 255.3]]) * 8.854*1e-12

D_elast = stiffness_from_youngpoisson(3, 0.02e9, 0.49)
D_cond = stiffness_from_youngpoisson(3, 200e9, 0.25)  # Cu?


def recovery_micro_dc(pb, corrs, macro):
    eps0 = macro['eps0']
    mesh = pb.domain.mesh
    regions = pb.domain.regions
    dim = mesh.dim
    Ys_map = regions['Ys'].get_entities(0)
    Yp_map = regions['Yp'].get_entities(0)
    # deformation
    gl = '_' + corrs.keys()[0].split('_')[-1]
    u1 = -corrs['corrs_p' + gl]['u'] * macro['press']
    phi1 = -corrs['corrs_p' + gl]['r'] * macro['press']

    for ii in range(dim):
        u1 += corrs['corrs_k' + gl]['u_%d' % ii]\
            * nm.expand_dims(macro['efield'][Ys_map, ii], axis=1)
        phi1 += corrs['corrs_k' + gl]['r_%d' % ii]\
            * nm.expand_dims(macro['efield'][Yp_map, ii], axis=1)

    for ii in range(dim):
        for jj in range(dim):
            kk = coor_to_sym(ii, jj, dim)
            u1 += corrs['corrs_rs' + gl]['u_%d%d' % (ii, jj)]\
                * nm.expand_dims(macro['strain'][Ys_map, kk], axis=1)
            phi1 += corrs['corrs_rs' + gl]['r_%d%d' % (ii, jj)]\
                * nm.expand_dims(macro['strain'][Yp_map, kk], axis=1)

    u = macro['u'][Ys_map, :] + eps0 * u1
    phi = macro['phi'][Yp_map, :] + eps0 * phi1

    mvar = pb.create_variables(['u', 'r', 'svar'])
    e_mac_Ys = [None] * macro['strain'].shape[1]
    for ii in range(dim):
        for jj in range(dim):
            kk = coor_to_sym(ii, jj, dim)
            mvar['svar'].set_data(macro['strain'][:, kk])
            mac_e_Ys = pb.evaluate('ev_volume_integrate.i2.Ys(svar)',
                                   mode='el_avg',
                                   var_dict={'svar': mvar['svar']})
            e_mac_Ys[kk] = mac_e_Ys.squeeze()

    e_mac_Ys = nm.vstack(e_mac_Ys).T[:, nm.newaxis, :, nm.newaxis]

    mvar['r'].set_data(phi)
    E_mic = pb.evaluate('ev_grad.i2.Yp(r)',
                        mode='el_avg',
                        var_dict={'r': mvar['r']}) / eps0

    mvar['u'].set_data(u1)
    e_mic = pb.evaluate('ev_cauchy_strain.i2.Ys(u)',
                        mode='el_avg',
                        var_dict={'u': mvar['u']})
    e_mic += e_mac_Ys

    out = {
        'u0': (macro['u'][Ys_map, :], 'u', 'p'),
        'u': (u, 'u', 'p'),
        'u1': (u1, 'u', 'p'),
        'e_mic': (e_mic, 'u', 'c'),
        'phi': (phi, 'r', 'p'),
        'E_mic': (E_mic, 'r', 'c'),
    }

    return data_to_struct(out)


def recovery_micro_cc(pb, corrs, macro):
    eps0 = macro['eps0']
    mesh = pb.domain.mesh
    regions = pb.domain.regions
    dim = mesh.dim
    Ys_map = regions['Ys'].get_entities(0)
    Yp_map = regions['Yp'].get_entities(0)
    # deformation
    gl = '_' + corrs.keys()[0].split('_')[-1]
    u1 = -corrs['corrs_p' + gl]['u'] * macro['press'][Ys_map, :]
    phi = -corrs['corrs_p' + gl]['r'] * macro['press'][Yp_map, :]

    ncond = macro['phi'].shape[-1]
    for ii in range(ncond):
        u1 += corrs['corrs_k%d' % ii + gl]['u'] * macro['phi'][ii]
        phi += corrs['corrs_k%d' % ii + gl]['r'] * macro['phi'][ii]

    for ii in range(dim):
        for jj in range(dim):
            kk = coor_to_sym(ii, jj, dim)
            phi += corrs['corrs_rs' + gl]['r_%d%d' % (ii, jj)]\
                * nm.expand_dims(macro['strain'][Yp_map, kk], axis=1)
            u1 += corrs['corrs_rs' + gl]['u_%d%d' % (ii, jj)]\
                * nm.expand_dims(macro['strain'][Ys_map, kk], axis=1)

    u = macro['u'][Ys_map, :] + eps0 * u1

    mvar = pb.create_variables(['u', 'r', 'svar'])

    e_mac_Ys = [None] * macro['strain'].shape[1]

    for ii in range(dim):
        for jj in range(dim):
            kk = coor_to_sym(ii, jj, dim)
            mvar['svar'].set_data(macro['strain'][:, kk])
            mac_e_Ys = pb.evaluate('ev_volume_integrate.i2.Ys(svar)',
                                   mode='el_avg',
                                   var_dict={'svar': mvar['svar']})

            e_mac_Ys[kk] = mac_e_Ys.squeeze()

    e_mac_Ys = nm.vstack(e_mac_Ys).T[:, nm.newaxis, :, nm.newaxis]

    mvar['r'].set_data(phi)
    E_mic = pb.evaluate('ev_grad.i2.Yp(r)',
                        mode='el_avg',
                        var_dict={'r': mvar['r']}) / eps0

    mvar['u'].set_data(u1)
    e_mic = pb.evaluate('ev_cauchy_strain.i2.Ys(u)',
                        mode='el_avg',
                        var_dict={'u': mvar['u']})
    e_mic += e_mac_Ys

    # Stokes in Yf
    press_mac_grad = macro['pressg']

    mvar = pb.create_variables(['p', 'w'])
    nvd = mvar['w'].field.n_vertex_dof

    nnod = corrs['corrs_psi']['p_0'].shape[0]  # only vertex dofs
    press_mic = nm.zeros((nnod, 1), dtype=nm.float64)
    dvel = nm.zeros((nnod, dim), dtype=nm.float64)

    Yf_map = mvar['w'].field.vertex_remap_i

    for key, val in corrs['corrs_psi'].items():
        if key[:2] == 'p_':
            kk = int(key[-1])
            press_mic += val * press_mac_grad[Yf_map, kk][:, None]
        elif key[:2] == 'w_':
            kk = int(key[-1])
            dvel -= val[:nvd, :] * press_mac_grad[Yf_map, kk][:, None]

    centre_Y = nm.sum(pb.domain.mesh.coors, axis=0) / pb.domain.mesh.n_nod
    nodes_Yf = pb.domain.regions['Yf'].vertices

    press_mic += \
        compute_p_from_macro(press_mac_grad[nm.newaxis, nm.newaxis, :, :],
                             pb.domain.mesh.coors[nodes_Yf], 0,
                             centre=centre_Y, extdim=-1).reshape((nnod, 1))

    out = {
        'u0': (macro['u'][Ys_map, :], 'u', 'p'),
        'u': (u, 'u', 'p'),
        'u1': (u1, 'u', 'p'),
        'e_mic': (e_mic, 'u', 'c'),
        'phi': (phi, 'r', 'p'),
        'E_mic': (E_mic, 'r', 'c'),
        'w': (dvel, 'w', 'p'),
        'p_mic': (press_mic, 'p', 'p'),
    }

    return data_to_struct(out)


def define(eps0=1e-3,
           filename_mesh=osp.join(wdir, 'meshes', 'micro_piezoela_cfcc2.vtk'),
           fluid_mode='C', cond_mode='C', mat_mode='elastic_part',
           flag='', filename_coefs=None):

    if filename_coefs is None:
        filename_coefs = f'coefs_poropiezo{flag}'

    filename_mesh = osp.join(wdir, filename_mesh)
    mesh = Mesh.from_file(filename_mesh)
    dim = mesh.dim
    n_mat_woc = 2 if mat_mode is None else 3
    n_conduct = len(nm.unique(mesh.cmesh.cell_groups)) - n_mat_woc

    mode_dict = {'C': 'connected', 'D': 'disconnected'}
    print('micromode:')
    print('  fluid - %s' % mode_dict[fluid_mode])
    print('  conductors - %s' % mode_dict[cond_mode])
    print('  num. of conductors - %d' % n_conduct)
    print('  elastic part - %s' % (mat_mode == 'elastic_part'))

    sym_eye = 'nm.array([1, 1, 0])' if dim == 2\
        else 'nm.array([1, 1, 1, 0, 0, 0])'

    bbox = mesh.get_bounding_box()
    regions = define_box_regions(dim, bbox[0], bbox[1], eps=1e-3)

    regions.update({
        'Y': 'all',
        'Gamma_sf': ('r.Ys *s r.Yf', 'facet', 'Ys'),
        # channel / inclusion
        'Yf0': ('r.Yf -s r.Gamma_sf', 'facet'),
    })

    regions.update(get_periodic_regions('Ys'))

    if mat_mode is None:
        # parts: piezo, fluid
        regions.update({
            'Ym': 'cells of group 1',
            'Yf': 'cells of group 2',
            'Yp': ('copy r.Ym', 'cell'),
            'Ye': ('copy r.Ym', 'cell'),
        })
        regions.update(get_periodic_regions('Yp'))
    else:
        # parts: piezo, elastic, fluid
        regions.update({
            'Ye': 'cells of group 1',
            'Yp': 'cells of group 2',
            'Yf': 'cells of group 3',
            'Ym': ('r.Ye +v r.Yp', 'cell'),
        })
        regions.update(get_periodic_regions('Yp'))

    if n_conduct > 0:
        regions.update({
            'Yc': (' +v '.join(['r.Yc%d' % k for k in range(n_conduct)]),
                   'cell'),
            'Ys': ('r.Ym +v r.Yc', 'cell'),
            'Gamma_mc': ('r.Ym *s r.Yc', 'facet', 'Ym'),
        })
        for k in range(n_conduct):
            sk = '%d' % k
            regions.update({
                'Yc' + sk: 'cells of group %d' % (n_mat_woc + 1 + k),
                'Gamma_c' + sk: ('r.Ym *s r.Yc' + sk, 'facet', 'Ym'),
            })

    else:
        regions.update({'Ys': ('copy r.Ym', 'cell')})
        cond_mode == 'N'

    if fluid_mode == 'C':
        mat_id = 2 if mat_mode is None else 3
        regions.update(get_periodic_regions('Yf0', label='Yf',
                                            mesh_data=(mat_id, mesh)))

    options = {
        'coefs_filename': filename_coefs,
        'volume': {
            'variables': ['svar'],
            'expression': 'd_volume.i2.Y(svar)',
        },
        'coefs': 'coefs',
        'requirements': 'requirements',
        'output_dir': 'output',
        'ls': 'ls',
        'file_per_var': True,
        'absolute_mesh_path': True,
        # 'multiprocessing': False,
        'multiprocessing': True,
    }

    fields = {
        'displacement': ('real', 'vector', 'Ys', 1),
        'potential': ('real', 'scalar', 'Yp', 1),
        'sfield': ('real', 'scalar', 'Y', 1),
        'velocity': ('real', 'vector', 'Yf', 2),
        'pressure': ('real', 'scalar', 'Yf', 1),
    }

    variables = {
        # displacement
        'u': ('unknown field', 'displacement'),
        'v': ('test field', 'displacement', 'u'),
        'Pi_u': ('parameter field', 'displacement', 'u'),
        'U1': ('parameter field', 'displacement', '(set-to-None)'),
        'U2': ('parameter field', 'displacement', '(set-to-None)'),
        # electric potential
        'r': ('unknown field', 'potential'),
        's': ('test field', 'potential', 'r'),
        'Pi_r': ('parameter field', 'potential', 'r'),
        'R1': ('parameter field', 'potential', '(set-to-None)'),
        'R2': ('parameter field', 'potential', '(set-to-None)'),
        'svar': ('parameter field', 'sfield', '(set-to-None)'),
        # fluid pressure
        'p': ('unknown field', 'pressure'),
        'q': ('test field', 'pressure', 'p'),
        'P1': ('parameter field', 'pressure', '(set-to-None)'),
        'P2': ('parameter field', 'pressure', '(set-to-None)'),
        'ls': ('unknown field', 'pressure'),
        'lv': ('test field', 'pressure', 'ls'),
        # fluid velocity
        'w': ('unknown field', 'velocity'),
        'z': ('test field', 'velocity', 'w'),
        'Pi_w': ('parameter field', 'velocity', 'w'),
        'W1': ('parameter field', 'velocity', '(set-to-None)'),
        'W2': ('parameter field', 'velocity', '(set-to-None)'),
    }

    if 'piezo_micro_6x6x6_' in filename_mesh:
        del(regions['Yp_bottom'], regions['Yp_top'])

    epbcs, periodic = get_periodic_bc([('u', 'Ys'), ('r', 'Yp')],
                                      regions=regions)

    mat_g_sc, mat_d_sc = (eps0, eps0**2) if cond_mode == 'C' else (1., 1.)  # ?
    print(mat_g_sc, mat_d_sc)

    materials = {
        'matrix': ({
            'D': {'Yp': D_piezo},
        },),
        'piezo': ({
            'g': g_piezo / mat_g_sc,
            'd': d_piezo / mat_d_sc,
        },),
        'fluid': ({  # water
            'one': 1.0,
            'gamma': 1.0 / 2.15e9,
            'bar_eta': 8.9e-4 / eps0**2,  # dynamic viscosity
            'D': stiffness_from_youngpoisson(dim, 1.0, 0.3),
        },),
    }

    if mat_mode is not None:
        materials['matrix'][0]['D'].update({'Ye': D_elast})

    functions = {
        'match_x_plane': (match_x_plane,),
        'match_y_plane': (match_y_plane,),
        'match_z_plane': (match_z_plane,),
    }

    ebcs = {
        'fixed_u': ('Corners', {'u.all': 0.0}),
    }

    if cond_mode == 'C':
        ebcs.update({'fixed_r': ('Gamma_mc', {'r.all': 0.0})})
        fixed_r = ['fixed_r']
    else:
        fixed_r = []  # ???!!!

    integrals = {
        'i2': 2,
        'i3': 3,
    }

    solvers = {
        'ls': ('ls.mumps', {}),
        'ns_em6': ('nls.newton', {
                   'i_max': 1,
                   'eps_a': 1e-6,
                   'eps_r': 1e-3,
                   'problem': 'nonlinear'}),
        'ns_em12': ('nls.newton', {
                   'i_max': 1,
                   'eps_a': 1e-12,
                   'eps_r': 1e-3,
                   'problem': 'nonlinear'}),
        'ns_em1': ('nls.newton', {
                   'i_max': 1,
                   'eps_a': 1e-1,
                   'eps_r': 1e-3,
                   'problem': 'nonlinear'}),
    }

    coefs = {
        'A': {
            'requires': ['pis_u', 'corrs_rs'],
            'expression': '   dw_lin_elastic.i2.Ys(matrix.D, U1, U2)'
                          ' + dw_diffusion.i2.Yp(piezo.d, R1, R2)',
            'set_variables': [[('U1', ('corrs_rs', 'pis_u'), 'u'),
                               ('R1', 'corrs_rs', 'r')],
                              [('U2', ('corrs_rs', 'pis_u'), 'u'),
                               ('R2', 'corrs_rs', 'r')]],
            'class': cb.CoefSymSym,
        },
        'B1': {
            'status': 'auxiliary',
            'requires': ['pis_u', 'corrs_p'],
            'expression': '   dw_lin_elastic.i2.Ys(matrix.D, U1, U2)'
                          ' - dw_piezo_coupling.i2.Yp(piezo.g, U1, R2)',
            'set_variables': [('U1', 'pis_u', 'u'),
                              ('U2', 'corrs_p', 'u'),
                              ('R2', 'corrs_p', 'r')],
            'class': cb.CoefSym,
        },
        'B': {
            'requires': ['c.Phi', 'c.B1'],
            'expression': 'c.B1 + c.Phi * %s' % sym_eye,
            'class': cb.CoefEval,
        },
        'N': {
            'status': 'auxiliary',
            'requires': ['corrs_p'],
            'expression': 'dw_surface_ltr.i2.Gamma_sf(U1)',
            'set_variables': [('U1', 'corrs_p', 'u')],
            'class': cb.CoefOne,
        },
        'M': {
            'requires': ['c.Phi', 'c.N'],
            'expression': 'c.N + c.Phi * %e' % materials['fluid'][0]['gamma'],
            'class': cb.CoefEval,
        },
        'Phi': {
            'requires': ['c.vol'],
            'expression': 'c.vol["fraction_Yf"]',
            'class': cb.CoefEval,
        },
        'vol': {
            'regions': ['Ym', 'Yf'] + ['Yc%d' % k for k in range(n_conduct)],
            'expression': 'd_volume.i2.%s(svar)',
            'class': cb.VolumeFractions,
        },
        'eps0': {
            'requires': [],
            'expression': '%e' % eps0,
            'class': cb.CoefEval,
        },
        'bar_eta': {
            'expression': '%e' % materials['fluid'][0]['bar_eta'],
            'class': cb.CoefEval,
        },
        'filenames': {},
    }

    requirements = {
        'pis_u': {
            'variables': ['u'],
            'class': cb.ShapeDimDim,
        },
        'pis_r': {
            'variables': ['r'],
            'class': cb.ShapeDim,
        },
        'corrs_rs': {
            'requires': ['pis_u'],
            'ebcs': ['fixed_u'] + fixed_r,
            'epbcs': periodic['per_u'] + periodic['per_r'],
            'is_linear': True,
            'equations': {
                'eq1':
                    """dw_lin_elastic.i2.Ys(matrix.D, v, u)
                     - dw_piezo_coupling.i2.Yp(piezo.g, v, r)
                   = - dw_lin_elastic.i2.Ys(matrix.D, v, Pi_u)""",
                'eq2':
                    """
                     - dw_piezo_coupling.i2.Yp(piezo.g, u, s)
                     - dw_diffusion.i2.Yp(piezo.d, s, r)
                     = dw_piezo_coupling.i2.Yp(piezo.g, Pi_u, s)""",
            },
            'set_variables': [('Pi_u', 'pis_u', 'u')],
            'class': cb.CorrDimDim,
            'save_name': 'corrs_rs' + flag,
            'dump_variables': ['u', 'r'],
            'solvers': {'ls': 'ls', 'nls': 'ns_em4'},
        },
        'corrs_p': {
            'requires': [],
            'ebcs': ['fixed_u'] + fixed_r,
            'epbcs': periodic['per_u'] + periodic['per_r'],
            'is_linear': True,
            'equations': {
                'eq1':
                    """dw_lin_elastic.i2.Ys(matrix.D, v, u)
                     - dw_piezo_coupling.i2.Yp(piezo.g, v, r)
                     = dw_surface_ltr.i2.Gamma_sf(v)""",
                'eq2':
                    """
                     - dw_piezo_coupling.i2.Yp(piezo.g, u, s)
                     - dw_diffusion.i2.Yp(piezo.d, s, r)
                     = 0"""
            },
            'class': cb.CorrOne,
            'save_name': 'corrs_p' + flag,
            'dump_variables': ['u', 'r'],
            'solvers': {'ls': 'ls', 'nls': 'ns_em6'},
        },
    }

    if cond_mode == 'D':
        options.update({'recovery_hook': recovery_micro_dc})
        coefs.update({
            'Dx': {
                'requires': ['pis_r', 'corrs_k'],
                'expression': '   dw_diffusion.i2.Yp(piezo.d, R1, R2)'
                              ' + dw_lin_elastic.i2.Ys(matrix.D, U1, U2)',
                'set_variables': [[('R1', ('corrs_k', 'pis_r'), 'r'),
                                   ('U1', 'corrs_k', 'u')],
                                  [('R2', ('corrs_k', 'pis_r'), 'r'),
                                   ('U2', 'corrs_k', 'u')]],
                'class': cb.CoefDimDim,
            },
            'D1': {
                'status': 'auxiliary',
                'requires': ['pis_r', 'corrs_k'],
                'expression': 'dw_diffusion.i2.Yp(piezo.d, R1, R2)',
                'set_variables': [('R1', ('corrs_k', 'pis_r'), 'r'),
                                  ('R2', ('corrs_k', 'pis_r'), 'r')],
                'class': cb.CoefDimDim,
            },
            'D2': {
                'status': 'auxiliary',
                'requires': ['corrs_k'],
                'expression': 'dw_lin_elastic.i2.Ys(matrix.D, U1, U2)',
                'set_variables': [('U1', 'corrs_k', 'u'),
                                  ('U2', 'corrs_k', 'u')],
                'class': cb.CoefDimDim,
            },
            'D': {
                'requires': ['c.D1', 'c.D2'],
                'expression': 'c.D1 + c.D2',
                'class': cb.CoefEval,
            },
            'Gx': {
                'requires': ['pis_u', 'pis_r', 'corrs_k'],
                'expression': '   dw_piezo_coupling.i2.Yp(piezo.g, U2, R1)'
                              ' - dw_lin_elastic.i2.Ys(matrix.D, U1, U2)',
                'set_variables': [[('R1', ('corrs_k', 'pis_r'), 'r'),
                                   ('U1', 'corrs_k', 'u')],
                                  ('U1', 'pis_u', 'u')],
                'class': cb.CoefDimSym,
            },
            'G1': {
                'status': 'auxiliary',
                'requires': ['pis_u', 'pis_r', 'corrs_k'],
                'expression': 'dw_piezo_coupling.i2.Yp(piezo.g, U1, R1)',
                'set_variables': [('R1', ('corrs_k', 'pis_r'), 'r'),
                                  ('U1', 'pis_u', 'u')],
                'class': cb.CoefDimSym,
            },
            'G2': {
                'status': 'auxiliary',
                'requires': ['pis_u', 'corrs_k'],
                'expression': 'dw_lin_elastic.i2.Ys(matrix.D, U1, U2)',
                'set_variables': [('U1', 'corrs_k', 'u'),
                                  ('U2', 'pis_u', 'u')],
                'class': cb.CoefDimSym,
            },
            'G': {
                'requires': ['c.G1', 'c.G2'],
                'expression': 'c.G1 - c.G2',
                'class': cb.CoefEval,
            },
            'F': {
                'requires': ['corrs_k'],
                'expression': 'dw_surface_ltr.i2.Gamma_sf(U1)',
                'set_variables': [('U1', 'corrs_k', 'u')],
                'class': cb.CoefDim,
            },
        })
        requirements.update({
            'corrs_k': {
                'requires': ['pis_r'],
                'ebcs': ['fixed_u'],
                'epbcs': periodic['per_u'] + periodic['per_r'],
                'is_linear': True,
                'equations': {
                    'eq1':
                        """dw_lin_elastic.i2.Ys(matrix.D, v, u)
                         - dw_piezo_coupling.i2.Yp(piezo.g, v, r)
                         = dw_piezo_coupling.i2.Yp(piezo.g, v, Pi_r)""",
                    'eq2':
                        """
                         - dw_piezo_coupling.i2.Yp(piezo.g, u, s)
                         - dw_diffusion.i2.Yp(piezo.d, s, r)
                         = dw_diffusion.i2.Yp(piezo.d, s, Pi_r)"""
                },
                'set_variables': [('Pi_r', 'pis_r', 'r')],
                'class': cb.CorrDim,
                'save_name': 'corrs_k' + flag,
                'dump_variables': ['u', 'r'],
                'solvers': {'ls': 'ls', 'nls': 'ns_em6'},
            },
        })

    elif cond_mode == 'C':
        options.update({'recovery_hook': recovery_micro_cc})
        requirements.update({
            'corrs_rho': {
                'requires': [],
                'ebcs': ['fixed_u', 'fixed_r'],
                'epbcs': periodic['per_u'] + periodic['per_r'],
                'is_linear': True,
                'equations': {
                    'eq1':
                        """dw_lin_elastic.i2.Ys(matrix.D, v, u)
                         - dw_piezo_coupling.i2.Yp(piezo.g, v, r)
                         = 0""",
                    'eq2':
                        """
                         - dw_piezo_coupling.i2.Yp(piezo.g, u, s)
                         - dw_diffusion.i2.Yp(piezo.d, s, r)
                         =
                         - dw_surface_integrate.i2.Gamma_sf(s)"""
                    },
                'class': cb.CorrOne,
                'save_name': 'corrs_p' + flag,
                'dump_variables': ['u', 'r'],
                'solvers': {'ls': 'ls', 'nls': 'ns_em6'},
            },
        })

        # coefs S, R not implemented!

        for k in range(n_conduct):
            sk = '%d' % k

            materials['matrix'][0]['D'].update({'Yc' + sk: D_cond})

            ebcs.update({
                'fixed_r1_k_' + sk: ('Gamma_c' + sk, {'r.0': 1.0}),
                'fixed_r0_k_' + sk: ('Gamma_c' + sk, {'r.0': 0.0}),
            })

            fixed_r0_k = ['fixed_r0_k_%d' % ii for ii in range(n_conduct)
                          if not ii == k]

            requirements.update({
                'corrs_k' + sk: {
                    'requires': ['pis_r'],
                    'ebcs': ['fixed_u', 'fixed_r1_k_' + sk] + fixed_r0_k,
                    'epbcs': periodic['per_u'] + periodic['per_r'],
                    'is_linear': True,
                    'equations': {
                        'eq1':
                            """dw_lin_elastic.i2.Ys(matrix.D, v, u)
                            - dw_piezo_coupling.i2.Yp(piezo.g, v, r)
                            = 0""",
                        'eq2':
                            """
                            - dw_piezo_coupling.i2.Yp(piezo.g, u, s)
                            - dw_diffusion.i2.Yp(piezo.d, s, r)
                            = 0"""
                        },
                    'class': cb.CorrOne,
                    'save_name': 'corrs_k' + sk + flag,
                    'dump_variables': ['u', 'r'],
                    'solvers': {'ls': 'ls', 'nls': 'ns_em6'},
                },
            })

            coefs.update({
                'V' + sk: {
                    'requires': ['pis_u', 'corrs_k' + sk],
                    'expression': '   dw_lin_elastic.i2.Ys(matrix.D, U1, U2)'
                                  ' - dw_piezo_coupling.i2.Yp(piezo.g, U1, R2)',
                    'set_variables': [('U1', 'pis_u', 'u'),
                                      ('U2', 'corrs_k' + sk, 'u'),
                                      ('R2', 'corrs_k' + sk, 'r')],
                    'class': cb.CoefSym,
                },
                'Z' + sk: {
                    'requires': ['corrs_k' + sk],
                    'expression': 'dw_surface_ltr.i2.Gamma_sf(U1)',
                    'set_variables': [('U1', 'corrs_k' + sk, 'u')],
                    'class': cb.CoefOne,
                },
            })

    else:
        print('unknown mode for fluid: %s!' % fluid_mode)
        debug()

    if fluid_mode == 'C':
        ebcs.update({
            'fixed_w': ('Gamma_sf', {'w.all': 0.0}),
        })
        epbcs_w, periodic_w = get_periodic_bc([('w', 'Yf')], regions=regions)
        epbcs.update(epbcs_w)
        periodic.update(periodic_w)

        lcbcs = {
            'imv': ('Yf', {'ls.all': None}, None, 'integral_mean_value'),
        }

        coefs.update({
            'K': {
                'requires': ['corrs_psi'],
                'expression': 'dw_div_grad.i3.Yf(W1, W2)',
                'set_variables': [('W1', 'corrs_psi', 'w'),
                                  ('W2', 'corrs_psi', 'w')],
                'class': cb.CoefDimDim,
            },
        })

        requirements.update({
            'pis_w': {
                'variables': ['w'],
                'class': cb.OnesDim,
            },
            'corrs_psi': {
                'requires': ['pis_w'],
                'ebcs': ['fixed_w'],
                'epbcs': periodic['per_w'],
                'lcbcs': ['imv'],
                'is_linear': False,
                'equations': {
                    'balance_of_forces':
                        """dw_div_grad.i3.Yf(z, w)
                         - dw_stokes.i3.Yf(z, p)
                         =
                           dw_volume_dot.i3.Yf(z, Pi_w)""",
                    'incompressibility':
                        """
                         - dw_stokes.i3.Yf(w, q)
                         + dw_dot.i3.Yf(q, ls)
                         = 0""",
                    'imv': 'dw_dot.i3.Yf(lv, p) = 0',
                },
                'set_variables': [('Pi_w', 'pis_w', 'w')],
                'class': cb.CorrDim,
                'save_name': 'corrs_psi' + flag,
                'dump_variables': ['w', 'p'],
                'solvers': {'ls': 'ls', 'nls': 'ns_em12'},
            },
        })

    return locals()
