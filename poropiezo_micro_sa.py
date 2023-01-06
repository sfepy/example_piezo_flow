import os.path as osp
import numpy as nm
from copy import deepcopy
import poropiezo_micro
from sfepy.base.base import Struct, debug, get_default
import sfepy.homogenization.coefs_base as cb
from sfepy.homogenization.utils import coor_to_sym, iter_sym
from poroela_utils import create_pis, get_periodic_bc,\
    data_to_struct
import sfepy.base.multiproc as multiproc

wdir = osp.dirname(__file__)

mp_module, _ = multiproc.get_multiproc()
multiproc_dependecies = mp_module.get_dict('dependecies', clear=True)


def recovery_micro(pb, corrs, macro):
    eps0 = macro['eps0']
    mesh = pb.domain.mesh
    dim = mesh.dim

    regions = pb.domain.regions

    map_flag = macro['press'].shape[0] > 1
    Ys_map = regions['Ys'].get_entities(0)

    ncond = macro['phi'].shape[-1]
    gl = '_' + '_'.join(pb.conf.filename_coefs.split('_')[2:])

    mvar = pb.create_variables(['u', 'r', 'svar', 'p', 'w', 'uf'])

    if map_flag:
        Yp_map = regions['Yp'].get_entities(0)
        Yf_map = mvar['w'].field.vertex_remap_i
        press_mac_s = macro['press'][Ys_map, :, 0]
        press_mac_p = macro['press'][Yp_map, :, 0]
        press_mac_f = macro['press'][Yf_map, :, 0]
        phi_mac_s = macro['phi'][Ys_map, 0, :]
        phi_mac_p = macro['phi'][Yp_map, 0, :]
        strain_mac_s = macro['strain'][Ys_map, :, 0]
        strain_mac_p = macro['strain'][Yp_map, :, 0]
        displ_mac_s = macro['displ'][Ys_map, :, 0]
        press_mac_grad_f = macro['pressg'][Yf_map, 0, :]
    else:
        from sfepy.base.base import debug; debug()
        # press_mac_s = press_mac_p = press_mac_f = macro['press']
        # phi_mac_s = phi_mac_p = macro['phi']
        # strain_mac_s = strain_mac_p = macro['strain'].T
        # displ_mac_s = macro['displ'].T
        # press_mac_grad_f = macro['pressg'].T

    u1 = -corrs['corrs_p' + gl]['u'] * press_mac_s
    phi = -corrs['corrs_p' + gl]['r'] * press_mac_p
    for ii in range(ncond):
        u1 += corrs['corrs_k%d' % ii + gl]['u'] * phi_mac_s[..., [ii]]
        phi += corrs['corrs_k%d' % ii + gl]['r'] * phi_mac_p[..., [ii]]

    for ii in range(dim):
        for jj in range(dim):
            kk = coor_to_sym(ii, jj, dim)
            u1 += corrs['corrs_rs' + gl]['u_%d%d' % (ii, jj)]\
                * strain_mac_s[:, [kk]]
            phi += corrs['corrs_rs' + gl]['r_%d%d' % (ii, jj)]\
                * strain_mac_p[:, [kk]]

    u = displ_mac_s + eps0 * u1
    if map_flag:
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
    else:
        e_mac_Ys = macro['strain']

    mvar['r'].set_data(phi)
    E_mic = pb.evaluate('ev_grad.i2.Yp(r)', mode='el_avg',
                        var_dict={'r': mvar['r']}) / eps0

    mvar['u'].set_data(u1)
    e_mic = pb.evaluate('ev_cauchy_strain.i2.Ys(u)', mode='el_avg',
                        var_dict={'u': mvar['u']})

    e_mic += e_mac_Ys

    mvar['u'].set_data(u)
    uc = pb.evaluate('ev_integrate.i2.Ys(u)', mode='el_avg',
                        var_dict={'u': mvar['u']})

    # Stokes in Yf
    nvd = mvar['w'].field.n_vertex_dof
    nnod = corrs['corrs_psi' + gl]['p_0'].shape[0]  # only vertex dofs
    p1 = nm.zeros((nnod, 1), dtype=nm.float64)
    w = nm.zeros((nnod, dim), dtype=nm.float64)
    fluid = [v for v in pb.conf.materials.values() if v.name == 'fluid'][0]
    bar_eta = fluid.values['bar_eta']
    for key, val in corrs['corrs_psi' + gl].items():
        gp = press_mac_grad_f[:, [int(key[-1])]]
        if key[:2] == 'p_':
            p1 += -val * gp  # -/+ ???
        elif key[:2] == 'w_':
            w += -val[:nvd, :] * gp / bar_eta  # -/+ ???

    p = press_mac_f + eps0 * p1

    if map_flag:
        p_grad = press_mac_grad_f
    else:
        p_grad = p * 0 + press_mac_grad_f

    u_Ys = nm.zeros((mesh.n_nod, dim), dtype=nm.float64)
    u_Ys[Ys_map, :] = u.reshape(Ys_map.shape[0], dim)
    u_Yf = define_Yf_dvel(pb, u_Ys)

    mvar['uf'].set_data(w)
    grad_w = pb.evaluate('ev_grad.i2.Yf(uf)', mode='el_avg',
                         var_dict={'uf': mvar['uf']})
    wc = pb.evaluate('ev_integrate.i2.Yf(uf)', mode='el_avg',
                     var_dict={'uf': mvar['uf']})

    out = {
        'u': (u, 'u', 'p'),
        'uc': (uc, 'u', 'c'),
        'u1': (u1, 'u', 'p'),
        'e_mic': (e_mic, 'u', 'c'),
        'phi': (phi, 'r', 'p'),
        'E_mic': (E_mic, 'r', 'c'),
        'w': (w, 'w', 'p'),
        'grad_w': (grad_w.reshape((grad_w.shape[0], 1, 9, 1)), 'w', 'c'),
        'wc': (wc, 'w', 'c'),
        'uf_w': (u_Yf, 'w', 'p'),
        'p_grad': (p_grad, 'p', 'p'),
        'p': (p, 'p', 'p'),
        'p1': (p1, 'p', 'p'),
        'p_mac': (press_mac_f, 'p', 'p'),
        'pgrad_mac': (press_mac_grad_f, 'p', 'p'),
        'uf_p': (u_Yf, 'p', 'p'),
    }

    return data_to_struct(out)


def modify_mesh(val, spbox, dv_mode, cp_pos):
    cpoints0 = spbox.get_control_points().copy()
    for pts, dir in dv_mode:
        for pt in pts:
            spbox.move_control_point(cp_pos[pt], nm.array(dir) * val)

    new_coors = spbox.evaluate()
    spbox.set_control_points(cpoints0)

    return new_coors


def set_V_on_sf(ts, coor, problem=None, **kwargs):
    cmmap = problem.domain.regions['Gamma_sf'].vertices
    val = problem.dvelocity[cmmap]
    return val.flat


def define_Yf_dvel(pb, dvel):
    from sfepy.discrete import Problem

    dim = pb.domain.mesh.dim
    mx = nm.max(pb.domain.mesh.coors[pb.domain.regions['Yf'].vertices], axis=0)
    bbox = pb.domain.mesh.get_bounding_box()
    periodic = nm.abs(bbox[1] - mx) / (bbox[1] - bbox[0])
    epbcs, _ = get_periodic_bc([('uf', 'Yf')])
    for k in range(dim):
        if periodic[k] > 1e-3:
            del(epbcs['per_uf_' + 'xyz'[k]])

    conf = pb.conf.copy()
    conf.equations = {'eq': 'dw_lin_elastic.i2.Yf(fluid.D, vf, uf) = 0'}
    conf.edit('ebcs', {'fixed_uf': ('Gamma_sf', {'uf.all': 'set_V_on_sf'})})
    conf.edit('lcbcs', {})
    conf.edit('epbcs', epbcs)
    functions = {'set_V_on_sf': (set_V_on_sf,)}
    functions.update(conf._raw['functions'])
    conf.edit('functions', functions)
    aeps = nm.linalg.norm(dvel) * 1e-3
    conf.edit('solvers', {
        'ns': ('nls.newton', {'i_max': 1, 'eps_a': aeps,
                              'problem': 'nonlinear'}),
        'ls': ('ls.mumps', {})})
    lpb = Problem.from_conf(conf)
    lpb.dvelocity = dvel
    lpb.time_update()
    lpb.init_solvers()
    lpb.set_linear(False)
    dvel_f = lpb.solve().get_state_parts()

    nnod = pb.domain.regions['Yf'].vertices.shape[0]
    return dvel_f['uf'].reshape((nnod, dim))


class CorrDVel(cb.CorrMiniApp):
    def __call__(self, problem=None, data=None):
        pb = get_default(problem, self.problem)
        odir = pb.conf.options['output_dir']

        dim = self.dim
        domain = pb.domain
        mesh = domain.mesh
        mmap = domain.regions['Ys'].vertices
        cmap = domain.regions['Yf'].vertices

        lab = '_' + self.corr_flag
        if self.corr_flag == 'p':
            state_u = -data[self.requires[-1]].state['u']
        elif self.corr_flag[0] == 'r':
            state_u = data[self.requires[-1]].state['u']
        elif self.corr_flag == 'e':
            state_u = data[self.requires[-1]].states[self.idxs]['u']
            lab += '%d%d' % self.idxs

        out = nm.zeros((mesh.n_nod, dim), dtype=nm.float64)
        out[mmap, :] = state_u.reshape(mmap.shape[0], dim)
        out[cmap, :] = define_Yf_dvel(pb, out)

        if self.corr_flag == 'e':
            pis = data[self.requires[0]]
            out += pis.states[self.idxs]['V'].reshape(mesh.n_nod, dim)

        corr_sol = cb.CorrSolution(name=self.name + lab,
                                   state={'v': out.flatten()})

        mout = {'dvelocity': Struct(name='output_data', mode='vertex',
                                    data=out, dofs=None)}
        mesh.write(osp.join(odir, 'piezo_micro_dvelocity%s.vtk' % lab),
                   out=mout, io='auto')

        multiproc_dependecies['dvelocity' + lab] = out

        return corr_sol


class CorrDVel_d_pis_u(cb.CorrMiniApp):
    def __call__(self, problem=None, data=None):
        problem = get_default(problem, self.problem)
        domain = problem.domain
        mesh = domain.mesh

        mmap = domain.regions['Ys'].vertices
        key = list(data.keys())[0]
        dvel = data[key].state['v'].reshape((-1, mesh.dim))
        clist, dout = create_pis(dvel[mmap, :], vname='u')
        corr_sol = cb.CorrSolution(name=self.name,
                                   states=dout,
                                   components=clist)

        return corr_sol


def replace_dvel_in_def(coef, dvlab):
    def repl_in_list(svlist, key, dvlab):
        for k in range(len(svlist)):
            item = svlist[k]
            if isinstance(item, list):
                repl_in_list(item, key, dvlab)
            elif item == key:
                svlist[k] = (item[0], item[1] + dvlab, item[2])

    out = deepcopy(coef)
    # replace "dvelocity" by "dvelocity_xy"
    req = out['requires']
    if 'dvelocity' in req:
        idx = req.index('dvelocity')
        req[idx] = 'dvelocity' + dvlab

    if 'd_pis_u' in req:
        idx = req.index('d_pis_u')
        req[idx] = 'd_pis_u' + dvlab

    if 'set_variables' in out:
        repl_in_list(out['set_variables'], ('V', 'dvelocity', 'v'), dvlab)
        repl_in_list(out['set_variables'], ('U1', 'd_pis_u', 'u'), dvlab)
        repl_in_list(out['set_variables'], ('U2', 'd_pis_u', 'u'), dvlab)

    if out['class'] is cb.CoefEval:
        repl = []
        new_req = []

        for jj in out['requires']:
            if jj[0:3] == 'c.s' or jj == 'c.divV_Y0' or jj == 'c.divV_Yf':
                new_req.append(jj + dvlab)
                repl.append((jj, jj + dvlab))
            else:
                new_req.append(jj)

        out['requires'] = new_req
        for jj in repl:
            out['expression'] = out['expression'].replace(jj[0], jj[1])

    return out


def merge_locals(locals, d):
    out = {}
    dkeys = list(d.keys())
    for k, v in locals.items():
        if k in d:
            if isinstance(v, dict) and isinstance(d[k], dict):
                d[k].update(v)
                out[k] = d[k]
            elif isinstance(v, list) and isinstance(d[k], list):
                out[k] = v + d[k]
            elif isinstance(v, tuple) and isinstance(d[k], tuple):
                out[k] = v + d[k]
            else:
                out[k] = v

            dkeys.remove(k)
        else:
            out[k] = v

    out.update({k: d[k] for k in dkeys})

    return out


def define(eps0=1e-3,
           filename_mesh=osp.join(wdir, 'mesh_micro.vtk'),
           fluid_mode='C', cond_mode='C', mat_mode='elastic_part', flag='',
           filename_coefs=None):

    d = poropiezo_micro.define(eps0, filename_mesh, fluid_mode,
                               cond_mode, mat_mode,
                               flag=flag, filename_coefs=filename_coefs)
    dim = d['dim']

    fields = {
        'dvelocity': ('real', 'vector', 'Y', 1),
        'displacement_Yf': ('real', 'vector', 'Yf', 1),
    }

    variables = {
        'V': ('parameter field', 'dvelocity', '(set-to-None)'),
        'uf': ('unknown field', 'displacement_Yf'),
        'vf': ('test field', 'displacement_Yf', 'uf'),
    }

    options = {
        'return_all': True,
        'recovery_hook': recovery_micro,
    }

    coefs = {
        'divV_Y0': {
            'requires': ['dvelocity'],
            'expression': 'ev_div.i2.Y(V)',
            'set_variables': [('V', 'dvelocity', 'v')],
            'class': cb.CoefOne,
        },
        'divV_Yf': {
            'requires': ['dvelocity'],
            'expression': 'ev_div.i2.Yf(V)',
            'set_variables': [('V', 'dvelocity', 'v')],
            'class': cb.CoefOne,
        },
        'sPhi': {
            'status': 'auxiliary',
            'requires': ['c.divV_Y0', 'c.divV_Yf', 'c.vol'],
            'expression': 'c.divV_Yf - c.vol["fraction_Yf"] * c.divV_Y0',
            'class': cb.CoefEval,
        },
        ##### sA
        'sA1': {
            'status': 'auxiliary',
            'requires': ['pis_u', 'corrs_rs', 'dvelocity'],
            'expression': '   ev_sd_lin_elastic.i2.Ys(matrix.D, U1, U2, V)'
                          ' - ev_sd_diffusion.i2.Yp(piezo.d, R1, R2, V)',
            'set_variables': [[('U1', ('corrs_rs', 'pis_u'), 'u'),
                               ('R1', 'corrs_rs', 'r')],
                              [('U2', ('corrs_rs', 'pis_u'), 'u'),
                               ('R2', 'corrs_rs', 'r')],
                              ('V', 'dvelocity', 'v')],
            'class': cb.CoefSymSym,
        },
        'sA2': {
            'status': 'auxiliary',
            'requires': ['pis_u', 'corrs_rs', 'd_pis_u', 'dvelocity'],
            'expression': '   dw_lin_elastic.i2.Ys(matrix.D, U1, U2)'
                          ' - dw_piezo_coupling.i2.Yp(piezo.g, U1, R2)'
                          ' - ev_sd_piezo_coupling.i2.Yp(piezo.g, U2, R1, V)',
            'set_variables': [[('U1', 'd_pis_u', 'u'),
                               ('R1', 'corrs_rs', 'r')],
                              [('U2', ('corrs_rs', 'pis_u'), 'u'),
                               ('R2', 'corrs_rs', 'r')],
                              ('V', 'dvelocity', 'v')],
            'class': cb.CoefSymSym,
        },
        'sA': {
            'requires': ['c.sA1', 'c.sA2'],
            'expression': 'c.sA1 + c.sA2 + c.sA2.T',
            'class': cb.CoefEval,
        },
        #### sB
        'sB1_div1': {
            'status': 'auxiliary',
            'requires': ['corr_one', 'corrs_rs', 'dvelocity'],
            'expression': 'ev_sd_div.i2.Ys(U1, svar, V)',
            'set_variables': [('U1', 'corrs_rs', 'u'),
                              ('svar', 'corr_one', 'sv'),
                              ('V', 'dvelocity', 'v')],
            'class': cb.CoefSym,
        },
        'sB1_div2': {
            'status': 'auxiliary',
            'requires': ['corrs_rs', 'dvelocity'],
            'expression': 'ev_div.i2.Ys(U1)',
            'set_variables': [('U1', 'corrs_rs', 'u')],
            'class': cb.CoefSym,
        },
        'sB2': {
            'status': 'auxiliary',
            'requires': ['corrs_p', 'd_pis_u'],
            'expression': '   dw_lin_elastic.i2.Ys(matrix.D, U1, U2)'
                          ' - dw_piezo_coupling.i2.Yp(piezo.g, U1, R1)',
            'set_variables': [('U1', 'd_pis_u', 'u'),
                              ('U2', 'corrs_p', 'u'),
                              ('R1', 'corrs_p', 'r')],
            'class': cb.CoefSym,
        },
        'sB3': {
            'status': 'auxiliary',
            'requires': ['pis_u', 'corrs_rs', 'corrs_p', 'dvelocity'],
            'expression': '   ev_sd_lin_elastic.i2.Ys(matrix.D, U1, U2, V)'
                          ' - ev_sd_piezo_coupling.i2.Yp(piezo.g, U2, R1, V)'
                          ' - ev_sd_piezo_coupling.i2.Yp(piezo.g, U1, R2, V)'
                          ' - ev_sd_diffusion.i2.Yp(piezo.d, R1, R2, V)',
            'set_variables': [('U1', ('corrs_rs', 'pis_u'), 'u'),
                              ('U2', 'corrs_p', 'u'),
                              ('R1', 'corrs_rs', 'r'),
                              ('R2', 'corrs_p', 'r'),
                              ('V', 'dvelocity', 'v')],
            'class': cb.CoefSym,
        },
        'sB': {
            'requires': ['c.sB1_div1', 'c.sB1_div2', 'c.sB2', 'c.sB3',
                         'c.divV_Y0', 'c.sPhi'],
            'expression': '%s * c.sPhi' % d['sym_eye'] +
                          ' - (c.sB1_div1 - c.divV_Y0 * c.sB1_div2)'
                          ' + c.sB2 + c.sB3',
            'class': cb.CoefEval,
        },
        #### sM
        'sMsurf1': {
            'status': 'auxiliary',
            'requires': ['corr_one', 'corrs_p', 'dvelocity'],
            'expression': 'ev_sd_div.i2.Ys(U1, svar, V)',
            'set_variables': [('U1', 'corrs_p', 'u'),
                              ('svar', 'corr_one', 'sv'),
                              ('V', 'dvelocity', 'v')],
            'class': cb.CoefOne,
        },
        'sMsurf2': {
            'status': 'auxiliary',
            'requires': ['corrs_p', 'dvelocity'],
            'expression': 'ev_div.i2.Ys(U1)',
            'set_variables': [('U1', 'corrs_p', 'u')],
            'class': cb.CoefOne,
        },
        'sMsurf': {
            'status': 'auxiliary',
            'requires': ['c.sMsurf1', 'c.sMsurf2', 'c.divV_Y0'],
            # \int_{Gamma_c} x \cdot n = - \int_{Ys} div(z) + ...
            'expression': '-(c.sMsurf1 - c.divV_Y0 * c.sMsurf2)',
            'class': cb.CoefEval,
        },
        'sM2': {
            'status': 'auxiliary',
            'requires': ['corrs_p', 'dvelocity'],
            'expression': ' 2*ev_sd_piezo_coupling.i2.Yp(piezo.g, U1, R1, V)'
                          ' + ev_sd_diffusion.i2.Yp(piezo.d, R1, R2, V)'
                          ' - ev_sd_lin_elastic.i2.Ys(matrix.D, U1, U2, V)',
            'set_variables': [('U1', 'corrs_p', 'u'),
                              ('U2', 'corrs_p', 'u'),
                              ('R1', 'corrs_p', 'r'),
                              ('R2', 'corrs_p', 'r'),
                              ('V', 'dvelocity', 'v')],
            'class': cb.CoefOne,
        },
        'sM': {
            'requires': ['c.sMsurf', 'c.sM2', 'c.sPhi'],
            'expression': '%e * c.sPhi' % d['materials']['fluid'][0]['gamma'] +
                          ' - 2*c.sMsurf + c.sM2',
            'class': cb.CoefEval,
        },
        ### sK
        'sK2': {
            'status': 'auxiliary',
            'requires': ['corrs_psi'],
            'expression': 'dw_stokes.i3.Yf(W1, P1)',
            'set_variables': [('W1', 'corrs_psi', 'w'),
                              ('P1', 'corrs_psi', 'p')],
            'class': cb.CoefDimDim,
        },
        'sK3': {
            'status': 'auxiliary',
            'requires': ['pis_w', 'corrs_psi', 'dvelocity'],
            'expression': '   ev_sd_volume_dot.i3.Yf(W1, W2, V)'
                          ' + ev_sd_div.i3.Yf(W1, P2, V)',
            'set_variables': [('W1', 'corrs_psi', 'w'),
                              [('W2', 'pis_w', 'w'),
                               ('P2', 'corrs_psi', 'p')],
                              ('V', 'dvelocity', 'v')],
            'class': cb.CoefDimDim,
        },
        'sK4': {
            'status': 'auxiliary',
            'requires': ['corrs_psi', 'dvelocity'],
            'expression': 'ev_sd_div_grad.i3.Yf(W1, W2, V)',
            'set_variables': [('W1', 'corrs_psi', 'w'),
                              ('W2', 'corrs_psi', 'w'),
                              ('V', 'dvelocity', 'v')],
            'class': cb.CoefDimDim,
        },
        'sK': {
            'requires': ['c.K', 'c.sK2', 'c.sK3', 'c.sK4', 'c.divV_Y0'],
            'expression': '-(c.K + c.sK2 + c.sK2.T) * c.divV_Y0'
                          ' - c.sK4 + c.sK3 + c.sK3.T',
            'class': cb.CoefEval,
        },
    }

    requirements = {
        #  = V_j \delta_{ik} in Ys
        'd_pis_u': {
            'requires': ['dvelocity'],
            'class': CorrDVel_d_pis_u,
        },
        'pis_v': {
            'variables': ['V'],
            'class': cb.ShapeDimDim,
        },
        'corr_one': {
            'variable': 'sv',
            'expression': "nm.ones((problem.fields['sfield'].n_vertex_dof, 1), dtype=nm.float64)",
            'class': cb.CorrEval,
        },
        'dvelocity': {
            'variable': 'v',
            'expression': 'problem.conf.inter_data',
            'class': cb.CorrEval,
            # 'save_name': 'design_velocity',
            # 'dump_variables': ['v'],
        },
        'dvelocity_p': {
            'requires': ['corrs_p'],
            'dim': dim,
            'corr_flag': 'p',
            'class': CorrDVel,
            # 'save_name': 'dvelocity_p',
            # 'dump_variables': ['v'],
        },
    }

    dvlist = ['_p']

    for k in range(d['n_conduct']):
        sk = '%d' % k
        ck = '_' + sk
        coefs.update({
            'sV1' + ck: {
                'status': 'auxiliary',
                'requires': ['d_pis_u', 'corrs_k' + sk],
                'expression': '   dw_lin_elastic.i2.Ys(matrix.D, U1, U2)'
                              ' - dw_piezo_coupling.i2.Yp(piezo.g, U1, R1)',
                'set_variables': [('U1', 'd_pis_u', 'u'),
                                  ('U2', 'corrs_k' + sk, 'u'),
                                  ('R1', 'corrs_k' + sk, 'r')],
                'class': cb.CoefSym,
            },
            'sV2' + ck: {
                'status': 'auxiliary',
                'requires': ['pis_u', 'corrs_rs', 'corrs_k' + sk, 'dvelocity'],
                'expression': '   ev_sd_lin_elastic.i2.Ys(matrix.D, U2, U1, V)'
                              ' - ev_sd_piezo_coupling.i2.Yp(piezo.g, U1, R1, V)'
                              ' - ev_sd_diffusion.i2.Yp(piezo.d, R1, R2, V)'
                              ' - ev_sd_piezo_coupling.i2.Yp(piezo.g, U2, R2, V)',
                'set_variables': [('R1', 'corrs_rs', 'r'),
                                  ('R2', 'corrs_k' + sk, 'r'),
                                  ('U1', 'corrs_k' + sk, 'u'),
                                  ('U2', ('corrs_rs', 'pis_u'), 'u'),
                                  ('V', 'dvelocity', 'v')],
                'class': cb.CoefSym,
            },
            'sV' + sk: {
                'requires': ['c.sV1' + ck, 'c.sV2' + ck],
                'expression': 'c.sV1%s + c.sV2%s' % (ck, ck),
                'class': cb.CoefEval,
            },
            #### sZ
            'sZsurf1' + ck: {
                'status': 'auxiliary',
                'requires': ['corr_one', 'corrs_k' + sk, 'dvelocity'],
                'expression': 'ev_sd_div.i2.Ys(U1, svar, V)',
                'set_variables': [('U1', 'corrs_k' + sk, 'u'),
                                  ('svar', 'corr_one', 'sv'),
                                  ('V', 'dvelocity', 'v')],
                'class': cb.CoefOne,
            },
            'sZsurf2' + ck: {
                'status': 'auxiliary',
                'requires': ['corrs_k' + sk, 'dvelocity'],
                'expression': 'ev_div.i2.Ys(U1)',
                'set_variables': [('U1', 'corrs_k' + sk, 'u')],
                'class': cb.CoefOne,
            },
            'sZsurf' + ck: {
                'status': 'auxiliary',
                'requires': ['c.sZsurf1' + ck, 'c.sZsurf2' + ck, 'c.divV_Y0'],
                # \int_{Gamma_c} z \cdot n = - \int_{Ys} div(z) + ...
                'expression': '-(c.sZsurf1%s - c.divV_Y0 * c.sZsurf2%s)' % (ck, ck),
                'class': cb.CoefEval,
            },  
            'sZ1' + ck: {
                'status': 'auxiliary',
                'requires': ['corrs_p', 'corrs_k' + sk, 'dvelocity'],
                'expression': '   ev_sd_piezo_coupling.i2.Yp(piezo.g, U1, R1, V)'
                              ' - ev_sd_lin_elastic.i2.Ys(matrix.D, U1, U2, V)'
                              ' + ev_sd_piezo_coupling.i2.Yp(piezo.g, U2, R2, V)'
                              ' + ev_sd_diffusion.i2.Yp(piezo.d, R1, R2, V)',
                'set_variables': [('U1', 'corrs_p', 'u'),
                                  ('U2', 'corrs_k' + sk, 'u'),
                                  ('R1', 'corrs_k' + sk, 'r'),
                                  ('R2', 'corrs_p', 'r'),
                                  ('V', 'dvelocity', 'v')],
                'class': cb.CoefOne,
            },
            'sZ' + sk: {
                'requires': ['c.sZ1' + ck, 'c.sZsurf' + ck],
                'expression': 'c.sZ1%s - c.sZsurf%s' % (ck, ck),
                'class': cb.CoefEval,
            },
        })

        dvlab = '_r' + sk
        dvlist.append(dvlab)
        requirements.update({
            'dvelocity' + dvlab: {
                'requires': ['corrs_k' + sk],
                'dim': dim,
                'corr_flag': 'r' + sk,
                'class': CorrDVel,
            },
        })

    for ii, irc in enumerate(iter_sym(dim)):
        dvlab = '_e%d%d' % irc
        dvlist.append(dvlab)
        requirements.update({
            'dvelocity' + dvlab: {
                'requires': ['pis_v', 'corrs_rs'],
                'dim': dim,
                'corr_flag': 'e',
                'idxs': irc,
                'class': CorrDVel,
            },
        })

    coefs_keys = [k for k in coefs.keys()
                  if (k.startswith('s') or k.startswith('divV'))]

    for dvlab in dvlist:
        requirements.update({
            'd_pis_u' + dvlab: {
                'requires': ['dvelocity' + dvlab],
                'class': CorrDVel_d_pis_u,
            },
        })

        for ck in coefs_keys:
            coefs[ck + dvlab] = replace_dvel_in_def(coefs[ck], dvlab)

    for ck in coefs_keys:
        del(coefs[ck])

    odir = d['options']['output_dir']
    with open(osp.join(odir, 'coefs_poropiezo_def.py'), 'wt') as f:
        for ck in coefs.keys():
            f.write('===== %s =====\n' % ck)
            cv = coefs[ck]
            for k in cv.keys():
                f.write('    %s: %s\n' % (k, str(cv[k])))

    def_dict_new = merge_locals(locals(), d)
    # def_dict_new['coefs'] = coefs

    return def_dict_new
