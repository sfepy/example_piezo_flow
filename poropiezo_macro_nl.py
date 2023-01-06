import sys
import numpy as nm
import os.path as osp
from sfepy.base.base import Struct
from sfepy.base.base import debug
from sfepy.homogenization.micmac import get_homog_coefs_linear
from sfepy.homogenization.utils import iter_sym, define_box_regions
import sfepy.discrete.fem.periodic as per
from scipy.io import savemat
from poroela_utils import coefs2qp
from sfepy.linalg.utils import dot_sequences
from sfepy.solvers.ts import TimeStepper
from sfepy.postprocess.probes_vtk import Probe, ProbeFromFile
from sfepy.homogenization.recovery import recover_micro_hook

wdir = osp.dirname(__file__)


def save_recovery_region(pb):
    rreg = pb.domain.regions['Recovery']
    rreg_id = nm.zeros((pb.domain.mesh.n_el, 1, 1, 1), dtype=nm.int32)
    rreg_id[rreg.get_cells()] = 1
    out_rreg = {'recovery': Struct(name='recovery_region',
                                   mode='cell', data=rreg_id)}
    filename = f'{pb.ofn_trunk}_rreg.vtk'
    filename = osp.join(pb.conf.options.get('output_dir', '.'), filename)
    pb.domain.mesh.write(filename, out=out_rreg)


def recover_micro(pb, state):
    rreg = pb.domain.regions['Recovery']
    ts = pb.conf.tstep

    eval_vars = pb.get_variables(['U', 'P'])

    if ts.step == 0:
        save_recovery_region(pb)

    hflag = f'_{pb.conf.fluid_mode}{pb.conf.cond_mode}' +\
            f'{pb.conf.cf_flag}_{pb.conf.eps0}'
    coefs_filename = f'coefs_poropiezo{hflag}'

    def_args = {
        'eps0': pb.conf.eps0,
        'filename_mesh': pb.conf.filename_mesh_micro,
        'fluid_mode': pb.conf.fluid_mode,
        'cond_mode': pb.conf.cond_mode,
        'filename_coefs': coefs_filename,
    }

    coors = eval_vars['P'].field.coors
    phi_ampl = pb.conf.phi_ampl
    phi_, _ = get_bolus(pb.conf.tstep, coors, pb)
    phi = nm.hstack([phi_.T * phi_ampl[0], phi_.T * phi_ampl[1]])

    macro_data = {
        'displ': ('val', 'U', state['u'].reshape((-1, 3))),
        'strain': ('cauchy_strain', 'U', state['u'].reshape((-1, 3))),
        'press': ('val', 'P', state['p'].reshape((-1, 1))),
        'pressg': ('grad', 'P', state['p'].reshape((-1, 1))),
        'phi': ('val', 'P', nm.ascontiguousarray(phi)),
    }

    recover_micro_hook(osp.join(wdir, pb.conf.filename_micro), rreg,
                       macro=macro_data, eps0=pb.conf.eps0,
                       eval_vars=eval_vars, eval_mode='continuous',
                       recovery_file_tag=f'_{pb.conf.eps0}.{ts.step:03d}',
                       define_args=def_args)


def post_process_final(pb):
    print('>>> final post-processing')
    cache = pb.cache
    out = {}

    out_data = cache['out_data']
    for k, field in enumerate(out_data[0]):
        val = []
        for step in out_data[1:]:
            val.append(step[k])

        out[field] = nm.array(val)

    out_coefs = cache['out_coefs']
    for k, v in out_coefs.items():
        out[k] = nm.vstack(v)

    out['niter'] = cache['niter']

    out.update({k: nm.array(v) for k, v in cache.items()
                if k.startswith('fluxw_')})

    out.update({k: nm.array(v) for k, v in cache.items()
                if k.startswith('flux_')})

    filename = osp.join(pb.conf.options.get('output_dir', '.'),
                        f'poropiezo_data{pb.conf.flag}.mat')
    savemat(filename, out)
    print(f'>>> data saved to {filename}')


def post_process(pb):
    print('>>> post-processing')
    out = {}
    cache = pb.cache
    tstep = pb.conf.tstep
    bolus_mode = pb.conf.bolus_mode
    state = cache['state']

    out['u'] = Struct(name='output_data', mode='vertex',
                      var_name='u', data=state['u'].reshape((-1, 3)))
    out['p'] = Struct(name='output_data', mode='vertex',
                      var_name='p', data=state['p'][:, None])

    pvars = pb.get_variables()
    pvars['U'].set_data(state['u'])
    pvars['P'].set_data(state['p'])

    strain = pb.evaluate('ev_cauchy_strain.i2.Omega(U)', mode='el_avg')
    out['e'] = Struct(name='output_data', mode='cell',
                      var_name='u', data=strain)

    k_eta = 'tKeta'
    dvel = pb.evaluate(f'ev_diffusion_velocity.i2.Omega(hom.{k_eta}, P)',
                       mode='el_avg')
    out['w'] = Struct(name='output_data', mode='cell',
                      var_name='p', data=dvel)

    matK = pb.evaluate(f'ev_integrate_mat.i2.Omega(hom.{k_eta}, P)',
                       mode='el_avg')
    out['K'] = Struct(name='output_data', mode='cell',
                      var_name='p', data=matK[:, :, :1, :1])

    r0 = pb.evaluate('ev_volume_integrate_mat.i2.Omega(hom.r0, P)',
                     mode='el_avg')
    out['r0'] = Struct(name='output_data', mode='cell', data=r0)
    r1 = pb.evaluate('ev_volume_integrate_mat.i2.Omega(hom.r1, P)',
                     mode='el_avg')
    out['r1'] = Struct(name='output_data', mode='cell', data=r1)

    bolus = get_bolus(tstep, pb.domain.mesh.coors, pb)[0]
    bolus = nm.vstack([bolus * 0, bolus * 0, bolus]).T * pb.conf.phi_ampl[0]
    out['bolus'] = Struct(name='output_data', mode='vertex', data=bolus)

    fname = f'{pb.ofn_trunk}{pb.conf.flag}_{tstep.step:03d}.vtk'
    pb.domain.mesh.write(osp.join(pb.conf.options.get('output_dir', '.'),
                                  fname), out=out)

    # boundary flux
    pp_regs = [
        ('Left', 0, -1), ('Right', 0, 1),
        ('Near', 1, -1), ('Far', 1, 1),
        ('Bottom', 2, -1), ('Top', 2, 1),
        ('Middle', 0, 1),
    ]

    for reg, dir, mul in pp_regs:
        w = pb.evaluate(f'ev_diffusion_velocity.i2.{reg}(hom.{k_eta}, P)',
                        mode='el_avg')
        S = pb.evaluate(f'ev_volume.i2.{reg}(P)', mode='el_avg')
        Q = (w[..., dir, 0] * S[..., 0, 0]).sum() * tstep.dt * mul
        fl = -pb.evaluate(f'ev_surface_flux.i2.{reg}(hom.{k_eta}, P)',
                          mode='eval')

        key = 'flux_%s' % reg
        if key not in cache:
            cache[key] = []
        cache[key].append(fl * tstep.dt)

        key = 'fluxw_%s' % reg
        if key not in cache:
            cache[key] = []
        cache[key].append(Q)

        key = 'S_%s' % reg
        if key not in cache:
            cache[key] = []
        cache[key].append(S)

        w_avg = nm.average(w, axis=0).squeeze() * mul
        key = 'w_avg_%s' % reg
        if key not in cache:
            cache[key] = []
        cache[key].append(w_avg[dir])

    out_coef_names = ['tA', 'tB', 'tM', 'tKeta']

    if 'out_coefs' not in cache:
        cache['out_coefs'] = {k: [] for k in out_coef_names}

    nc = pb.domain.mesh.n_el
    coefs = cache['mat_coefs_i2_Omega']
    out_coefs = cache['out_coefs']
    cid = out_coefs['cid'] = nm.array([0, nc // 2, nc])

    for k in out_coef_names:
        out_coefs[k].append(coefs[k][cid, ...])

    tstep = pb.conf.tstep
    state = pb.cache['state']

    out_data_head = ('time', 'lc', 'bolus', 'displ', 'pressure', 'strain',
                     'potential_1', 'potential_2', 'velocity')
    out_data_head_rec = ('displ_rec', 'potential_1_rec', 'potential_2_rec',
                         'pressure_rec', 'velocity_rec')

    if 'out_data' not in cache:
        cache['out_data'] = [out_data_head + out_data_head_rec]\
            if pb.conf.is_recovery else [out_data_head]

    eps0 = pb.conf.eps0

    n_points = 100

    probe = Probe(out, pb.domain.mesh)
    p1, p2 = nm.array([[0.0, 0.5, 0.5], [0.1 / eps0, 0.5, 0.5]]) * eps0
    probe.add_line_probe('line0', p1, p2, n_points)

    prb_press = probe('line0', 'p')
    prb_strain = probe('line0', 'e')
    prb_r0 = probe('line0', 'r0')
    prb_r1 = probe('line0', 'r1')
    prb_w = probe('line0', 'w')
    prb_u = probe('line0', 'u')
    lc = prb_press[0][:, None]
    cx = p1 * (1 - lc) + p2 * lc

    bolus = get_bolus(tstep, cx, pb)[0] * pb.conf.phi_ampl[0]

    out_data = (tstep.time, lc, bolus,
                prb_u[1], prb_press[1], prb_strain[1],
                prb_r0[1], prb_r1[1], prb_w[1])

    if pb.conf.is_recovery:
        recover_micro(pb, state)

        fname = osp.split(pb.conf.filename_mesh_micro)[1]
        fname = osp.splitext(fname)[0]
        fname = fname + f'.recovered_{eps0}.{tstep.step:03d}_%s.vtk'
        fname = osp.join(pb.conf.options.get('output_dir', '.'), fname)

        probe2 = ProbeFromFile(fname % 'Ys')
        p1, p2 = nm.array([[0.0, 0.1, 0.3], [0.1 / eps0, 0.1, 0.3]]) * eps0
        probe2.add_line_probe('line1', p1, p2, n_points)
        prb_u_rec = probe2('line1', 'u')

        probe3 = ProbeFromFile(fname % 'Yp')
        p1, p2 = nm.array([[0.0, 0.1, 0.14], [0.1 / eps0, 0.1, 0.14]]) * eps0
        probe3.add_line_probe('line2', p1, p2, n_points)
        prb_r0_rec = probe3('line2', 'phi')
        p1, p2 = nm.array([[0.0, 0.1, 0.86], [0.1 / eps0, 0.1, 0.86]]) * eps0
        probe3.add_line_probe('line3', p1, p2, n_points)
        prb_r1_rec = probe3('line3', 'phi')

        p1, p2 = nm.array([[0.0, 0.5, 0.5], [0.1 / eps0, 0.5, 0.5]]) * eps0
        probe4 = ProbeFromFile(fname % 'Yf')
        probe4.add_line_probe('line4', p1, p2, n_points)
        prb_p_rec = probe4('line4', 'p')
        probe4.add_line_probe('line5', p1, p2, n_points)
        prb_w_rec = probe4('line5', 'w')

        out_data_rec = (prb_u_rec[1], prb_r0_rec[1], prb_r1_rec[1],
                        prb_p_rec[1], prb_w_rec[1])

        cache['out_data'].append(out_data + out_data_rec)
    else:
        cache['out_data'].append(out_data)


def get_homog(nqp, pb):
    cache = pb.cache

    if 'coefs0' not in cache:
        hflag = f'_{pb.conf.fluid_mode}{pb.conf.cond_mode}' +\
                f'{pb.conf.cf_flag}_{pb.conf.eps0}'
        coefs_filename_ = f'coefs_poropiezo{hflag}'
        coefs_filename = osp.join(pb.output_dir, coefs_filename_ + '.h5')

        def_args = {
            'eps0': pb.conf.eps0,
            'filename_mesh': pb.conf.filename_mesh_micro,
            'fluid_mode': pb.conf.fluid_mode,
            'cond_mode': pb.conf.cond_mode,
            'mat_mode': pb.conf.mat_mode,
            'flag': hflag,
            'filename_coefs': coefs_filename_,
        }

        coefs = get_homog_coefs_linear(0, 0, None,
                                       micro_filename=pb.conf.filename_micro,
                                       coefs_filename=coefs_filename,
                                       define_args=def_args)

        out, others = coefs2qp(coefs, nqp, ret_others=True)

        cache['coefs0'] = out
        cache['coefs_others'] = others

    return cache['coefs0']


def clear_cache(cache, key):
    keys = list(cache.keys())
    for k in keys:
        if key in k:
            del(cache[k])


def get_mat(ts, coors, mode=None, problem=None, **kwargs):
    if not (mode == 'qp'):
        return

    print('>>> material function')

    pb = problem
    cache = pb.cache

    nlmul, nlmulK = (1, 1) if pb.conf.is_nonlin else (0, 0)

    term = kwargs['term']
    int_reg = term.integral.name, term.region.name
    cf_cache_name = 'mat_coefs_%s_%s' % int_reg

    if (cf_cache_name not in cache):
        print('>>>   updating')
        nqp, dim = coors.shape
        sym = 3 * dim - 3

        tstep = pb.conf.tstep

        phi_ampl = problem.conf.phi_ampl
        phi_, dphi_ = get_bolus(tstep, coors, pb)
        phi_ = phi_.reshape((nqp, 1, 1))
        dphi_ = dphi_.reshape((nqp, 1, 1))

        phi = [phi_ * phi_ampl[0], phi_ * phi_ampl[1]]
        dphi = [dphi_ * phi_ampl[0], dphi_ * phi_ampl[1]]

        coefs = get_homog(nqp, pb)

        state = cache['state']

        vars = pb.create_variables(['U', 'P'])
        st = {}
        for ivar, iout in [('u', 'e'), ('u0', 'e0'),
                           ('p', 'p'), ('p0', 'p0'), ('p', 'gp')]:
            if ivar[0] == 'u':
                vars['U'].set_data(state[ivar])
                st[iout] = pb.evaluate('ev_cauchy_strain.%s.%s(U)' % int_reg,
                                       mode='qp',
                                       U=vars['U']).reshape((nqp, sym, 1))
            elif ivar[0] == 'p':
                vars['P'].set_data(state[ivar])
                if iout[0] == 'g':
                    st[iout] = pb.evaluate('ev_grad.%s.%s(P)' % int_reg,
                                           mode='qp',
                                           P=vars['P']).reshape((nqp, dim, 1))
                else:
                    st[iout] = pb.evaluate('ev_volume_integrate.%s.%s(P)' % int_reg,
                                           mode='qp',
                                           P=vars['P']).reshape((nqp, 1, 1))

        assert(nqp == nm.prod(st['p'].shape[:2]))

        bar_eta = coefs['bar_eta']
        n_cond = len(phi)
        coefs_sa = ['A', 'B', 'M', 'K']

        coefs_out = {}
        for cf in coefs_sa:
            val = coefs[f's{cf}_p'] * st['p']
            for ii, irc in enumerate(iter_sym(dim)):
                key = 's%s_e%d%d' % ((cf,) + irc)
                val += coefs[key] * (st['e'][:, ii, :].reshape(nqp, 1, 1))

            for ir in range(n_cond):
                key = 's%s_r%d' % (cf, ir)
                val += coefs[key] * phi[ir]

            if cf == 'K':
                coefs_out[cf + 'eta'] = coefs[cf] / bar_eta
                coefs_out[f't{cf}eta'] = (coefs[cf] + val * nlmulK) / bar_eta
            else:
                coefs_out[cf] = coefs[cf]
                coefs_out[f't{cf}'] = coefs[cf] + val * nlmul

        st['de'] = st['e'] - st['e0']
        st['dp'] = st['p'] - st['p0']

        cf_str_phi = []
        for k in range(n_cond):
            st[f'r{k}'] = phi[k]
            coefs_out[f'r{k}'] = phi[k]
            st[f'dr{k}'] = dphi[k]
            cf_str_phi.append(f'V{k}|r{k}|e')
            cf_str_phi.append(f'V{k}|r{k}|p')
            cf_str_phi.append(f'Z{k}|dr{k}|e')
            cf_str_phi.append(f'Z{k}|dr{k}|p')

        coefs_x = {}
        for cf_str in ['A|e|e', 'A|e|p',
                       'B|p|e', 'B|p|p',
                       'BT|e|e', 'BT|de|e', 'BT|e|p', 'BT|de|p',
                       'M|dp|e', 'M|dp|p',
                       'K|gp|e', 'K|gp|p'] + cf_str_phi:
            cf, mul, op = cf_str.split('|')
            t_flag = False
            if cf[-1] == 'T':
                cf = cf[:-1]
                t_flag = True

            stval = st[mul]

            if op == 'e':
                d1 = coefs[cf].shape[2] if t_flag else coefs[cf].shape[1]
                d2 = st[op].shape[1]
                val = nm.zeros((nqp, d1, d2), dtype=nm.float64)
                for ii, irc in enumerate(iter_sym(dim)):
                    key = 's%s_e%d%d' % ((cf,) + irc)
                    cfval = coefs[key].transpose((0, 2, 1)) if t_flag\
                        else coefs[key]
                    if mul[0] == 'e' or mul[:2] == 'gp' or mul[:2] == 'de':
                        val[..., ii] = dot_sequences(cfval, stval)[..., 0]
                    elif mul[0] == 'p' or mul[0] == 'r'\
                            or mul[:2] == 'dp' or mul[:2] == 'dr':
                        val[..., ii] = (cfval * stval)[..., 0]
                    else:
                        raise(ValueError)

            elif op == 'p':
                key = 's%s_p' % cf
                cfval = coefs[key].transpose((0, 2, 1)) if t_flag\
                    else coefs[key]
                if mul[0] == 'e' or mul[:2] == 'gp' or mul[:2] == 'de':
                    val = dot_sequences(cfval, stval)
                elif mul[0] == 'p' or mul[0] == 'r'\
                        or mul[:2] == 'dp' or mul[:2] == 'dr':
                    val = cfval * stval
                else:
                    raise(ValueError)

            else:
                raise(ValueError)

            if val.shape[-1] > val.shape[-2]:
                val = val.transpose((0, 2, 1))
            coefs_x[f's{cf}_{mul}{op}'] = val

        coefs_out['bA'] = coefs_out['tA']\
            + (coefs_x['sA_ee'] - coefs_x['sB_pe']) * nlmul
        coefs_out['bB'] = coefs_out['tB']\
            + (coefs_x['sB_pp'] - coefs_x['sA_ep']) * nlmul
        coefs_out['bD'] = coefs_out['tB']\
            + (coefs_x['sB_dee'] + coefs_x['sM_dpe']) * nlmul
        coefs_out['bM'] = coefs_out['tM']\
            + (coefs_x['sM_dpp'] + coefs_x['sB_dep']) * nlmul
        coefs_out['bKeta'] = coefs_out['tKeta']
        coefs_out['bGeta'] = coefs_x['sK_gpe'].transpose((0, 2, 1))\
            / bar_eta * nlmulK
        coefs_out['bQeta'] = coefs_x['sK_gpp'] / bar_eta * nlmulK

        # potential
        Vf, Zf, dZf = 0, 0, 0
        for k in range(len(phi)):
            coefs_out[f'r{k}'] = phi[k]
            coefs_out[f'dr{k}'] = dphi[k]
            Vf += coefs[f'V{k}'] * phi[k]
            Zf += coefs[f'Z{k}'] * phi[k]
            dZf += coefs[f'Z{k}'] * dphi[k]

            coefs_out['bA'] += coefs_x[f'sV{k}_r{k}e'] * nlmul
            coefs_out['bB'] -= coefs_x[f'sV{k}_r{k}p'] * nlmul
            coefs_out['bD'] -= coefs_x[f'sZ{k}_dr{k}e'] * nlmul
            coefs_out['bM'] -= coefs_x[f'sZ{k}_dr{k}p'] * nlmul

        coefs_out['Vf'] = Vf
        coefs_out['Zf'] = Zf
        coefs_out['dZf'] = dZf

        cache[cf_cache_name] = coefs_out

    return cache[cf_cache_name]


def time_stepping(pb):
    pb.cache = cache = {}
    vars = pb.get_variables()
    cache['state'] = state = {k: nm.zeros((vars[k[0]].n_dof,),
                                          dtype=nm.float64)
                              for k in ['u', 'p', 'u0', 'p0']}
    cache['niter'] = []

    tstep = pb.conf.tstep

    pb.do_post_process = False

    out = []
    for step, time in tstep:
        print('##################################################')
        print(f'  step: {step}')
        print('##################################################')

        norm_state = {'u': 1., 'p': 1.}

        iiter = 1
        next_iteration = True
        while next_iteration:

            for k in ['u', 'p', 'u0', 'p0']:
                vars[k.upper()].set_data(state[k])

            clear_cache(cache, 'mat_')

            yield pb, out

            new_state = out[-1][1].get_state_parts()
            for k in ['u', 'p']:
                state[k] += new_state[k]
                norm_state[k] = nm.linalg.norm(new_state[k])

            print('--------------------------------------------------')
            print(f'  iter: {iiter}')
            print(f'  u: {norm_state["u"]}')
            print(f'  p: {norm_state["p"]}')
            print('--------------------------------------------------')

            iiter += 1

            pb.ebcs.zero_dofs()

            if norm_state['u'] < 1e-6:
                next_iteration = False
                post_process(pb)

            if iiter > 20:
                print('maximal number of iterations!!!')
                debug()

            yield None

        st = state
        print('##################################################')
        print(f'  step: {step}')
        print(f"  u: min = {nm.min(nm.abs(st['u']))}, max = {nm.max(nm.abs(st['u']))}")
        print(f"  p: min = {nm.min(nm.abs(st['p']))}, max = {nm.max(nm.abs(st['p']))}")
        print('##################################################')

        cache['niter'].append(iiter)

        for k in ['u', 'p']:
            state[k + '0'][:] = state[k]

    post_process_final(pb)


def get_bolus_cos_xy(ts, coors):
    L = 0.1
    c = 3  # wave speed [L/s]
    X = 0.6  # wave length [L]
    # Y = 0.4  # wave shift in y [L]
    Y = 0  # wave shift in y [L]

    k = 2 * nm.pi / (X * L)
    t, dt = ts.time, ts.dt

    n = coors.shape[0]
    x = coors[:, 0].reshape((1, n))
    y = coors[:, 1].reshape((1, n))
    if isinstance(t, nm.ndarray):
        t = t[:, None]

    out = nm.zeros((1, n), dtype=nm.float64)
    dout = nm.zeros((1, n), dtype=nm.float64)

    # x0 = y * Y
    x0 = (L - y) * Y
    par = x - c * L * t + x0
    ang = k * par + nm.pi

    psi = 0.5 * (1 + nm.cos(ang))
    lv = par < 0
    out[lv] = psi[lv]
    dout[lv] = 0.5 * c * L * k * nm.sin(ang[lv])

    return out, dout * dt


def get_bolus_cos(ts, coors):
    L = 0.1
    c = 3  # wave speed [L/s]
    X = 0.6  # wave length [L]

    k = 2 * nm.pi / (X * L)
    t, dt = ts.time, ts.dt

    n = coors.shape[0]
    x = coors[:, 0].reshape((1, n))
    if isinstance(t, nm.ndarray):
        t = t[:, None]

    out = nm.zeros((1, n), dtype=nm.float64)
    dout = nm.zeros((1, n), dtype=nm.float64)

    ang = k * (x - c * L * t) + nm.pi

    psi = 0.5 * (1 + nm.cos(ang))
    lv = x - c * L * t < 0
    out[lv] = 0.5 * (1 + nm.cos(ang[lv]))
    dout[lv] = 0.5 * c * L * k * nm.sin(ang[lv])

    return out, dout * dt


def get_bolus_abssin(ts, coors):
    L = 0.1
    c = 3  # wave speed [L/s]
    X = 1.2  # wave length [L]

    k = 2 * nm.pi / (X * L)
    t, dt = ts.time, ts.dt

    n = coors.shape[0]
    x = coors[:, 0].reshape((1, n))
    if isinstance(t, nm.ndarray):
        t = t[:, None]

    out = nm.zeros((1, n), dtype=nm.float64)
    dout = nm.zeros((1, n), dtype=nm.float64)

    ang = k * (x - c * L * t) + nm.pi
    psi = nm.abs(nm.sin(ang))
    lv = x - c * L * t < 0
    out[lv] = nm.abs(nm.sin(ang[lv]))
    dout[lv] = -c * L * k * nm.cos(ang[lv]) * nm.sign(nm.sin(ang[lv]))

    return out, dout * dt


def get_const(ts, coors):
    n = coors.shape[0]
    out = nm.ones((1, n))

    return out, out * 0


def get_lin(ts, coors):
    t, t1, dt = ts.time, ts.t1, ts.dt
    n = coors.shape[0]

    out = nm.ones((1, n), dtype=nm.float64) * t / t1
    dout = nm.zeros((1, n), dtype=nm.float64) / t1

    return out, dout


# phi, dphi (*dt)
def get_bolus(ts, coors, problem):
    ts = problem.conf.tstep
    mode = problem.conf.bolus_mode

    if mode == 'cos':
        out = get_bolus_cos(ts, coors)
    elif mode == 'abssin':
        out = get_bolus_abssin(ts, coors)
    elif mode == 'cos_xy':
        out = get_bolus_cos_xy(ts, coors)
    elif mode == 'lin':
        out = get_lin(ts, coors)

    return out


def move_ux(ts, coor, **kwargs):
    problem = kwargs['problem']
    ts = problem.conf.tstep

    ex = -0.02 if problem.conf.bc_flag > 0 else 0.

    return coor[:, 0] * ts.dt * ex


def define(mode='1D', is_nonlin=True, is_recovery=True, t_end=1, t_nstep=50):
    N, eps0, phi_ampl, p_bar = 50, 1e-3, nm.array([1, 0]) * 7e4, 1.0e3  # non-zero pressure gradient

    fluid_mode, cond_mode = 'C', 'C'
    filename_mesh = f'mesh_macro_{N}x1x1.vtk'
    region_box = [[0, 0, 0], [0.1, 0.1/N, 0.1/N]]

    mat_mode = 'elastic_part'
    periodic_mode = ['y', 'z']
    bolus_mode = 'cos'
    bcs = {
        'fix_u': ('Left', {'u.all': 0.0}),
        'fix_p_left': ('Left', {'p.0': 0.0}),
        'fix_p_right': ('Right', {'p.0': p_bar}),
    }

    filename_mesh_micro = f'mesh_micro_{mode}.vtk'
    flag = f'_{mode}_{"nl" if is_nonlin else "l"}_{N}'
    cf_flag = f'_{mode}'

    filename_mesh = osp.join(wdir, filename_mesh)
    filename_mesh_micro = osp.join(wdir, filename_mesh_micro)
    filename_micro = osp.join(wdir, 'poropiezo_micro_sa.py')

    fields = {
        'displacement': ('real', 'vector', 'Omega', 1),
        'pressure': ('real', 'scalar', 'Omega', 1),
        'sfield': ('real', 'scalar', 'Omega', 1),
    }

    variables = {
        'u': ('unknown field', 'displacement', 0, 1),
        'v': ('test field', 'displacement', 'u'),
        'p': ('unknown field', 'pressure', 1, 1),
        'q': ('test field', 'pressure', 'p'),
        'U': ('parameter field', 'displacement', 'u'),
        'P': ('parameter field', 'pressure', 'p'),
        'U0': ('parameter field', 'displacement', 'u'),
        'P0': ('parameter field', 'pressure', 'p'),
        'Q': ('parameter field', 'pressure', 'p'),
        'svar': ('parameter field', 'sfield',  'set-to-none'),
    }

    functions = {
        'move_ux': (move_ux,),
        'get_mat': (get_mat,),
        'match_y_plane': (per.match_y_plane,),
        'match_z_plane': (per.match_z_plane,),
    }

    materials = {
        'hom': 'get_mat',
    }

    integrals = {
        'i2': 2,
    }

    solvers = {
    }

    options = {
        'output_dir': osp.join(wdir, 'results'),
        'nls': 'newton',
        'parametric_hook': 'time_stepping',
    }

    x0, x1 = region_box[0][0], region_box[1][0]
    x2, delta = (x0 + x1) * 0.5, (x1 - x0) * 1e-3

    regions = {
        'Omega': 'all',
        'Recovery': (f'vertices in (x > {x0 + (x1 - x0)*0.8})', 'cell'),
        'OL': (f'vertices in (x < {x2 + delta})', 'cell'),
        'OR': (f'vertices in (x > {x2 - delta})', 'cell'),
        'Middle': ('r.OL *s r.OR', 'facet', 'OL'),
        'OL1': (f'vertices in (x < {x0 + delta})', 'facet'),
        'OR1': (f'vertices in (x > {x0 - delta})', 'facet'),
        'Left1': ('r.OL1 *s r.OR1', 'facet', 'OR1'),
        'OL2': (f'vertices in (x < {x1 + delta})', 'facet'),
        'OR2': (f'vertices in (x > {x1 - delta})', 'facet'),
        'Right1': ('r.OL2 *s r.OR2', 'facet', 'OL2'),
    }

    regions.update(define_box_regions(3, region_box[0], region_box[1]))

    per_tab = {
        'x': ['Left', 'Right'],
        'y': ['Far', 'Near'],
        'z': ['Bottom', 'Top'],
    }

    epbcs = {}
    for k in periodic_mode:
        epbcs['periodic_' + k] = (per_tab[k], {'u.all': 'u.all', 'p.0': 'p.0'},
                                  'match_%s_plane' % k)

    ebcs = bcs.copy()

    solvers = {
        'ls': ('ls.mumps', {}),
        'newton': ('nls.newton',
                   {'i_max': 10,
                    'eps_a': 1e-4,
                    'eps_r': 1e-3,
                    'problem': 'nonlinear',
                    }),
    }

    tstep = TimeStepper(0.0, t_end, n_step=t_nstep)
    idt = 1./tstep.dt

    equations = {
        'balance_of_forces': """
            dw_lin_elastic.i2.Omega(hom.bA, v, u)
          - dw_biot.i2.Omega(hom.bB, v, p)
            =
          - dw_lin_elastic.i2.Omega(hom.tA, v, U)
          + dw_biot.i2.Omega(hom.tB, v, P)
          - dw_lin_prestress.i2.Omega(hom.Vf, v)""",
        'mass_conservation': """
     - %e * dw_biot.i2.Omega(hom.bD, u, q)
     - %e * dw_volume_dot.i2.Omega(hom.bM, q, p)
          - dw_diffusion.i2.Omega(hom.bKeta, q, p)
          - dw_diffusion_coupling.i2.Omega(hom.bQeta, q, p)
          - dw_piezo_coupling.i2.Omega(hom.bGeta, u, q)
            =
     + %e * dw_biot.i2.Omega(hom.tB, U, q)
     - %e * dw_biot.i2.Omega(hom.tB, U0, q)
     + %e * dw_volume_dot.i2.Omega(hom.tM, q, P)
     - %e * dw_volume_dot.i2.Omega(hom.tM, q, P0)
     - %e * dw_volume_integrate.i2.Omega(hom.dZf, q)
          + dw_diffusion.i2.Omega(hom.tKeta, q, P)""" % ((idt,) * 7)
    }

    return locals()
