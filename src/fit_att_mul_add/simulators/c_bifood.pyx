# cython: language_level=3
# distutils: language=c++
"""Behavioral simulators retained from the submitted BIFood model equations.

This is a focused extraction of the two functions used by the 8-model
behavioral pipeline. It preserves their historical signatures, value ordering
(vs=[right,left]), fixation coding (1=left, 0=right), absorbing/reflecting
state dynamics, and gaze modulation equations.
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp, fabs, fmax, log
from libc.stdlib cimport rand, RAND_MAX

ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t INT_t

cdef double _uniform() noexcept nogil:
    return (<double>rand()) / RAND_MAX

cdef double _normal() noexcept nogil:
    cdef double x1, x2, w
    w = 2.0
    while w >= 1.0 or w == 0.0:
        x1 = 2.0 * _uniform() - 1.0
        x2 = 2.0 * _uniform() - 1.0
        w = x1*x1 + x2*x2
    return x1 * sqrt((-2.0 * log(w)) / w)


cpdef object sim_trial_aDDM(
    dict params,
    np.ndarray[DTYPE_t, ndim=1] vs,
    np.ndarray[INT_t, ndim=1] arr_pos,
    np.ndarray[INT_t, ndim=1] arr_du,
    int extend_last=0,
    int repeat=0,
    double dt=0.01,
    int rcd=0,
    double max_rt=1,
):
    cdef list rcd_all_e = []
    cdef list rcd_all_a = []
    cdef list rcd_cslope = []
    cdef int total_t=0, i=0, t_infix=0, f=0, hit=0, done=0
    cdef double a=0, e=0, drift_t_mu=0, drift_t=0, drift=0, cslope=0
    cdef double rt=0
    cdef int choice=0
    cdef double d=float(params['d']), theta=float(params['theta'])
    cdef double gamma=float(params['gamma']), s=float(params['s'])
    cdef double ndt=float(params['ndt']), lam=float(params['lam'])
    cdef double aval=float(params['a']), kval=float(params.get('k',1.0))
    cdef double var_v=float(params.get('var_v',0.0))
    cdef double b0_cslope=float(params.get('b0_cslope',0.0))
    cdef double d_cslope=float(params.get('d_cslope',0.0))
    cdef double std_cslope=float(params.get('std_cslope',0.0))

    arr_pos = arr_pos.copy(); arr_du = arr_du.copy()
    if extend_last > 0:
        arr_du[len(arr_du)-1] += extend_last
    if repeat > 0:
        arr_du = np.tile(arr_du, repeat)
        arr_pos = np.tile(arr_pos, repeat)

    for i in range(len(arr_du)):
        f = arr_pos[i]
        if f == 1:
            drift_t_mu = vs[1] - theta*vs[0] + gamma
        else:
            drift_t_mu = theta*vs[1] - vs[0] - gamma
        drift_t = drift_t_mu*d + var_v*_normal()
        cslope = b0_cslope + d_cslope*fabs(drift_t) + std_cslope*_normal()
        rcd_cslope.append(cslope)

        for t_infix in range(arr_du[i]):
            total_t += 1
            a = aval if lam < 0 else aval*exp(-((total_t*dt/lam)**kval))
            drift = drift_t*dt + s*sqrt(dt)*_normal()
            e += drift
            if rcd == 1:
                rcd_all_e.append(e); rcd_all_a.append(a)
            if fabs(e) > a:
                done=1; hit=1; break
            elif total_t*dt > max_rt:
                done=1; break
        if done:
            break

    rt = total_t*dt + ndt
    choice = 1 if e > 0 else 0
    sim_arr_pos = arr_pos[:i+1].copy()
    sim_arr_time = arr_du[:i+1].copy()
    if len(sim_arr_time): sim_arr_time[-1] = t_infix + 1
    if rcd == 1:
        return ((choice,rt,hit),(np.asarray(rcd_all_e),np.asarray(rcd_all_a),np.asarray(sim_arr_pos),np.asarray(sim_arr_time)),rcd_cslope)
    return ((choice,rt,hit),None)


cpdef object sim_trial_aRACE(
    dict params,
    np.ndarray[DTYPE_t, ndim=1] vs,
    np.ndarray[INT_t, ndim=1] arr_pos,
    np.ndarray[INT_t, ndim=1] arr_du,
    int extend_last=0,
    int repeat=0,
    double dt=0.01,
    int rcd=0,
    double max_rt=1,
    int rb=1,
    int abslope=0,
):
    cdef list rcd_all_e1=[], rcd_all_e0=[], rcd_all_a=[], rcd_cslope=[]
    cdef int total_t=0, i=0, t_infix=0, f=0, hit=0, done=0
    cdef double a=0, e1=0, e0=0, drift1=0, drift0=0
    cdef double m1=0, m0=0, t1=0, t0=0, cslope=0, rt=0
    cdef int choice=0
    cdef double d=float(params['d']), theta=float(params['theta'])
    cdef double gamma=float(params['gamma']), b=float(params.get('b',0.0))
    cdef double s=float(params['s']), ndt=float(params['ndt'])
    cdef double lam=float(params['lam']), aval=float(params['a'])
    cdef double kval=float(params.get('k',1.0)), var_v=float(params.get('var_v',0.0))
    cdef double b0=float(params.get('b0_cslope',0.0)), ds=float(params.get('d_cslope',0.0))
    cdef double ss=float(params.get('std_cslope',0.0)), ov=float(params.get('ov_cslope',0.0))

    arr_pos = arr_pos.copy(); arr_du = arr_du.copy()
    if extend_last > 0: arr_du[len(arr_du)-1] += extend_last
    if repeat > 0:
        arr_du=np.tile(arr_du,repeat); arr_pos=np.tile(arr_pos,repeat)

    for i in range(len(arr_du)):
        f=arr_pos[i]
        if f == 1:
            m1 = vs[1] - b*theta*vs[0] + gamma
            m0 = theta*vs[0] - b*vs[1]
        else:
            m1 = theta*vs[1] - b*vs[0]
            m0 = vs[0] - b*theta*vs[1] + gamma
        t1=m1*d + var_v*_normal(); t0=m0*d + var_v*_normal()
        if abslope == 1:
            cslope=ds*fabs(t1-t0)+ss*_normal()
        elif f == 1:
            cslope=b0+ds*(t1-ov*t0)+ss*_normal()
        else:
            cslope=b0+ds*(t0-ov*t1)+ss*_normal()
        rcd_cslope.append(cslope)

        for t_infix in range(arr_du[i]):
            total_t += 1
            a = aval if lam < 0 else aval*exp(-((total_t*dt/lam)**kval))
            drift1=t1*dt+s*sqrt(dt)*_normal(); drift0=t0*dt+s*sqrt(dt)*_normal()
            e1 += drift1; e0 += drift0
            if rb == 1:
                e1=fmax(0.0,e1); e0=fmax(0.0,e0)
            if rcd == 1:
                rcd_all_e1.append(e1); rcd_all_e0.append(e0); rcd_all_a.append(a)
            if e1 > a or e0 > a:
                done=1; hit=1; break
            elif total_t*dt > max_rt:
                done=1; break
        if done: break

    rt=total_t*dt+ndt
    choice=1 if e1 > e0 else 0
    sim_arr_pos=arr_pos[:i+1].copy(); sim_arr_time=arr_du[:i+1].copy()
    if len(sim_arr_time): sim_arr_time[-1]=t_infix+1
    if rcd == 1:
        return ((choice,rt,hit),((np.asarray(rcd_all_e1),np.asarray(rcd_all_e0)),np.asarray(rcd_all_a),np.asarray(sim_arr_pos),np.asarray(sim_arr_time)),rcd_cslope)
    return ((choice,rt,hit),None)
