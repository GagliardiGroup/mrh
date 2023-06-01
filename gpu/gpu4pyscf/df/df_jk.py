#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import copy

import ctypes
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import scf
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from gpu4pyscf.lib.utils import patch_cpu_kernel

import libgpu

DEBUG = False

def get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13):
    print(" -- -- Inside mrh/gpu/gpu4pyscf/df/df_jk.py::get_jk() w/ use_gpu= ", dfobj.mol.use_gpu, " with_jk= ", with_j, with_k)
    gpu = dfobj.mol.use_gpu
    #libgpu.libgpu_dev_properties(gpu, 1)
    
    assert (with_j or with_k)
    if (not with_k and not dfobj.mol.incore_anyway and
        # 3-center integral tensor is not initialized
        dfobj._cderi is None):
        return get_j(dfobj, dm, hermi, direct_scf_tol), None

    t0 = t1 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(dfobj.stdout, dfobj.verbose)
    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
    null = lib.c_null_ptr()

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]
    vj = 0
    vk = numpy.zeros_like(dms)

    if with_j:
        idx = numpy.arange(nao)
        dmtril = lib.pack_tril(dms + dms.conj().transpose(0,2,1))
        dmtril[:,idx*(idx+1)//2+idx] *= .5

    if not with_k:
        for eri1 in dfobj.loop():
            rho = numpy.einsum('ix,px->ip', dmtril, eri1)
            vj += numpy.einsum('ip,px->ix', rho, eri1)

    elif getattr(dm, 'mo_coeff', None) is not None:
        #TODO: test whether dm.mo_coeff matching dm
        mo_coeff = numpy.asarray(dm.mo_coeff, order='F')
        mo_occ   = numpy.asarray(dm.mo_occ)
        nmo = mo_occ.shape[-1]
        mo_coeff = mo_coeff.reshape(-1,nao,nmo)
        mo_occ   = mo_occ.reshape(-1,nmo)
        if mo_occ.shape[0] * 2 == nset: # handle ROHF DM
            mo_coeff = numpy.vstack((mo_coeff, mo_coeff))
            mo_occa = numpy.array(mo_occ> 0, dtype=numpy.double)
            mo_occb = numpy.array(mo_occ==2, dtype=numpy.double)
            assert (mo_occa.sum() + mo_occb.sum() == mo_occ.sum())
            mo_occ = numpy.vstack((mo_occa, mo_occb))

        orbo = []
        for k in range(nset):
            c = numpy.einsum('pi,i->pi', mo_coeff[k][:,mo_occ[k]>0],
                             numpy.sqrt(mo_occ[k][mo_occ[k]>0]))
            orbo.append(numpy.asarray(c, order='F'))

        max_memory = dfobj.max_memory - lib.current_memory()[0]
        blksize = max(4, int(min(dfobj.blockdim, max_memory*.3e6/8/nao**2)))
        buf = numpy.empty((blksize*nao,nao))
        for eri1 in dfobj.loop(blksize):
            naux, nao_pair = eri1.shape
            assert (nao_pair == nao*(nao+1)//2)
            if with_j:
                rho = numpy.einsum('ix,px->ip', dmtril, eri1)
                vj += numpy.einsum('ip,px->ix', rho, eri1)

            for k in range(nset):
                nocc = orbo[k].shape[1]
                if nocc > 0:
                    buf1 = buf[:naux*nocc]
                    fdrv(ftrans, fmmm,
                         buf1.ctypes.data_as(ctypes.c_void_p),
                         eri1.ctypes.data_as(ctypes.c_void_p),
                         orbo[k].ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(naux), ctypes.c_int(nao),
                         (ctypes.c_int*4)(0, nocc, 0, nao),
                         null, ctypes.c_int(0))
                    vk[k] += lib.dot(buf1.T, buf1)
            t1 = log.timer_debug1('jk', *t1)
    else:
        print(" -- -- Inside else branch inside mrh/gpu/gpu4pyscf/df/df_jk.py::get_jk()")
        #:vk = numpy.einsum('pij,jk->pki', cderi, dm)
        #:vk = numpy.einsum('pki,pkj->ij', cderi, vk)
        rargs = (ctypes.c_int(nao), (ctypes.c_int*4)(0, nao, 0, nao),
                 null, ctypes.c_int(0))
        dms = [numpy.asarray(x, order='F') for x in dms]
        max_memory = dfobj.max_memory - lib.current_memory()[0]
        blksize = max(4, int(min(dfobj.blockdim, max_memory*.22e6/8/nao**2)))
        buf = numpy.empty((2,blksize,nao,nao))
        
        for eri1 in dfobj.loop(blksize):
            naux, nao_pair = eri1.shape            
            print("naux= ", naux, "  nao_pair= ",nao_pair, "  nset= ", nset)
            
            print("dmtril.shape()= ", dmtril.shape)
            print("dmtril[0][0:4]= ", dmtril[0][0:4])
            print("eri1.shape()= ", eri1.shape)
            print("eri1[0][0:4]= ", eri1[0][0:4])
            print("eri1[1][0:4]= ", eri1[1][0:4])

            
            libgpu.libgpu_compute_df_get_jk(gpu, eri1)
            
            if with_j:
                rho = numpy.einsum('ix,px->ip', dmtril, eri1)
                vj += numpy.einsum('ip,px->ix', rho, eri1)

            for k in range(nset):
                print("k= ", k)
                buf1 = buf[0,:naux]
                fdrv(ftrans, fmmm,
                     buf1.ctypes.data_as(ctypes.c_void_p),
                     eri1.ctypes.data_as(ctypes.c_void_p),
                     dms[k].ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(naux), *rargs)

                buf2 = lib.unpack_tril(eri1, out=buf[1])
                vk[k] += lib.dot(buf1.reshape(-1,nao).T, buf2.reshape(-1,nao))
            t1 = log.timer_debug1('jk', *t1)

    if with_j: vj = lib.unpack_tril(vj, 1).reshape(dm_shape)
    if with_k: vk = vk.reshape(dm_shape)
    logger.timer(dfobj, 'df vj and vk', *t0)
#    quit()
    return vj, vk

def get_j(dfobj, dm, hermi=1, direct_scf_tol=1e-13):
    from pyscf.scf import _vhf
    from pyscf.scf import jk
    from pyscf.df import addons
    t0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = dfobj.mol
    if dfobj._vjopt is None:
        dfobj.auxmol = auxmol = addons.make_auxmol(mol, dfobj.auxbasis)
        opt = _vhf.VHFOpt(mol, 'int3c2e', 'CVHFnr3c2e_schwarz_cond')
        opt.direct_scf_tol = direct_scf_tol

        # q_cond part 1: the regular int2e (ij|ij) for mol's basis
        opt.init_cvhf_direct(mol, 'int2e', 'CVHFsetnr_direct_scf')
        mol_q_cond = lib.frompointer(opt._this.contents.q_cond, mol.nbas**2)

        # Update q_cond to include the 2e-integrals (auxmol|auxmol)
        j2c = auxmol.intor('int2c2e', hermi=1)
        j2c_diag = numpy.sqrt(abs(j2c.diagonal()))
        aux_loc = auxmol.ao_loc
        aux_q_cond = [j2c_diag[i0:i1].max()
                      for i0, i1 in zip(aux_loc[:-1], aux_loc[1:])]
        q_cond = numpy.hstack((mol_q_cond, aux_q_cond))
        fsetqcond = _vhf.libcvhf.CVHFset_q_cond
        fsetqcond(opt._this, q_cond.ctypes.data_as(ctypes.c_void_p),
                  ctypes.c_int(q_cond.size))

        try:
            opt.j2c = j2c = scipy.linalg.cho_factor(j2c, lower=True)
            opt.j2c_type = 'cd'
        except scipy.linalg.LinAlgError:
            opt.j2c = j2c
            opt.j2c_type = 'regular'

        # jk.get_jk function supports 4-index integrals. Use bas_placeholder
        # (l=0, nctr=1, 1 function) to hold the last index.
        bas_placeholder = numpy.array([0, 0, 1, 1, 0, 0, 0, 0],
                                      dtype=numpy.int32)
        fakemol = mol + auxmol
        fakemol._bas = numpy.vstack((fakemol._bas, bas_placeholder))
        opt.fakemol = fakemol
        dfobj._vjopt = opt
        t1 = logger.timer_debug1(dfobj, 'df-vj init_direct_scf', *t1)

    opt = dfobj._vjopt
    fakemol = opt.fakemol
    dm = numpy.asarray(dm, order='C')
    dm_shape = dm.shape
    nao = dm_shape[-1]
    dm = dm.reshape(-1,nao,nao)
    n_dm = dm.shape[0]

    # First compute the density in auxiliary basis
    # j3c = fauxe2(mol, auxmol)
    # jaux = numpy.einsum('ijk,ji->k', j3c, dm)
    # rho = numpy.linalg.solve(auxmol.intor('int2c2e'), jaux)
    nbas = mol.nbas
    nbas1 = mol.nbas + dfobj.auxmol.nbas
    shls_slice = (0, nbas, 0, nbas, nbas, nbas1, nbas1, nbas1+1)
    with lib.temporary_env(opt, prescreen='CVHFnr3c2e_vj_pass1_prescreen',
                           _dmcondname='CVHFsetnr_direct_scf_dm'):
        jaux = jk.get_jk(fakemol, dm, ['ijkl,ji->kl']*n_dm, 'int3c2e',
                         aosym='s2ij', hermi=0, shls_slice=shls_slice,
                         vhfopt=opt)
    # remove the index corresponding to bas_placeholder
    jaux = numpy.array(jaux)[:,:,0]
    t1 = logger.timer_debug1(dfobj, 'df-vj pass 1', *t1)

    if opt.j2c_type == 'cd':
        rho = scipy.linalg.cho_solve(opt.j2c, jaux.T)
    else:
        rho = scipy.linalg.solve(opt.j2c, jaux.T)
    # transform rho to shape (:,1,naux), to adapt to 3c2e integrals (ij|k)
    rho = rho.T[:,numpy.newaxis,:]
    t1 = logger.timer_debug1(dfobj, 'df-vj solve ', *t1)

    # Next compute the Coulomb matrix
    # j3c = fauxe2(mol, auxmol)
    # vj = numpy.einsum('ijk,k->ij', j3c, rho)
    with lib.temporary_env(opt, prescreen='CVHFnr3c2e_vj_pass2_prescreen',
                           _dmcondname=None):
        # CVHFnr3c2e_vj_pass2_prescreen requires custom dm_cond
        aux_loc = dfobj.auxmol.ao_loc
        dm_cond = [abs(rho[:,:,i0:i1]).max()
                   for i0, i1 in zip(aux_loc[:-1], aux_loc[1:])]
        dm_cond = numpy.array(dm_cond)
        fsetcond = _vhf.libcvhf.CVHFset_dm_cond
        fsetcond(opt._this, dm_cond.ctypes.data_as(ctypes.c_void_p),
                  ctypes.c_int(dm_cond.size))

        vj = jk.get_jk(fakemol, rho, ['ijkl,lk->ij']*n_dm, 'int3c2e',
                       aosym='s2ij', hermi=1, shls_slice=shls_slice,
                       vhfopt=opt)

    t1 = logger.timer_debug1(dfobj, 'df-vj pass 2', *t1)
    logger.timer(dfobj, 'df-vj', *t0)
    return numpy.asarray(vj).reshape(dm_shape)

def _get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13):
    
    print(" -- -- Inside mrh/gpu/gpu4pyscf/df/df_jk.py::_get_jk()")
    vj, vk = get_jk(dfobj, dm, hermi, with_j, with_k, direct_scf_tol)

    return vj, vk
