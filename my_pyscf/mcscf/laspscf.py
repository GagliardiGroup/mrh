from pyscf import gto, scf, lib
from mrh.my_pyscf.mcscf import laspscf_sync, _DFLASCI
from mrh.my_pyscf.df.sparse_df import sparsedf_array
from mrh.my_pyscf.mcscf import lasci
from scipy import linalg
import numpy as np

# TODO: clean up
density_fit = lasci.density_fit
def LASPSCF (mf_or_mol, ncas_sub, nelecas_sub, **kwargs):
    if isinstance(mf_or_mol, gto.Mole):
        mf = scf.RHF(mf_or_mol)
    else:
        mf = mf_or_mol
    if mf.mol.symmetry: 
        las = LASPSCFSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    else:
        las = LASPSCFNoSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    if getattr (mf, 'with_df', None):
        las = density_fit (las, with_df = mf.with_df) 
    return las

def get_grad (las, mo_coeff=None, ci=None, ugg=None, h1eff_sub=None, h2eff_sub=None,
              veff=None, dm1s=None):
    '''Return energy gradient for orbital rotation and CI relaxation.

    Args:
        las : instance of :class:`LASPSCFNoSymm`

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains molecular orbitals
        ci : list (length=nfrags) of list (length=nroots) of ndarray
            Contains CI vectors
        ugg : instance of :class:`LASPSCF_UnitaryGroupGenerators`
        h1eff_sub : list (length=nfrags) of list (length=nroots) of ndarray
            Contains effective one-electron Hamiltonians experienced by each fragment
            in each state
        h2eff_sub : ndarray of shape (nmo,ncas**2*(ncas+1)/2)
            Contains ERIs (p1a1|a2a3), lower-triangular in the a2a3 indices
        veff : ndarray of shape (2,nao,nao)
            Spin-separated, state-averaged 1-electron mean-field potential in the AO basis
        dm1s : ndarray of shape (2,nao,nao)
            Spin-separated, state-averaged 1-RDM in the AO basis

    Returns:
        gorb : ndarray of shape (ugg.nvar_orb,)
            Orbital rotation gradients as a flat array
        gci : ndarray of shape (sum(ugg.ncsf_sub),)
            CI relaxation gradients as a flat array
        gx : ndarray
            Orbital rotation gradients for temporarily frozen orbitals in the "LASPSCF" problem
    '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci is None: ci = las.ci
    if ugg is None: ugg = las.get_ugg (mo_coeff, ci)
    if dm1s is None: dm1s = las.make_rdm1s (mo_coeff=mo_coeff, ci=ci)
    if h2eff_sub is None: h2eff_sub = las.get_h2eff (mo_coeff)
    if veff is None:
        veff = las.get_veff (dm = dm1s)
    if h1eff_sub is None: h1eff_sub = las.get_h1eff (mo_coeff, ci=ci, veff=veff,
                                                     h2eff_sub=h2eff_sub)

    if callable (getattr (las, 'get_grad_orb', None)):
        gorb = las.get_grad_orb (mo_coeff=mo_coeff, ci=ci, h2eff_sub=h2eff_sub, veff=veff, dm1s=dm1s)
    else:
        gorb = get_grad_orb (las, mo_coeff=mo_coeff, ci=ci, h2eff_sub=h2eff_sub, veff=veff, dm1s=dm1s)
    if callable (getattr (las, 'get_grad_ci', None)):
        gci = las.get_grad_ci (mo_coeff=mo_coeff, ci=ci, h1eff_sub=h1eff_sub, h2eff_sub=h2eff_sub,
                               veff=veff)
    else:
        gci = get_grad_ci (las, mo_coeff=mo_coeff, ci=ci, h1eff_sub=h1eff_sub, h2eff_sub=h2eff_sub,
                           veff=veff)

    idx = ugg.get_gx_idx ()
    gx = gorb[idx]
    gint = ugg.pack (gorb, gci)
    gorb = gint[:ugg.nvar_orb]
    gci = gint[ugg.nvar_orb:]
    return gorb, gci, gx.ravel ()

def get_grad_orb (las, mo_coeff=None, ci=None, h2eff_sub=None, veff=None, dm1s=None, hermi=-1):
    '''Return energy gradient for orbital rotation.

    Args:
        las : instance of :class:`LASPSCFNoSymm`

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains molecular orbitals
        ci : list (length=nfrags) of list (length=nroots) of ndarray
            Contains CI vectors
        h2eff_sub : ndarray of shape (nmo,ncas**2*(ncas+1)/2)
            Contains ERIs (p1a1|a2a3), lower-triangular in the a2a3 indices
        veff : ndarray of shape (2,nao,nao)
            Spin-separated, state-averaged 1-electron mean-field potential in the AO basis
        dm1s : ndarray of shape (2,nao,nao)
            Spin-separated, state-averaged 1-RDM in the AO basis
        hermi : integer
            Control (anti-)symmetrization. 0 means to return the effective Fock matrix,
            F1 = h.D + g.d. -1 means to return the true orbital-rotation gradient, which is skew-
            symmetric: gorb = F1 - F1.T. +1 means to return the symmetrized effective Fock matrix,
            (F1 + F1.T) / 2. The factor of 2 difference between hermi=-1 and the other two options
            is intentional and necessary.

    Returns:
        gorb : ndarray of shape (nmo,nmo)
            Orbital rotation gradients as a square antihermitian array
    '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci is None: ci = las.ci
    if dm1s is None: dm1s = las.make_rdm1s (mo_coeff=mo_coeff, ci=ci)
    if h2eff_sub is None: h2eff_sub = las.get_h2eff (mo_coeff)
    if veff is None:
        veff = las.get_veff (dm = dm1s)
    nao, nmo = mo_coeff.shape
    ncore = las.ncore
    ncas = las.ncas
    nocc = las.ncore + las.ncas
    smo_cas = las._scf.get_ovlp () @ mo_coeff[:,ncore:nocc]
    smoH_cas = smo_cas.conj ().T

    # The orbrot part
    h1s = las.get_hcore ()[None,:,:] + veff
    f1 = h1s[0] @ dm1s[0] + h1s[1] @ dm1s[1]
    f1 = mo_coeff.conjugate ().T @ f1 @ las._scf.get_ovlp () @ mo_coeff
    # ^ I need the ovlp there to get dm1s back into its correct basis
    casdm2 = las.make_casdm2 (ci=ci)
    casdm1s = np.stack ([smoH_cas @ d @ smo_cas for d in dm1s], axis=0)
    casdm1 = casdm1s.sum (0)
    casdm2 -= np.multiply.outer (casdm1, casdm1)
    casdm2 += np.multiply.outer (casdm1s[0], casdm1s[0]).transpose (0,3,2,1)
    casdm2 += np.multiply.outer (casdm1s[1], casdm1s[1]).transpose (0,3,2,1)
    eri = h2eff_sub.reshape (nmo*ncas, ncas*(ncas+1)//2)
    eri = lib.numpy_helper.unpack_tril (eri).reshape (nmo, ncas, ncas, ncas)
    f1[:,ncore:nocc] += np.tensordot (eri, casdm2, axes=((1,2,3),(1,2,3)))

    if hermi == -1:
        return f1 - f1.T
    elif hermi == 1:
        return .5*(f1+f1.T)
    elif hermi == 0:
        return f1
    else:
        raise ValueError ("kwarg 'hermi' must = -1, 0, or +1")

def get_grad_ci (las, mo_coeff=None, ci=None, h1eff_sub=None, h2eff_sub=None, veff=None):
    '''Return energy gradient for CI relaxation.

    Args:
        las : instance of :class:`LASPSCFNoSymm`

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains molecular orbitals
        ci : list (length=nfrags) of list (length=nroots) of ndarray
            Contains CI vectors
        h1eff_sub : list (length=nfrags) of list (length=nroots) of ndarray
            Contains effective one-electron Hamiltonians experienced by each fragment
            in each state
        h2eff_sub : ndarray of shape (nmo,ncas**2*(ncas+1)/2)
            Contains ERIs (p1a1|a2a3), lower-triangular in the a2a3 indices
        veff : ndarray of shape (2,nao,nao)
            Spin-separated, state-averaged 1-electron mean-field potential in the AO basis

    Returns:
        gci : list (length=nfrags) of list (length=nroots) of ndarray
            CI relaxation gradients in the shape of CI vectors
    '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci is None: ci = las.ci
    if h2eff_sub is None: h2eff_sub = las.get_h2eff (mo_coeff)
    if h1eff_sub is None: h1eff_sub = las.get_h1eff (mo_coeff, ci=ci, veff=veff,
                                                     h2eff_sub=h2eff_sub)
    gci = []
    for isub, (fcibox, h1e, ci0, ncas, nelecas) in enumerate (zip (
            las.fciboxes, h1eff_sub, ci, las.ncas_sub, las.nelecas_sub)):
        eri_cas = las.get_h2eff_slice (h2eff_sub, isub, compact=8)
        linkstrl = fcibox.states_gen_linkstr (ncas, nelecas, True)
        linkstr  = fcibox.states_gen_linkstr (ncas, nelecas, False)
        h2eff = fcibox.states_absorb_h1e(h1e, eri_cas, ncas, nelecas, .5)
        hc0 = fcibox.states_contract_2e(h2eff, ci0, ncas, nelecas, link_index=linkstrl)
        hc0 = [hc.ravel () for hc in hc0]
        ci0 = [c.ravel () for c in ci0]
        gci.append ([2.0 * (hc - c * (c.dot (hc))) for c, hc in zip (ci0, hc0)])
    return gci

class LASPSCFNoSymm (lasci.LASCINoSymm):

    get_grad = get_grad
    get_grad_orb = get_grad_orb
    get_grad_ci = get_grad_ci
    _hop = laspscf_sync.LASPSCF_HessianOperator
    _kern = laspscf_sync.kernel
    def get_hop (self, mo_coeff=None, ci=None, ugg=None, **kwargs):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if ugg is None: ugg = self.get_ugg ()
        return self._hop (self, ugg, mo_coeff=mo_coeff, ci=ci, **kwargs)

    def kernel(self, mo_coeff=None, ci0=None, casdm0_fr=None, conv_tol_grad=None,
            assert_no_dupes=False, verbose=None, _kern=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if ci0 is None: ci0 = self.ci
        if verbose is None: verbose = self.verbose
        if conv_tol_grad is None: conv_tol_grad = self.conv_tol_grad
        if _kern is None: _kern = self._kern
        log = lib.logger.new_logger(self, verbose)

        if self.verbose >= lib.logger.WARN:
            self.check_sanity()
        self.dump_flags(log)

        for fcibox in self.fciboxes:
            fcibox.verbose = self.verbose
            fcibox.stdout = self.stdout
            fcibox.nroots = self.nroots
            fcibox.weights = self.weights
        # TODO: local excitations and locally-impure states in LASSCF kernel
        do_warn=False
        if ci0 is not None:
            for i, ci0_i in enumerate (ci0):
                if ci0_i is None: continue
                for j, ci0_ij in enumerate (ci0_i):
                    if ci0_ij is None: continue
                    if np.asarray (ci0_ij).ndim>2:
                        do_warn=True
                        ci0_i[j] = ci0_ij[0]
        if do_warn: log.warn ("Discarding all but the first root of guess CI vectors!")

        self.converged, self.e_tot, self.e_states, self.mo_energy, self.mo_coeff, self.e_cas, \
                self.ci, h2eff_sub, veff = _kern(mo_coeff=mo_coeff, ci0=ci0, verbose=verbose, \
                casdm0_fr=casdm0_fr, conv_tol_grad=conv_tol_grad, assert_no_dupes=assert_no_dupes)

        self._finalize ()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy, h2eff_sub, veff

    _ugg = laspscf_sync.LASPSCF_UnitaryGroupGenerators
    def get_ugg (self, mo_coeff=None, ci=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        return self._ugg (self, mo_coeff, ci)

    def fast_veffa (self, casdm1s_sub, bmPu, mo_coeff=None, ci=None):
        '''Compute the effective potential exerted by active electrons on the whole orbital space
        using integrals and density matrices stored in the MO basis. This only makes sense to
        do if density fitting is used and is not implemented with GPUs at present.

        Args:
            casdm1s_sub : list of ndarray of shape (2,nlas,nlas)
            bmPu : ndarray of shape (nao,naux,ncas) or None
                Cholesky vectors with one AO index transformed into active orbitals

        Kwargs:
            mo_coeff : ndarray of shape (nao,nmo)
            ci : nested list of ndarrays

        Returns:
            veff : ndarray of shape (2,nao,nao)
        '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        ncore = self.ncore
        ncas_sub = self.ncas_sub
        ncas = sum (ncas_sub)
        nocc = ncore + ncas
        nao, nmo = mo_coeff.shape
        gpu=self.use_gpu
        mo_cas = mo_coeff[:,ncore:nocc]
        moH_cas = mo_cas.conjugate ().T
        moH_coeff = mo_coeff.conjugate ().T
        dma = linalg.block_diag (*[dm[0] for dm in casdm1s_sub])
        dmb = linalg.block_diag (*[dm[1] for dm in casdm1s_sub])
        casdm1s = np.stack ([dma, dmb], axis=0)
        if (bmPu is None) or gpu or not (isinstance (self, _DFLASCI)):
            dm1s = np.dot (mo_cas, np.dot (casdm1s, moH_cas)).transpose (1,0,2)
            return self.get_veff (dm = dm1s)
        casdm1 = casdm1s.sum (0)
        dm1 = np.dot (mo_cas, np.dot (casdm1, moH_cas))
        bPmn = sparsedf_array (self.with_df._cderi)

        # vj
        dm_tril = dm1 + dm1.T - np.diag (np.diag (dm1.T))
        rho = np.dot (bPmn, lib.pack_tril (dm_tril))
        vj = lib.unpack_tril (np.dot (rho, bPmn))

        # vk
        vmPsu = np.dot (bmPu, casdm1s)
        vk = np.tensordot (vmPsu, bmPu, axes=((1,3),(1,2))).transpose (1,0,2)
        return vj[None,:,:] - vk

    lasci = lasci_ = lasci.LASCINoSymm.kernel

    def dump_flags (self, verbose=None, _method_name='LASPSCF'):
        super().dump_flags (verbose=verbose, _method_name=_method_name)

    def check_sanity (self):
        super().check_sanity ()
        self.get_ugg () # constructor encounters impossible states and raises error

class LASPSCFSymm (lasci.LASCISymm, LASPSCFNoSymm):

    get_veff = LASPSCFNoSymm.get_veff
    _ugg = laspscf_sync.LASPSCFSymm_UnitaryGroupGenerators

    def kernel(self, mo_coeff=None, ci0=None, casdm0_fr=None, verbose=None, assert_no_dupes=False):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ci0 is None:
            ci0 = self.ci

        # Initialize/overwrite mo_coeff.orbsym. Don't pass ci0 because it's not the right shape
        lib.logger.info (self, ("LASPSCF lazy hack note: lines below reflect the point-group "
                                "symmetry of the whole molecule but not of the individual "
                                "subspaces"))
        mo_coeff = self.mo_coeff = self.label_symmetry_(mo_coeff)
        return LASPSCFNoSymm.kernel(self, mo_coeff=mo_coeff, ci0=ci0,
            casdm0_fr=casdm0_fr, verbose=verbose, assert_no_dupes=assert_no_dupes)

 
