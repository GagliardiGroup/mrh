import numpy as np
import time
from copy import deepcopy
from scipy import linalg
from pyscf import gto, dft, ao2mo, fci, mcscf, lib, __config__
from pyscf.lib import logger, temporary_env
from pyscf.fci import cistring
from pyscf.dft import gen_grid
from pyscf.mcscf import mc_ao2mo, mc1step, casci
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.mcscf.addons import StateAverageMCSCFSolver, state_average_mix
from pyscf.mcscf.addons import state_average_mix_, StateAverageMixFCISolver
from pyscf.mcscf.addons import StateAverageFCISolver
from mrh.my_pyscf.mcpdft import pdft_veff, ci_scf
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density
from mrh.my_pyscf.mcpdft.otfnal import otfnal, transfnal, get_transfnal
from mrh.util.rdm import get_2CDM_from_2RDM, get_2CDMs_from_2RDMs

# TODO: 
# 1. Clean up "make_rdms_mcpdft":
#       a. Make better use of existing API and name conventions
#       b. Get rid of pointless _os, _ss computation unless necessary. (Tags?)
# 2. Unify calling signatures of the "energy_*" functions: mo_coeff, ci, ot,
#    state.
# 3. Hybrid API and unittests. NotImplementedErrors for omega and alpha.

def energy_tot (mc, ot=None, mo_coeff=None, ci=None, state=0, verbose=None):
    ''' Calculate MC-PDFT total energy

        Args:
            mc : an instance of CASSCF or CASCI class
                Note: this function does not currently run the CASSCF or
                CASCI calculation itself prior to calculating the
                MC-PDFT energy. Call mc.kernel () before passing to this
                function!

        Kwargs:
            ot : an instance of on-top functional class - see otfnal.py
            mo_coeff : ndarray of shape (nao, nmo)
                Molecular orbital coefficients
            ci : ndarray or list
                CI vector or vectors. Must be consistent with the nroots
                of mc.
            state : int
                If mc describes a state-averaged calculation, select the
                state (0-indexed).
            verbose : int
                Verbosity of logger output; defaults to mc.verbose

        Returns:
            e_tot : float
                Total MC-PDFT energy including nuclear repulsion energy
            E_ot : float
                On-top (cf. exchange-correlation) energy
    '''
    if ot is None: ot = mc.otfnal
    ot.reset (mol=mc.mol) # scanner mode safety
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if verbose is None: verbose = mc.verbose
    t0 = (logger.process_clock (), logger.perf_counter ())

    # Allow MC-PDFT to be subclassed, and also allow this function to be
    # called without mc being an instance of MC-PDFT class

    if callable (getattr (mc, 'make_rdms_mcpdft', None)):
        dm_list = mc.make_rdms_mcpdft (ot=ot, mo_coeff=mo_coeff, ci=ci,
            state=state)
    else:
        dm_list = make_rdms_mcpdft (mc, ot=ot, mo_coeff=mo_coeff, ci=ci,
            state=state)
    t0 = logger.timer (ot, 'rdms', *t0)


    if callable (getattr (mc, 'energy_mcwfn', None)):
        e_mcwfn = mc.energy_mcwfn (ot=ot, mo_coeff=mo_coeff, dm_list=dm_list,
            verbose=verbose)
    else:
        e_mcwfn = energy_mcwfn (mc, ot=ot, mo_coeff=mo_coeff, dm_list=dm_list,
            verbose=verbose)
    t0 = logger.timer (ot, 'MC wfn energy', *t0)


    if callable (getattr (mc, 'energy_dft', None)):
        e_dft = mc.energy_dft (ot=ot, dm_list=dm_list, mo_coeff=mo_coeff)
    else:
        e_dft = energy_dft (mc, ot=ot, dm_list=dm_list, mo_coeff=mo_coeff)
    t0 = logger.timer (ot, 'E_ot', *t0)

    e_tot = e_mcwfn + e_dft
    return e_tot, e_dft

# Consistency with PySCF convention
kernel = energy_tot # backwards compatibility
def energy_elec (mc, *args, **kwargs):
    e_tot, E_ot = energy_tot (mc, *args, **kwargs)
    e_elec = e_tot - mc._scf.energy_nuc ()
    return e_elec, E_ot

def make_rdms_mcpdft (mc, ot=None, mo_coeff=None, ci=None, state=0):
    ''' Build the necessary density matrices for an MC-PDFT calculation 

        Args:
            mc : an instance of CASSCF or CASCI class
                Note: this function does not currently run the CASSCF or
                CASCI calculation itself

        Kwargs:
            ot : an instance of on-top functional class - see otfnal.py
            mo_coeff : ndarray of shape (nao, nmo)
                Molecular orbital coefficients
            ci : ndarray or list
                CI vector or vectors. If a list of many CI vectors, mc
                must be a state-average object with the correct nroots
            state : integer
                Indexes the CI vector. If negative and if mc.fcisolver
                is a state-average object, state-averaged density
                matrices are returned.

        Returns:
            dm1s : ndarray of shape (2,nao,nao)
                Spin-separated 1-RDM
            adm : (adm1s, adm2s)
                adm1s : ndarray of shape (2,ncas,ncas)
                    Spin-separated 1-RDM for the active orbitals
                adm2s : 3 ndarrays of shape (ncas,ncas,ncas,ncas)
                    First ndarray is spin-summed casdm2
                    Second ndarray is casdm2_aa + casdm2_bb
                    Third ndarray is casdm2_ab
    '''
    if ci is None: ci = mc.ci
    if ot is None: ot = mc.otfnal
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    nocc = ncore + ncas
    nroots = getattr (mc.fcisolver, 'nroots', 1)

    # figure out the correct RDMs to build (SA or SS?)
    _casdms = mc.fcisolver
    if nroots>1:
        ci = ci[state]
        if isinstance (mc.fcisolver, StateAverageMixFCISolver):
            p0 = 0
            _casdms = None
            for s in mc.fcisolver.fcisolvers:
                p1 = p0 + s.nroots
                if p0 <= state and state < p1:
                    _casdms = s
                    nelecas = mc.fcisolver._get_nelec (s, nelecas)
                    break
                p0 = p1
            if _casdms is None:
                raise RuntimeError ("Can't find FCI solver for state", state)
        elif isinstance (mc.fcisolver, StateAverageFCISolver):
            _casdms = fci.solver (mc._scf.mol, singlet=False, symm=False)

    # Make the rdms
    # make_rdm12s returns (a, b), (aa, ab, bb)
    mo_cas = mo_coeff[:,ncore:nocc]
    moH_cas = mo_cas.conj ().T
    mo_core = mo_coeff[:,:ncore]
    moH_core = mo_core.conj ().T
    adm1s = np.stack (_casdms.make_rdm1s (ci, ncas, nelecas), axis=0)
    adm2s = _casdms.make_rdm12s (ci, ncas, nelecas)[1]
    adm2s = get_2CDMs_from_2RDMs (adm2s, adm1s)
    adm2_ss = adm2s[0] + adm2s[2]
    adm2_os = adm2s[1]
    adm2 = adm2_ss + adm2_os + adm2_os.transpose (2,3,0,1)
    dm1s = np.dot (adm1s, moH_cas)
    dm1s = np.dot (mo_cas, dm1s).transpose (1,0,2)
    dm1s += np.dot (mo_core, moH_core)[None,:,:]
    return dm1s, (adm1s, (adm2, adm2_ss, adm2_os))

def energy_mcwfn (mc, ot=None, mo_coeff=None, ci=None, dm_list=None,
        verbose=None):
    ''' Compute the parts of the MC-PDFT energy arising from the wave
        function

        Args:
            mc : an instance of CASSCF or CASCI class
                Note: this function does not currently run the CASSCF or
                CASCI calculation itself prior to calculating the
                MC-PDFT energy. Call mc.kernel () before passing to thiswould
                function!

        Kwargs:
            ot : an instance of on-top functional class - see otfnal.py
            mo_coeff : ndarray of shape (nao, nmo)
                contains molecular orbital coefficients
            ci : list or ndarray
                contains ci vectors
            dm_list : (dm1s, adm2)
                return arguments of make_rdms_mcpdft

        Returns:
            e_mcwfn : float
                Energy from the multiconfigurational wave function:
                nuclear repulsion + 1e + coulomb
    '''

    if ot is None: ot = mc.otfnal
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if verbose is None: verbose = mc.verbose
    if dm_list is None: dm_list = mc.make_rdms_mcpdft (ot=ot,
        mo_coeff=mo_coeff, ci=ci)
    log = logger.new_logger (mc, verbose=verbose)
    ncas, nelecas = mc.ncas, mc.nelecas
    dm1s, (adm1s, (adm2, adm2_ss, adm2_os)) = dm_list

    spin = abs(nelecas[0] - nelecas[1])
    omega, alpha, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, spin=spin)
    if omega or alpha:
        raise NotImplementedError ("range-separated on-top functionals")
    hyb_x, hyb_c = hyb

    Vnn = mc._scf.energy_nuc ()
    h = mc._scf.get_hcore ()
    dm1 = dm1s[0] + dm1s[1]
    if log.verbose >= logger.DEBUG or abs (hyb_x) > 1e-10:
        vj, vk = mc._scf.get_jk (dm=dm1s)
        vj = vj[0] + vj[1]
    else:
        vj = mc._scf.get_j (dm=dm1)
    Te_Vne = np.tensordot (h, dm1)
    # (vj_a + vj_b) * (dm_a + dm_b)
    E_j = np.tensordot (vj, dm1) / 2  
    # (vk_a * dm_a) + (vk_b * dm_b) Mind the difference!
    if log.verbose >= logger.DEBUG or abs (hyb_x) > 1e-10:
        E_x = -(np.tensordot (vk[0], dm1s[0]) + np.tensordot (vk[1], dm1s[1]))
        E_x /= 2.0
    else:
        E_x = 0
    log.debug ('CAS energy decomposition:')
    log.debug ('Vnn = %s', Vnn)
    log.debug ('Te + Vne = %s', Te_Vne)
    log.debug ('E_j = %s', E_j)
    log.debug ('E_x = %s', E_x)
    E_c = 0
    if log.verbose >= logger.DEBUG or abs (hyb_c) > 1e-10:
        # g_pqrs * l_pqrs / 2
        #if log.verbose >= logger.DEBUG:
        aeri = ao2mo.restore (1, mc.get_h2eff (mo_coeff), mc.ncas)
        E_c = np.tensordot (aeri, adm2, axes=4) / 2
        E_c_ss = np.tensordot (aeri, adm2_ss, axes=4) / 2
        E_c_os = np.tensordot (aeri, adm2_os, axes=4) # ab + ba -> factor of 2
        log.info ('E_c = %s', E_c)
        log.info ('E_c (SS) = %s', E_c_ss)
        log.info ('E_c (OS) = %s', E_c_os)
        e_err = E_c_ss + E_c_os - E_c
        assert (abs (e_err) < 1e-8), e_err
    if abs (hyb_x) > 1e-10 or abs (hyb_c) > 1e-10:
        log.debug (('Adding %s * %s CAS exchange, %s * %s CAS correlation to '
                    'E_ot'), hyb_x, E_x, hyb_c, E_c)
    e_mcwfn = Vnn + Te_Vne + E_j + (hyb_x * E_x) + (hyb_c * E_c) 
    return e_mcwfn

def energy_dft (mc, ot=None, mo_coeff=None, dm_list=None, max_memory=None,
        hermi=1):
    ''' Wrap to get_E_ot for subclassing. '''
    if ot is None: ot = mc.otfnal
    if dm_list is None: dm_list = mc.make_rdms_mcpdft ()
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if max_memory is None: max_memory = mc.max_memory
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    mo_cas = mo_coeff[:,ncore:nocc]
    dm1s, (adm1s, (adm2, adm2_ss, adm2_os)) = dm_list
    return get_E_ot (ot, dm1s, adm2, mo_cas, max_memory=max_memory,
        hermi=hermi)

def get_E_ot (ot, oneCDMs, twoCDM_amo, ao2amo, max_memory=2000, hermi=1):
    ''' E_MCPDFT = h_pq l_pq + 1/2 v_pqrs l_pq l_rs + E_ot[rho,Pi] 
        or, in other terms, 
        E_MCPDFT = T_KS[rho] + E_ext[rho] + E_coul[rho] + E_ot[rho, Pi]
                 = E_DFT[1rdm] - E_xc[rho] + E_ot[rho, Pi] 
        Args:
            ot : an instance of otfnal class
            oneCDMs : ndarray of shape (2, nao, nao)
                containing spin-separated one-body density matrices
            twoCDM_amo : ndarray of shape (ncas, ncas, ncas, ncas)
                containing spin-summed two-body cumulant density matrix
                in an active space
            ao2amo : ndarray of shape (nao, ncas)
                containing molecular orbital coefficients for
                active-space orbitals

        Kwargs:
            max_memory : int or float
                maximum cache size in MB
                default is 2000
            hermi : int
                1 if 1CDMs are assumed hermitian, 0 otherwise

        Returns : float
            The MC-PDFT on-top exchange-correlation energy

    '''
    ni, xctype, dens_deriv = ot._numint, ot.xctype, ot.dens_deriv
    norbs_ao = ao2amo.shape[0]

    E_ot = 0.0

    t0 = (logger.process_clock (), logger.perf_counter ())
    make_rho = tuple (ni._gen_rho_evaluator (ot.mol, oneCDMs[i,:,:], hermi) for
        i in range(2))
    for ao, mask, weight, coords in ni.block_loop (ot.mol, ot.grids, norbs_ao,
            dens_deriv, max_memory):
        rho = np.asarray ([m[0] (0, ao, mask, xctype) for m in make_rho])
        t0 = logger.timer (ot, 'untransformed density', *t0)
        Pi = get_ontop_pair_density (ot, rho, ao, oneCDMs, twoCDM_amo, ao2amo,
            dens_deriv, mask) 
        t0 = logger.timer (ot, 'on-top pair density calculation', *t0) 
        E_ot += ot.get_E_ot (rho, Pi, weight)
        t0 = logger.timer (ot, 'on-top energy calculation', *t0) 

    return E_ot

def get_energy_decomposition (mc, mo_coeff=None, ci=None, ot=None):
    ''' Compute a decomposition of the MC-PDFT energy into nuclear potential
        (E0), one-electron (E1), Coulomb (E2c), exchange (EOTx), correlation
        (EOTc) terms, and additionally the nonclassical part (E2nc) of the
        MC-SCF energy:

        E(MC-SCF) = E0 + E1 + E2c + Enc
        E(MC-PDFT) = E0 + E1 + E2c + EOTx + EOTc

        Only compatible with pure translated or fully-translated functionals.
        If mc.fcisolver.nroots > 1, lists are returned for everything except
        the nuclear potential energy.

        Args:
            mc : an instance of CASSCF or CASCI class

        Kwargs:
            mo_coeff : ndarray
                Contains MO coefficients
            ci : ndarray or list of length nroots
                Contains CI vectors
            ot : an instance of (translated) on-top density fnal class

        Returns:
            e_nuc : float
                E0 = sum_A>B ZA*ZB/rAB
            e_1e : float or list of length nroots
                E1 = <T+sum_A ZA/rA> 
            e_coul : float or list of length nroots
                E2c = 1/2 int rho(1)rho(2)/r12 d1d2
            e_otx : float or list of length nroots
                EOTx = exchange part of translated functional
            e_otc : float or list of length nroots
                EOTx = correlation part of translated functional
            e_ncwfn : float or list of length nroots
                E2ncc = <H> - E0 - E1 - E2c
    '''            
    if mo_coeff is None: mo_coeff=mc.mo_coeff
    if ci is None: mo_coeff=mc.ci
    if ot is None: ot = mc.otfnal

    hyb_x, hyb_c = ot._numint.hybrid_coeff(ot.otxc)
    if hyb_x>1e-10 or hyb_c>1e-10:
        raise NotImplementedError ("Decomp for hybrid PDFT fnals")
    if not isinstance (ot, transfnal):
        raise NotImplementedError ("Decomp for non-translated PDFT fnals")

    e_nuc = mc._scf.energy_nuc ()
    h = mc.get_hcore ()
    xfnal, cfnal = ot.split_x_c ()
    nroots = getattr (mc.fcisolver, 'nroots', 1)
    if nroots>1:
        e_1e = []
        e_coul = []
        e_otx = []
        e_otc = []
        e_ncwfn = []
        nelec_root = [mc.nelecas,]*nroots
        if isinstance (mc.fcisolver, StateAverageMixFCISolver):
            nelec_root = []
            for s in mc.fcisolver.fcisolvers:
                ne_root_s = mc.fcisolver._get_nelec (s, mc.nelecas)
                nelec_root.extend ([ne_root_s,]*s.nroots)
        for ci_i, nelec in zip (ci, nelec_root):
            row = _get_e_decomp (mc, ot, mo_coeff, ci_i, e_nuc, h,
                xfnal, cfnal, nelec)
            e_1e.append  (row[0])
            e_coul.append  (row[1])
            e_otx.append   (row[2])
            e_otc.append   (row[3])
            e_ncwfn.append (row[4])
    else:
        e_1e, e_coul, e_otx, e_otc, e_ncwfn = _get_e_decomp (mc, ot,
            mo_coeff, ci, e_nuc, h, xfnal, cfnal, mc.nelecas)
    return e_nuc, e_1e, e_coul, e_otx, e_otc, e_ncwfn

def _get_e_decomp (mc, ot, mo_coeff, ci, e_nuc, h, xfnal, cfnal,
        nelecas):
    ncore, ncas = mc.ncore, mc.ncas
    _rdms = mcscf.CASCI (mc._scf, ncas, nelecas)
    _rdms.fcisolver = fci.solver (mc._scf.mol, singlet = False, symm = False)
    _rdms.mo_coeff = mo_coeff
    _rdms.ci = ci
    _casdms = _rdms.fcisolver
    h1, h0 = _rdms.h1e_for_cas ()
    h2 = ao2mo.restore (1,_rdms.ao2mo (), ncas)
    dm1s = np.stack (_rdms.make_rdm1s (), axis=0)
    dm1 = dm1s[0] + dm1s[1]
    j = _rdms._scf.get_j (dm=dm1)
    e_1e = np.tensordot (h, dm1, axes=2)
    e_coul = np.tensordot (j, dm1, axes=2) / 2
    adm1, adm2 = _casdms.make_rdm12 (_rdms.ci, ncas, nelecas)
    e_mcscf = h0 + np.dot (h1.ravel (), adm1.ravel ()) + (
                np.dot (h2.ravel (), adm2.ravel ())*0.5)
    adm1s = np.stack (_casdms.make_rdm1s (ci, ncas, nelecas), axis=0)
    adm2 = get_2CDM_from_2RDM (_casdms.make_rdm12 (_rdms.ci, ncas, nelecas)[1],
        adm1s)
    mo_cas = mo_coeff[:,ncore:][:,:ncas]
    e_otx = get_E_ot (xfnal, dm1s, adm2, mo_cas, max_memory=mc.max_memory)
    e_otc = get_E_ot (cfnal, dm1s, adm2, mo_cas, max_memory=mc.max_memory)
    e_ncwfn = e_mcscf - e_nuc - e_1e - e_coul
    return e_1e, e_coul, e_otx, e_otc, e_ncwfn

class _mcscf_env (object):
    ''' Prevent MC-SCF step of MC-PDFT from overwriting redefined
        quantities e_states and e_tot '''
    def __init__(self, mc):
        self.mc = mc
        self.e_tot = deepcopy (self.mc.e_tot)
        self.e_states = deepcopy (getattr (self.mc, 'e_states', None))
    def __enter__(self):
        self.mc._in_mcscf_env = True
    def __exit__(self, type, value, traceback):
        self.mc.e_tot = self.e_tot
        if getattr (self.mc, 'e_states', None) is not None:
            self.mc.e_mcscf = np.array (self.mc.e_states)
        if self.e_states is not None:
            try:
                self.mc.e_states = self.e_states
            except AttributeError as e:
                self.mc.fcisolver.e_states = self.e_states
                assert (self.mc.e_states is self.e_states), str (e)
            # TODO: redesign this. MC-SCF e_states is stapled to
            # fcisolver.e_states, but I don't want MS-PDFT to be 
            # because that makes no sense
        self.mc._in_mcscf_env = False

# TODO: docstring
class _PDFT ():
    # Metaclass parent; unusable on its own

    def __init__(self, scf, ncas, nelecas, my_ot=None, grids_level=None,
            grids_attr=None, **kwargs):
        # Keep the same initialization pattern for backwards-compatibility.
        # Use a separate intializer for the ot functional
        if grids_attr is None: grids_attr = {}
        try:
            super().__init__(scf, ncas, nelecas)
        except TypeError as e:
            # I think this is the same DFCASSCF problem as with the DF-SACASSCF
            # gradients earlier
            super().__init__()
        keys = set (('e_ot', 'e_mcscf', 'get_pdft_veff', 'e_states', 'otfnal',
            'grids', 'max_cycle_fp', 'conv_tol_ci_fp', 'mcscf_kernel'))
        self.max_cycle_fp = getattr (__config__, 'mcscf_mcpdft_max_cycle_fp',
            50)
        self.conv_tol_ci_fp = getattr (__config__,
            'mcscf_mcpdft_conv_tol_ci_fp', 1e-8)
        self.mcscf_kernel = super().kernel
        self._in_mcscf_env = False
        self._keys = set ((self.__dict__.keys ())).union (keys)
        if grids_level is not None:
            grids_attr['level'] = grids_level
        if my_ot is not None:
            self._init_ot_grids (my_ot, grids_attr=grids_attr)

    def _init_ot_grids (self, my_ot, grids_attr=None):
        if grids_attr is None: grids_attr = {}
        old_grids = getattr (self, 'grids', None)
        if isinstance (my_ot, (str, np.string_)):
            self.otfnal = get_transfnal (self.mol, my_ot)
        else:
            self.otfnal = my_ot
        if isinstance (old_grids, gen_grid.Grids):
            self.otfnal.grids = old_grids
        #self.grids = self.otfnal.grids
        self.grids.__dict__.update (grids_attr)
        for key in grids_attr:
            assert (getattr (self.grids, key, None) == getattr (
                self.otfnal.grids, key, None))
        # Make sure verbose and stdout don't accidentally change 
        # (i.e., in scanner mode)
        self.otfnal.verbose = self.verbose
        self.otfnal.stdout = self.stdout

    @property
    def grids (self): return self.otfnal.grids
    @grids.setter
    def grids (self, x):
        self.otfnal.grids = x
        return self.otfnal.grids 

    def optimize_mcscf_(self, mo_coeff=None, ci0=None, **kwargs):
        ''' Optimize the MC-SCF wave function underlying an MC-PDFT calculation.
            Has the same calling signature as the parent kernel method. '''
        with _mcscf_env (self):
            self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy = \
                super().kernel (mo_coeff, ci0=ci0, **kwargs)
        return self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def compute_pdft_energy_(self, mo_coeff=None, ci=None, ot=None, otxc=None,
                             grids_level=None, grids_attr=None, **kwargs):
        ''' Compute the MC-PDFT energy(ies) (and update stored data)
            with the MC-SCF wave function fixed. '''
        if mo_coeff is not None: self.mo_coeff = mo_coeff
        if ci is not None: self.ci = ci
        if ot is not None: self.otfnal = ot
        if otxc is not None: self.otxc = otxc
        if grids_attr is None: grids_attr = {}
        if grids_level is not None: grids_attr['level'] = grids_level
        if len (grids_attr): self.grids.__dict__.update (**grids_attr)
        nroots = getattr (self.fcisolver, 'nroots', 1)
        if nroots>1:
            epdft = [self.energy_tot (mo_coeff=self.mo_coeff, ci=self.ci, state=ix,
                     logger_tag='MC-PDFT state {}'.format (ix))
                     for ix in range (nroots)]
            self.e_ot = [e_ot for e_tot, e_ot in epdft]
            if isinstance (self, StateAverageMCSCFSolver):
                e_states = [e_tot for e_tot, e_ot in epdft]
                try:
                    self.e_states = e_states
                except AttributeError as e:
                    self.fcisolver.e_states = e_states
                    assert (self.e_states is e_states), str (e)
                # TODO: redesign this. MC-SCF e_states is stapled to
                # fcisolver.e_states, but I don't want MS-PDFT to be 
                # because that makes no sense
                self.e_tot = np.dot (e_states, self.weights)
                e_states = self.e_states
            else: # nroots>1 CASCI
                self.e_tot = [e_tot for e_tot, e_ot in epdft]
                e_states = self.e_tot
            return self.e_tot, self.e_ot, e_states
        else:
            self.e_tot, self.e_ot = self.energy_tot (mo_coeff=self.mo_coeff, ci=self.ci)
            return self.e_tot, self.e_ot, [self.e_tot] 

    def kernel (self, mo_coeff=None, ci0=None, otxc=None, grids_attr=None,
                grids_level=None, **kwargs):
        self.optimize_mcscf_(mo_coeff=mo_coeff, ci0=ci0, **kwargs)
        self.compute_pdft_energy_(otxc=otxc, grids_attr=grids_attr,
                                  grids_level=grids_level, **kwargs)
        # TODO: edit StateAverageMCSCF._finalize in pyscf.mcscf.addons
        # to use the proper name of the class rather than "CASCI", so
        # that I can meaningfully play with "finalize" here
        return (self.e_tot, self.e_ot, self.e_mcscf, self.e_cas, self.ci,
            self.mo_coeff, self.mo_energy)

    def dump_flags (self, verbose=None):
        super().dump_flags (verbose=verbose)
        log = logger.new_logger(self, verbose)
        log.info ('on-top pair density exchange-correlation functional: %s',
            self.otfnal.otxc)

    def get_pdft_veff (self, mo=None, ci=None, state=0, casdm1s=None,
            casdm2=None, incl_coul=False, paaa_only=False, aaaa_only=False,
            jk_pc=False):
        ''' Get the 1- and 2-body MC-PDFT effective potentials for a set
            of mos and ci vectors

            Kwargs:
                mo : ndarray of shape (nao,nmo)
                    A full set of molecular orbital coefficients. Taken
                    from self if not provided
                ci : list or ndarray
                    CI vectors. Taken from self if not provided
                state : integer
                    Indexes a specific state in state-averaged
                    calculations. If negative, it generates a
                    state-averaged effective potential.
                casdm1s : ndarray of shape (2,ncas,ncas)
                    Spin-separated 1-RDM in the active space. Overrides
                    CI if and only if both this and casdm2 are provided
                casdm2 : ndarray of shape (ncas,ncas,ncas,ncas)
                    2-RDM in the active space. Overrides CI if and only
                    if both this and casdm1s are provided 
                incl_coul : logical
                    If true, includes the Coulomb repulsion energy in
                    the 1-body effective potential.
                paaa_only : logical
                    If true, only the paaa 2-body effective potential
                    elements are evaluated; the rest of ppaa are filled
                    with zeros.
                aaaa_only : logical
                    If true, only the aaaa 2-body effective potential
                    elements are evaluated; the rest of ppaa are filled
                    with zeros.
                jk_pc : logical
                    If true, calculate the ppii=pipi 2-body effective
                    potential in veff2.j_pc and veff2.k_pc. Otherwise
                    these arrays are filled with zeroes.

            Returns:
                veff1 : ndarray of shape (nao, nao)
                    1-body effective potential in the AO basis
                    May include classical Coulomb potential term (see
                    incl_coul kwarg)
                veff2 : pyscf.mcscf.mc_ao2mo._ERIS instance
                    Relevant 2-body effective potential in the MO basis
        ''' 
        t0 = (logger.process_clock (), logger.perf_counter ())
        if mo is None: mo = self.mo_coeff
        if ci is None: ci = self.ci
        ncore, ncas, nelecas = self.ncore, self.ncas, self.nelecas
        nocc = ncore + ncas

        if (casdm1s is not None) and (casdm2 is not None):
            mo_core = mo[:,:ncore]
            mo_cas = mo[:,ncore:nocc]
            dm1s = np.dot (mo_cas, casdm1s).transpose (1,0,2)
            dm1s = np.dot (dm1s, mo_cas.conj ().T)
            dm1s += (mo_core @ mo_core.conj ().T)[None,:,:]
            adm1s = casdm1s
            adm2 = get_2CDM_from_2RDM (casdm2, casdm1s)
        else:
            dm_list = self.make_rdms_mcpdft (mo_coeff=mo, ci=ci, state=state)
            dm1s, (adm1s, (adm2, _ss, _os)) = dm_list

        mo_cas = mo[:,ncore:][:,:ncas]
        pdft_veff1, pdft_veff2 = pdft_veff.kernel (self.otfnal, adm1s, 
            adm2, mo, ncore, ncas, max_memory=self.max_memory, 
            paaa_only=paaa_only, aaaa_only=aaaa_only, jk_pc=jk_pc)
        
        if incl_coul:
            pdft_veff1 += self._scf.get_j (self.mol, dm1s[0] + dm1s[1])
        logger.timer (self, 'get_pdft_veff', *t0)
        return pdft_veff1, pdft_veff2

    def _state_average_nuc_grad_method (self, state=None):
        from mrh.my_pyscf.grad.mcpdft import Gradients
        return Gradients (self, state=state)

    def nuc_grad_method (self):
        return self._state_average_nuc_grad_method (state=None)

    def dip_moment (self, unit='Debye', state=0):
        if isinstance (self, StateAverageMCSCFSolver):
            # TODO: SA dipole moment unittests
            logger.warn (self, "State-averaged dipole moments are UNTESTED!")
        from mrh.my_pyscf.prop.dip_moment.mcpdft import ElectricDipole
        dip_obj =  ElectricDipole(self) 
        mol_dipole = dip_obj.kernel (state=state)
        return mol_dipole

    def get_energy_decomposition (self, mo_coeff=None, ci=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        return get_energy_decomposition (self, mo_coeff=mo_coeff,
            ci=ci, ot=self.otfnal)

    def state_average_mix (self, fcisolvers=None, weights=(0.5,0.5)):
        return state_average_mix (self, fcisolvers, weights)

    def state_average_mix_(self, fcisolvers=None, weights=(0.5,0.5)):
         state_average_mix_(self, fcisolvers, weights)
         return self

    def state_interaction (self, weights=(0.5,0.5), diabatization='CMS'):
        from mrh.my_pyscf.mcpdft.sipdft import state_interaction
        return state_interaction (self, weights=weights,
                                  diabatization=diabatization)

    @property
    def otxc (self):
        return self.otfnal.otxc

    @otxc.setter
    def otxc (self, x):
        self._init_ot_grids (x)

    make_rdms_mcpdft = make_rdms_mcpdft
    energy_mcwfn = energy_mcwfn
    energy_dft = energy_dft
    def energy_tot (self, mo_coeff=None, ci=None, ot=None, state=0,
                    verbose=None, otxc=None, grids_level=None, grids_attr=None,
                    logger_tag='MC-PDFT'):
        ''' Compute the MC-PDFT energy of a single state '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if grids_attr is None: grids_attr = {}
        if grids_level is not None: grids_attr['level'] = grids_level
        if len (grids_attr) or (otxc is not None):
            old_ot = ot if (ot is not None) else self.otfnal
            old_grids = old_ot.grids
            # TODO: general compatibility with arbitrary (non-translated) fnals
            if otxc is None: otxc = old_ot.otxc
            new_ot = get_transfnal (self.mol, otxc)
            new_ot.grids.__dict__.update (old_grids.__dict__)
            new_ot.grids.__dict__.update (**grids_attr)
            ot = new_ot
        elif ot is None:
            ot = self.otfnal
        e_tot, e_ot = energy_tot (self, mo_coeff=mo_coeff, ot=ot, ci=ci,
            state=state, verbose=verbose)
        logger.note (self, '%s E = %s, Eot(%s) = %s', logger_tag,
            e_tot, ot.otxc, e_ot)
        return e_tot, e_ot

def get_mcpdft_child_class (mc, ot, **kwargs):
    # Inheritance magic
    class PDFT (_PDFT, mc.__class__):
        pass

    pdft = PDFT (mc._scf, mc.ncas, mc.nelecas, my_ot=ot, **kwargs)
    _keys = pdft._keys.copy ()
    pdft.__dict__.update (mc.__dict__)
    pdft._keys = pdft._keys.union (_keys)
    return pdft

