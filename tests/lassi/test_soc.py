import unittest
import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from pyscf.fci.direct_spin1 import _unpack_nelec
from mrh.tests.lasscf.c2h6n4_struct import structure as struct
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.lassi import dms as lassi_dms
from mrh.my_pyscf.mcscf.soc_int import compute_hso, amfi_dm
from mrh.my_pyscf.lassi.op_o0 import ci_outer_product
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi.lassi import make_stdm12s, roots_make_rdm12s, roots_trans_rdm12s, ham_2q
from mrh.my_pyscf import lassi
import itertools

def setUpModule():
    global mol1, mf1, mol2, mf2, las2, lsi2, oldvars
    oldvars = {}
    from mrh.my_pyscf.lassi.op_o1 import frag
    oldvars['SCREEN_THRESH'] = frag.SCREEN_THRESH
    frag.SCREEN_THRESH = 1e-32
    mol1 = gto.M (atom="""
        O  0.000000  0.000000  0.000000
        H  0.758602  0.000000  0.504284
        H  -0.758602  0.000000  0.504284
    """, basis='631g',symmetry=True,
    output='/dev/null', #'test_soc1.log',
    verbose=0) #lib.logger.DEBUG)
    mf1 = scf.RHF (mol1).run ()
   
    # NOTE: Test systems don't have to be scientifically meaningful, but they do need to
    # be "mathematically" meaningful. I.E., you can't just test zero. You need a test case
    # where the effect of the thing you are trying to test is numerically large enough
    # to be reproduced on any computer. Calculations that don't converge can't be used
    # as test cases for this reason.
    mol2 = struct (2.0, 2.0, '6-31g', symmetry=False)
    mol2.output = '/dev/null' #'test_soc2.log'
    mol2.verbose = 0 #lib.logger.DEBUG
    mol2.build ()
    mf2 = scf.RHF (mol2).run ()
    las2 = LASSCF (mf2, (4,4), (4,4), spin_sub=(1,1))
    las2.mo_coeff = las2.localize_init_guess ((list (range (3)), list (range (9,12))), mf2.mo_coeff)
    # NOTE: for 2-fragment and above, you will ALWAYS need to remember to do the line above.
    # If you skip it, you can expect your orbitals not to converge.
    las2.state_average_(weights=[1,0,0,0,0,0,0],
                            spins=[[0,0],[2,0],[-2,0],[0,2],[0,-2],[0,0],[0,0]],
                            smults=[[1,1],[3,1],[3,1],[1,3],[1,3],[3,1],[1,3]])

    # NOTE: Be careful about state selection. You have to select states that can actually be coupled
    # by a 1-body SOC operator. For instance, spins=[0,0] and spins=[2,2] would need at least a 2-body
    # operator to couple.
    las2.kernel ()
    # Light speed value chosen because it changes the ground state from a triplet to 
    # a contaminated quasi-singlet.
    with lib.light_speed (5):
        lsi2 = lassi.LASSI (las2, soc=True, break_symmetry=True)
        lsi2.kernel (opt=0)

def tearDownModule():
    global mol1, mf1, mol2, mf2, las2, lsi2, oldvars
    from mrh.my_pyscf.lassi.op_o1 import frag
    for key, val in oldvars.items (): setattr (frag, key, val)
    mol1.stdout.close()
    mol2.stdout.close()
    del mol1, mf1, mol2, mf2, las2, lsi2, oldvars

def case_soc_stdm12s_slow (self, opt=0):
    stdm1s_test, stdm2s_test = make_stdm12s (las2, soc=True, opt=opt) 
    with self.subTest ('2-electron'):
        self.assertAlmostEqual (linalg.norm (stdm2s_test), 16.901692823561433, 6)
    with self.subTest ('1-electron'):
        self.assertAlmostEqual (linalg.norm (stdm1s_test), 7.075259874940101, 6)
    dm1s_test = lib.einsum ('ipqi->ipq', stdm1s_test)
    with self.subTest (oneelectron_sanity='diag'):
        # LAS states are spin-pure: there should be nothing in the spin-breaking sector
        self.assertAlmostEqual (np.amax(np.abs(dm1s_test[:,8:,:8])), 0)
        self.assertAlmostEqual (np.amax(np.abs(dm1s_test[:,:8,8:])), 0)
    dm2_test = lib.einsum ('iabcdi->iabcd', stdm2s_test.sum ((1,4)))
    e0, h1, h2 = ham_2q (las2, las2.mo_coeff, soc=True)
    e1 = lib.einsum ('pq,ipq->i', h1, dm1s_test)
    e2 = lib.einsum ('pqrs,ipqrs->i', h2, dm2_test) * .5
    e_test = e0 + e1 + e2
    with self.subTest (sanity='spin-free total energies'):
        self.assertAlmostEqual (lib.fp (e_test), lib.fp (las2.e_states), 6)
    # All the stuff below is about making sure that the nonzero part of this is in
    # exactly the right spot
    for i in range (1,5):
        ifrag, ispin = divmod (i-1, 2)
        jfrag = int (not bool (ifrag))
        with self.subTest (oneelectron_sanity='hermiticity', ket=i):
            self.assertAlmostEqual (lib.fp (stdm1s_test[0,:,:,i]),
                                    lib.fp (stdm1s_test[i,:,:,0].conj ().T), 16)
        d1 = stdm1s_test[0,:,:,i].reshape (2,2,4,2,2,4)
        with self.subTest (oneelectron_sanity='fragment-local', ket=i):
            self.assertAlmostEqual (np.amax(np.abs(d1[:,jfrag,:,:,:,:])), 0, 16)
            self.assertAlmostEqual (np.amax(np.abs(d1[:,:,:,:,jfrag,:])), 0, 16)
        d1 = d1[:,ifrag,:,:,ifrag,:]
        with self.subTest (oneelectron_sanity='sf sector zero', ket=i):
            self.assertAlmostEqual (np.amax(np.abs(d1[0,:,0,:])), 0, 16)
            self.assertAlmostEqual (np.amax(np.abs(d1[1,:,1,:])), 0, 16)
        with self.subTest (oneelectron_sanity='raising XOR lowering', ket=i):
            if ispin: # <0|pq|down>
                self.assertAlmostEqual (np.amax (np.abs (d1[0,:,1,:])), 0, 16)
                d1=d1[1,:,0,:] # [beta,alpha] -> alpha' beta
            else: # <0|pq|up>
                self.assertAlmostEqual (np.amax (np.abs (d1[1,:,0,:])), 0, 16)
                d1=d1[0,:,1,:] # [alpha,beta] -> beta' alpha
        with self.subTest (oneelectron_sanity='nonzero S.O.C.', ket=i):
            self.assertAlmostEqual (linalg.norm (d1), 1.1539612627187337, 8)
    # lassi_dms.make_trans and total electron count
    ncas = las2.ncas
    nelec_fr = [[_unpack_nelec (fcibox._get_nelec (solver, nelecas))
                 for solver in fcibox.fcisolvers]
                for fcibox, nelecas in zip (las2.fciboxes, las2.nelecas_sub)]
    ci_r, nelec_r = ci_outer_product (las2.ci, las2.ncas_sub, nelec_fr)
    for i in range (5):
        nelec_r_test = (int (round (np.trace (stdm1s_test[i,:ncas,:ncas,i]))),
                        int (round (np.trace (stdm1s_test[i,ncas:,ncas:,i]))))
        with self.subTest ('electron count', state=i):
            self.assertEqual (nelec_r_test, nelec_r[i])
    def dm_sector (dm, m):
        if m==1: return dm[ncas:2*ncas,0:ncas] # [beta,alpha] -> alpha' beta
        elif m==-1: return dm[0:ncas,ncas:2*ncas] # [alpha,beta] -> beta' alpha
        elif m==0:
            return (dm[0:ncas,0:ncas] - dm[ncas:2*ncas,ncas:2*ncas])
        else: assert (False)
    for i,j in itertools.product (range(5), repeat=2):
        for m in (-1,0,1):
            t_test = dm_sector (stdm1s_test[i,:,:,j], m)
            t_ref = lassi_dms.make_trans (m, ci_r[i], ci_r[j], 8, nelec_r[i], nelec_r[j])
            with self.subTest ('lassi_dms agreement', bra=i, ket=j, sector=m):
                self.assertAlmostEqual (lib.fp (t_test), lib.fp (t_ref), 9)

def case_soc_rdm12s_slow (self, opt=0):
    # trans part
    si_ket = lsi2.si
    si_bra = np.roll (lsi2.si, 1, axis=1)
    rdm1s_test, rdm2s_test = roots_trans_rdm12s (las2, las2.ci, si_bra, si_ket, opt=opt)
    stdm1s, stdm2s = make_stdm12s (las2, soc=True, opt=opt)    
    rdm1s_ref = lib.einsum ('ir,jr,iabj->rab', si_bra.conj (), si_ket, stdm1s)
    rdm2s_ref = lib.einsum ('ir,jr,jsabtcdi->rsabtcd', si_ket.conj (), si_bra, stdm2s)
    with self.subTest (sanity='dm1s trans'):
        self.assertAlmostEqual (lib.fp (rdm1s_test), lib.fp (rdm1s_ref), 10)
    with self.subTest (sanity='dm2s trans'):
        self.assertAlmostEqual (lib.fp (rdm2s_test), lib.fp (rdm2s_ref), 10)
    # cis part
    rdm1s_test, rdm2s_test = roots_make_rdm12s (las2, las2.ci, lsi2.si, opt=opt)
    stdm1s, stdm2s = make_stdm12s (las2, soc=True, opt=opt)    
    rdm1s_ref = lib.einsum ('ir,jr,iabj->rab', lsi2.si.conj (), lsi2.si, stdm1s)
    rdm2s_ref = lib.einsum ('ir,jr,jsabtcdi->rsabtcd', lsi2.si.conj (), lsi2.si, stdm2s)
    with self.subTest (sanity='dm1s cis'):
        self.assertAlmostEqual (lib.fp (rdm1s_test), lib.fp (rdm1s_ref), 10)
    with self.subTest (sanity='dm2s cis'):
        self.assertAlmostEqual (lib.fp (rdm2s_test), lib.fp (rdm2s_ref), 10)
    # Stationary test has the issue of two doubly-degenerate manifolds: 1,2 and 4,5.
    # Therefore their RDMs actually vary randomly. Average the second and third RDMs
    # together to deal with this.
    rdm1s_test[1:3] = rdm1s_test[1:3].sum (0) / 2
    rdm2s_test[1:3] = rdm2s_test[1:3].sum (0) / 2
    rdm1s_test[4:6] = rdm1s_test[4:6].sum (0) / 2
    rdm2s_test[4:6] = rdm2s_test[4:6].sum (0) / 2
    with self.subTest ('2-electron'):
        self.assertAlmostEqual (linalg.norm (rdm2s_test), 13.584509751113796)
    with self.subTest ('1-electron'):
        self.assertAlmostEqual (linalg.norm (rdm1s_test), 5.298727485035966)
    with lib.light_speed (5):
        e0, h1, h2 = ham_2q (las2, las2.mo_coeff, soc=True)
    rdm2_test = rdm2s_test.sum ((1,4))
    e1 = lib.einsum ('pq,ipq->i', h1, rdm1s_test)
    e2 = lib.einsum ('pqrs,ipqrs->i', h2, rdm2_test) * .5
    e_test = e0 + e1 + e2 - las2.e_states[0]
    e_ref = lsi2.e_roots - las2.e_states[0]
    for ix, (test, ref) in enumerate (zip (e_test, e_ref)):
        with self.subTest (sanity='spin-orbit coupled total energies', state=ix):
            self.assertAlmostEqual (test, ref, 6)

class KnownValues (unittest.TestCase):

    # NOTE: In OpenMolcas, when using the ANO-RCC basis sets, the AMFI operator is switched from the Breit-Pauli
    # to the Douglass-Kroll Hamiltonian. There is no convenient way to switch this off; the only workaround
    # I've found is to "disguise" the basis set as something unrelated to ANO-RCC by copying and pasting it into
    # a separate file. Therefore, for now, we can only compare results from non-relativistic basis sets between
    # the two codes, until we implement Douglass-Kroll ourselves.

    def test_soc_int (self):
        # Obtained from OpenMolcas v22.02
        int_ref = 2*np.array ([0.0000000185242348, 0.0000393310222742, 0.0000393310222742, 0.0005295974407740]) 
        
        amfi_int = compute_hso (mol1, amfi_dm (mol1), amfi=True)
        amfi_int = amfi_int[2][amfi_int[2] > 0]
        amfi_int = np.sort (amfi_int.imag)
        self.assertAlmostEqual (lib.fp (amfi_int), lib.fp (int_ref), 8)

    def test_soc_1frag (self):
        # References obtained from OpenMolcas v22.10 (locally-modified to enable changing the speed of light,
        # see https://gitlab.com/MatthewRHermes/OpenMolcas/-/tree/amfi_speed_of_light)
        esf_ref = [0.0000000000,] + ([0.7194945289,]*3) + ([0.7485251565,]*3)
        eso_ref = [-0.0180900821,0.6646578117,0.6820416863,0.7194945289,0.7485251565,0.8033618737,0.8040680811]
        hso_ref = np.zeros ((7,7), dtype=np.complex128)
        hso_ref[1,0] =  0 - 10982.305j # T(+1)
        hso_ref[3,0] =  0 + 10982.305j # T(-1)
        hso_ref[4,2] =  10524.501 + 0j # T(+1)
        hso_ref[5,1] = -10524.501 + 0j # T(-1)
        hso_ref[5,3] =  10524.501 + 0j # T(+1)
        hso_ref[6,2] = -10524.501 + 0j # T(-1)
        hso_ref[5,0] =  0 - 18916.659j # T(0) < testing both this and T(+-1) is the reason I did 2 triplets
        
        las = LASSCF (mf1, (6,), (8,), spin_sub=(1,), wfnsym_sub=('A1',)).run (conv_tol_grad=1e-7)
        las.state_average_(weights=[1,0,0,0,0,0,0],
                           spins=[[0,],[2,],[0,],[-2,],[2,],[0,],[-2,],],
                           smults=[[1,],[3,],[3,],[3,],[3,],[3,],[3,],],
                           wfnsyms=([['A1',],]+([['B1',],]*3)+([['A2',],]*3)))
                           #wfnsyms=([['A1',],['B1',],['A2',],['B2',]]))
        las.lasci ()
        e0 = las.e_states[0]
        with self.subTest (deltaE='SF'):
            esf_test = las.e_states - e0
            self.assertAlmostEqual (lib.fp (esf_test), lib.fp (esf_ref), 6)
        ham = [None, None]
        for dson in (False, True):
            for opt in (0,1):
                with lib.light_speed (10):
                    lsi = lassi.LASSI (las).run (opt=opt, soc=True, break_symmetry=True,
                                                 davidson_only=dson, nroots_si=7)
                    e_roots, si = lsi.e_roots, lsi.si
                    h0, h1, h2 = ham_2q (las, las.mo_coeff, soc=True)
                ham[opt] = (si * e_roots[None,:]) @ si.conj ().T
                eso_test = e_roots - e0
                with self.subTest (opt=opt, davidson_only=dson, deltaE='SO'):
                    self.assertAlmostEqual (lib.fp (eso_test), lib.fp (eso_ref), 6)
                from pyscf.data import nist
                au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2
                def test_hso (hso_test, tag='kernel'):
                    hso_test *= au2cm
                    hso_test = np.around (hso_test, 8)
                    # Align relative signs: 0 - 1,3,5 block (all imaginary; vide supra)
                    for i in (1,3,5):
                        if np.sign (hso_test.imag[i,0]) != np.sign (hso_ref.imag[i,0]):
                            hso_test[i,:] *= -1
                            hso_test[:,i] *= -1
                    # Align relative signs: 2 - 4,6 block (all real; vide supra)
                    for i in (4,6):
                        if np.sign (hso_test.real[i,2]) != np.sign (hso_ref.real[i,2]):
                            hso_test[i,:] *= -1
                            hso_test[:,i] *= -1
                    for i, j in zip (*np.where (hso_ref)):
                        with self.subTest (tag, opt=opt, davidson_only=dson, hso=(i,j)):
                            try:
                                self.assertAlmostEqual (hso_test[i,j],hso_ref[i,j],1)
                            except AssertionError as e:
                                if abs (hso_test[i,j]+hso_ref[i,j]) < 0.05:
                                    raise AssertionError ("Sign fix failed for element",i,j)
                                raise (e)
                            # NOTE: 0.1 cm-1 -> 0.5 * 10^-6 au. These are actually tight checks.
                test_hso ((si * eso_test[None,:]) @ si.conj ().T)
                stdm1s, stdm2s = make_stdm12s (las, soc=True, break_symmetry=True, opt=opt)
                stdm2 = stdm2s.sum ((1,4))
                e0eff = h0 - e0
                h0eff = np.eye (7) * e0eff
                h1eff = lib.einsum ('pq,ipqj->ij', h1, stdm1s)
                h2eff = lib.einsum ('pqrs,ipqrsj->ij', h2, stdm2) * .5
                test_hso (h0eff + h1eff + h2eff, 'make_stdm12s')
                rdm1s, rdm2s = roots_make_rdm12s (las, las.ci, si, soc=True, break_symmetry=True,
                                                  opt=opt)
                rdm2 = rdm2s.sum ((1,4))
                e1eff = lib.einsum ('pq,ipq->i', h1, rdm1s)
                e2eff = lib.einsum ('pqrs,ipqrs->i', h2, rdm2) * .5
                test_hso ((si * (e0eff+e1eff+e2eff)[None,:]) @ si.conj ().T, 'roots_make_rdm12s')
        with self.subTest ('o0-o1 ham agreement'):
            self.assertAlmostEqual (lib.fp (ham[0]), lib.fp (ham[1]), 8)

    def test_soc_2frag (self):
        ## stationary test for >1 frag calc
        with self.subTest (deltaE='SF'):
            self.assertAlmostEqual (lib.fp (lsi2._las.e_states), -214.8686632658775, 8)
        with self.subTest (opt=0, deltaE='SO'):
            self.assertAlmostEqual (lib.fp (lsi2.e_roots), -214.8684319949548, 8)
        for dson in (False, True):
            lsi = lassi.LASSI (lsi2._las, soc=True, break_symmetry=True, opt=1)
            lsi = lsi.set (davidson_only=dson, nroots_si=lsi2._las.nroots)
            with lib.light_speed (5): lsi.kernel (opt=1)
            with self.subTest (opt=1, davidson_only=dson, deltaE='SO'):
                self.assertAlmostEqual (lib.fp (lsi.e_roots), -214.8684319949548, 8)
            with self.subTest ('hamiltonian', opt=1, davidson_only=dson):
                ham_o0 = (lsi2.si * lsi2.e_roots[None,:]) @ lsi2.si.conj ().T
                ham_o1 = (lsi.si * lsi.e_roots[None,:]) @ lsi.si.conj ().T
                self.assertAlmostEqual (lib.fp (ham_o1), lib.fp (ham_o0), 8)

    def test_soc_stdm12s_slow_o0 (self):
        case_soc_stdm12s_slow (self, opt=0)

    def test_soc_stdm12s_slow_o1 (self):
        case_soc_stdm12s_slow (self, opt=1)
        d_test = make_stdm12s (las2, soc=True, opt=1)
        d_ref = make_stdm12s (las2, soc=True, opt=0)
        for i,j in itertools.product (range (len (d_test[0])), repeat=2): 
            for r in range (2):
                with self.subTest (rank=r+1, element=(i,j)):
                    self.assertAlmostEqual (lib.fp (d_test[r][i,...,j]),
                                            lib.fp (d_ref[r][i,...,j]), 8)

    def test_soc_rdm12s_slow_o0 (self):
        case_soc_rdm12s_slow (self, opt=0)

    def test_soc_rdm12s_slow_o1 (self):
        case_soc_rdm12s_slow (self, opt=1)
        d_test = roots_make_rdm12s (las2, las2.ci, lsi2.si, opt=1)
        d_ref = roots_make_rdm12s (las2, las2.ci, lsi2.si, opt=0)
        for i in range (len (d_test[0])):
            for r in range (2):
                with self.subTest (state=i, rank=r+1):
                    self.assertAlmostEqual (lib.fp (d_test[r][i]), lib.fp (d_ref[r][i]), 8)

if __name__ == "__main__":
    print("Full Tests for SOC")
    unittest.main()

