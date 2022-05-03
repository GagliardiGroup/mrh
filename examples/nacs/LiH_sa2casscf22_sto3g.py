from pyscf import gto, scf, mcscf, lib
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.tools.molcas2pyscf import *
from mrh.my_pyscf.grad.sacasscf_nacs import NonAdiabaticCouplings

# NAC signs are really, really hard to nail down.
# There are arbitrary signs associated with
# 1. The MO coefficients
# 2. The CI vectors
# 3. Almost any kind of post-processing (natural-orbital analysis, etc.)
# 4. Developer convention on whether the bra index or ket index is 1st
# It MIGHT help comparison to OpenMolcas if you load a rasscf.h5 file
# I TRIED to choose the same convention for #4 as OpenMolcas.
try:
    h5file = 'LiH_sa2casscf22_sto3g.rasscf.h5'
    mol = get_mol_from_h5 (h5file, 
                           output='LiH_sa2casscf22_sto3g.log',
                           verbose=lib.logger.INFO)
    mo = get_mo_from_h5 (mol, h5file)
except OSError as e:
    print ("OpenMolcas h5file not found; building CASSCF wfn myself...")
    mol = gto.M (atom='Li 0 0 0;H 1.5 0 0', basis='sto-3g',
                 output='LiH_sa2casscf22_sto3g.log', verbose=lib.logger.INFO)
    mo = None

mf = scf.RHF (mol).run ()
mc = mcscf.CASSCF (mf, 2, 2).set (fcisolver = csf_solver (mol, smult=1))
mc = mc.state_average ([0.5,0.5]).run (mo, conv_tol=1e-10)

mc_nacs = NonAdiabaticCouplings (mc)

# 1. Equivalent OpenMolcas input:
#    ```
#    &ALASKA
#    NAC=1 2
#    ```
print ("\nNAC <1|d0/dR>:\n",mc_nacs.kernel (state=(0,1)))

# 2. Equivalent OpenMolcas input:
#    ```
#    &ALASKA
#    NAC=1 2
#    NOCSF
#    ```
print ("\nNotice that according to the NACs printed above, rigidly moving the")
print ("molecule along the bond axis changes the electronic wave function, which")
print ("is obviously unphysical. This broken translational symmetry is due to the")
print ("'CSF contribution'. Omitting the CSF contribution corresponds to using")
print ("'electron-translation factors' and is requested by passing 'use_etfs=True'.")
print ("NAC <1|d0/dR> w/ ETFs:\n",mc_nacs.kernel (state=(0,1), use_etfs=True))
print ("These NACs are much more well-behaved: moving the molecule rigidly around")
print ("in space doesn't induce any change to the electronic wave function.")

print ("\nThe NACs are antisymmetric and diverge at conical intersections:")
print ("NAC <0|d1/dR>:\n",mc_nacs.kernel (state=(1,0)))

print ("\nWhat really matters for dynamics is how quickly it diverges. You can")
print ("get at this by calculating NACs multiplied by the energy difference")
print ("using the keyword 'mult_ediff=True'. This yields a symmetric quantity")
print ("which is real and finite at a CI and tells you one of the dimensions")
print ("of the branching plane.")
print ("NAC <1|d0/dR>*(E1-E0):\n",mc_nacs.kernel (state=(0,1), mult_ediff=True))
print ("NAC <0|d1/dR>*(E0-E1):\n",mc_nacs.kernel (state=(1,0), mult_ediff=True))

print ("\nUsing both 'use_etfs=True' and 'mult_ediff=True' corresponds to the")
print ("derivative of the off-diagonal element of the potential matrix.")
print ("<1|d0/dR>*(E1-E0) w/ ETFS = <1|dH/dR|0>:\n",
       mc_nacs.kernel(state=(0,1),use_etfs=True,mult_ediff=True))

