from mrh.my_pyscf.mcscf import chkfile as las_chkfile
from mrh.my_pyscf.lassi import chkfile as lsi_chkfile

KEYS_CONFIG_LASSIS = lsi_chkfile.KEYS_CONFIG_LASSI
KEYS_SACONSTR_LASSIS = lsi_chkfile.KEYS_SACONSTR_LASSI
KEYS_RESULTS_LASSIS = lsi_chkfile.KEYS_RESULTS_LASSI

def load_lsis_(lsis, chkfile=None, method_key='lsi',
               keys_config=KEYS_CONFIG_LASSIS,
               keys_saconstr=KEYS_SACONSTR_LASSIS,
               keys_results=KEYS_RESULTS_LASSIS):
    if chkfile is None: chkfile = lsis.chkfile
    if chkfile is None: raise RuntimeError ('chkfile not specified')
    data = load (chkfile, method_key)
    if data is None: raise KeyError ('{} record not in chkfile'.format (method_key.upper()))

    lsis = las_chkfile._load_las_1_(lsis, data,
                                    keys_config=keys_config,
                                    keys_saconstr=keys_saconstr,
                                    keys_results=keys_results)
    lsis = _load_lsis_ci_(lsis, data)
    return lsis

def _load_lsis_ci_(lsis, data):
    pass

def dump_lsis (lsis, chkfile=None, method_key='las', mo_coeff=None,
               overwrite_mol=True, keys_config=KEYS_CONFIG_LASSIS,
               keys_saconstr=KEYS_SACONSTR_LASSIS,
               keys_results=KEYS_RESULTS_LASSIS,
               **kwargs):
    if chkfile is None: chkfile = lsis.chkfile
    if not chkfile: return lsis
    if mo_coeff is None: mo_coeff = lsis.mo_coeff
    kwargs['mo_coeff'] = mo_coeff

    data = las_chkfile._dump_las_get_data (lsis, keys_config, keys_saconstr, keys_results,
                                           **kwargs)
    with h5py.File (chkfile, 'a') as fh5:
        chkdata = las_chkfile._dump_las_get_chkdata (lsis, fh5, overwrite_mol, method_key)
        las_chkfile._dump_las_1_(lsis, chkdata, data, mo_coeff)
        _dump_lsis_ci_(lsis, chkdata)
    return lsis

def _dump_lsis_ci_(lsis, chkdata):
    pass

