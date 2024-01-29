# gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
#
# Copyright (C) 2022 Qiming Sun
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Patch pyscf SCF modules to make all subclass of SCF class support GPU mode.
'''

from gpu4pyscf.df.df_jk import _get_jk
from pyscf.df import df_jk

#from gpu4pyscf.df import df as mrh_df
#from pyscf import df

from gpu4pyscf.lib.utils import patch_cpu_kernel

print(f'{df_jk} monkey-patched')
df_jk.get_jk = patch_cpu_kernel(df_jk.get_jk)(_get_jk)

#print(f'{df.DF.build} monkey-patched')
#df.DF.build = mrh_df._build.__get__(df)
