# vasp2spn

vasp2spn.py is a modified version of vaspspn.py provided in [wannierberri](https://github.com/wannier-berri/wannier-berri) written by Stepan Tsirkin, which aims to generate wannier90.spn from VASP output WAVECAR. Evaluation of spin matrix in vaspspn.py is approximate and based on normalized pseudo-wavefunction contained in WAVECAR which ignores the augmentation part of the all-electron wavefunction in PAW frameworks. My vasp2spn.py instead recovers the augementation part and in principle provides the exact results using all-electron wavefunctions. As a price, vasp2spn.py also requires pseudopotential information contained in POTCAR and atomic information contained in POSCAR.

A test file is also provided. orthotest.py 

