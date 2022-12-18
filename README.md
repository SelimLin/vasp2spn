# vasp2spn

vasp2spn.py is a modified version of vaspspn.py provided in [wannierberri](https://github.com/wannier-berri/wannier-berri) written by Stepan Tsirkin, which aims to generate wannier90.spn from VASP output WAVECAR. Evaluation of spin matrix in vaspspn.py is approximate and based on normalized pseudo-wavefunction contained in WAVECAR which ignores the augmentation part of the all-electron wavefunction in PAW frameworks. My vasp2spn.py instead recovers the augementation part and in principle provides the exact results using all-electron wavefunctions. As a price, vasp2spn.py also requires pseudopotential information contained in POTCAR and atomic information contained in POSCAR.

Two test scripts are also provided: 

orthotest.py calculates overlap between selected wavefunctions and print the results that should demonstrate the orthonormality between Kohn-Sham orbitals. It is a useful test to check whether the augmentation parts of all-electron wavefunctions are correctly recovered.

procartest.py calculate atomic contribution for each Kohn-Sham orbitals and compare the results with VASP output PROCAR. This script requires a PROCAR calculated with LORBIT=12 and the orbital resolved phase informations are also calculated and compared.   


## Usage
Run the following command in the working directory:
```
        python3 -m vasp2spn.py   option=value
```
    Options
        -h
            |  print this help message
        fwav
            |  WAVECAR file name.
            |  default: WAVECAR
        fpot
            |  POTCAR file name.
            |  default: POTCAR
        fpos
            |  POSCAR file name.
            |  default: POSCAR
        fout
            |  outputfile name
            |  default: wannier90.spn
        IBstart
            |  the first band to be considered (counting starts from 1).
            |  default: 1
        NB
            |  number of bands in the output. If NB<=0 all bands are used.
            |  default: 0
