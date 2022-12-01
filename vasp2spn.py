
""" An utility to calculate the ``.spn`` file for wannier90 from VASP input file ``POSCAR``,``POTCAR`` and output file``WAVECAR`` <https://www.vasp.at/>`_. Augmentation part of PAW wavefunction is recovered during evaluation of spin operator.
    usage : ::
        python3 -m vasp2spn.py   option=value
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
"""
import numpy as np
from numpy.linalg import inv,norm
import sys
from scipy.io import FortranFile
import datetime


from scipy.interpolate import CubicSpline
from scipy.special import sph_harm
from scipy.linalg import block_diag


class Pseudopotential:
	"""
	Contains important attributes from a VASP pseudopotential files. POTCAR
	"settings" can be read from the pymatgen POTCAR object

	Note: for the following attributes, 'index' refers to an elist
	quantum number epsilon and angular momentum quantum number l,
	which define one set consisting of a projector function, all electron
	partial waves, and pseudo partial waves.

	Attributes:
		rmax (np.float64): Maximum radius of the projection operators
		grid (np.array): radial grid on which partial waves are defined
		aepotential (np.array): All electron potential defined radially on grid
		aecorecharge (np.array): All electron core charge defined radially
			on grid (i.e. charge due to core, and not valence, electrons)
		kinetic (np.array): Core kinetic elist density, defined raidally on grid
		pspotential (np.array): pseudopotential defined on grid
		pscorecharge (np.array): pseudo core charge defined on grid
		ls (list): l quantum number for each index
		pswaves (list of np.array): pseudo partial waves for each index
		aewaves (list of np.array): all electron partial waves for each index
		projgrid (np.array): radial grid on which projector functions are defined
		recipprojs (list of np.array): reciprocal space projection operators
			for each index
		realprojs (list of np.array): real space projection operators
			for each index
	"""

	def __init__(self, data):
		"""
		Initializer for Pseudopotential.
		Should only be used by CoreRegion.

		Arguments:
			data (str): single-element pseudopotential
				(POTCAR) as a string
		"""
		nonradial, radial = data.split("PAW radial sets", 1)
		partial_waves = radial.split("pseudo wavefunction")
		gridstr, partial_waves = partial_waves[0], partial_waves[1:]
		self.pswaves = []
		self.aewaves = []
		self.recipprojs = []
		self.realprojs = []
		self.nonlocalprojs = []
		self.ls = []
		##################################################################################
		auguccstr, gridstr = gridstr.split("grid", 1)
		gridstr, aepotstr = gridstr.split("aepotential", 1)
		# aepotstr, corechgstr = aepotstr.split("core charge-density", 1)
		# try:
		# 	corechgstr, kenstr = corechgstr.split("kinetic elist-density", 1)
		# 	kenstr, pspotstr = kenstr.split("pspotential", 1)
		# except:
		# 	kenstr = "0 0"
		# 	corechgstr, pspotstr = corechgstr.split("pspotential", 1)
		# pspotstr, pscorechgstr = pspotstr.split("core charge-density (pseudized)", 1)
		self.grid = self.make_nums(gridstr)
		#self.aepotential = self.make_nums(aepotstr)
		#self.aecorecharge = self.make_nums(corechgstr)
		#self.kinetic = self.make_nums(kenstr)
		#self.pspotential = self.make_nums(pspotstr)
		#self.pscorecharge = self.make_nums(pscorechgstr)
		
		augstr, uccstr = auguccstr.split('uccopancies in atom', 1)
		head, augstr = augstr.split('augmentation charges (non sperical)', 1)
		augs = self.make_nums(augstr)
		##################################################################################
		for pwave in partial_waves:
			lst = pwave.split("ae wavefunction", 1)
			self.pswaves.append(self.make_nums(lst[0]))
			self.aewaves.append(self.make_nums(lst[1]))
		##################################################################################
		projstrs = nonradial.split("Non local Part")
		topstr, projstrs = projstrs[0], projstrs[1:]
		# self.Gmax = float(topstr[-24:-4])
		self.Gmax=float(topstr.split()[-2])
		space=len(topstr.split(".")[-1])+4
		topstr, atpschgstr = topstr[:-space].split("atomic pseudo charge-density", 1)
		try:
			topstr, corechgstr = topstr.split("core charge-density (partial)", 1)
			settingstr, localstr = topstr.split("local part", 1)
		except:
			corechgstr = "0 0"
			settingstr, localstr = topstr.split("local part", 1)
		
		# if "gradient corrections used for XC" in localstr:
		# 	localstr, self.gradxc = localstr.split("gradient corrections used for XC", 1)
		# 	self.gradxc = int(self.gradxc)
		# else:
		# 	self.gradxc = None
		# self.localpart = self.make_nums(localstr)
		# self.localnum = self.localpart[0]
		# self.localpart = self.localpart[1:]
		# self.coredensity = self.make_nums(corechgstr)
		# self.atomicdensity = self.make_nums(atpschgstr)

		for projstr in projstrs:
			lst = projstr.split("Reciprocal Space Part")
			nonlocalvals, projs = lst[0], lst[1:]
			self.rmax = self.make_nums(nonlocalvals.split()[2])[0]
			nonlocalvals = self.make_nums(nonlocalvals)
			l = int(nonlocalvals[0])
			count = int(nonlocalvals[1])
			self.nonlocalprojs.append(nonlocalvals[2:])
			for proj in projs:
				recipproj, realproj = proj.split("Real Space Part")
				self.recipprojs.append(self.make_nums(recipproj))
				self.realprojs.append(self.make_nums(realproj))
				self.ls.append(l)
		self.augs = augs.reshape([len(self.ls),len(self.ls)])
		settingstr, projgridstr = settingstr.split("STEP   =")
		self.ndata = int(settingstr.split()[-1])
		#projgridstr = projgridstr.split("END")[0]
		self.projgrid = np.arange(len(self.realprojs[0])) * self.rmax / len(self.realprojs[0]) #uniform radial grid
		self.step = (self.projgrid[0], self.projgrid[1])

	def make_nums(self, numstring):
		return np.fromstring(numstring, dtype = np.float64, sep = ' ')


class POTCAR:
	"""
	List of Pseudopotential objects to describe the core region of a structure.

	Attributes:
		pps (dict of Pseudopotential): keys are element symbols,
			values are Pseudopotential objects
	"""
	def __init__(self, filename):

		file = open(filename,'r')
		text = file.read()
		file.close()
		potcar = [term for term in text.strip().split('End of Dataset') if term !='']
		self.pps = {}
		for potsingle in potcar:
			element = potsingle[0:16].split()[1].split('_')[0]
			self.pps[element] = Pseudopotential(potsingle)

class WAVECAR:
    def __init__(self, filename):
        wave = open(filename,'rb')
        data=np.fromfile(wave,dtype='float64',count=3)
        nrecl,nspin,nprec=data.astype(int)
        nq = "complex64" if nprec==45200 else "complex128"

        wave.seek(nrecl*1)
        data = np.fromfile(wave,dtype='float64',count=3+3*3+1)
        nk,nband = data[0:2].astype(int) # number of kpoints and bands
        ecut = data[2] # cutoff elist
        A = data[3:3+9].reshape(3,3) # Lattice constant
        Efermi = data[12] #Fermi level

        c=0.26246582250211 #2m/hbar^2 I think
        B = 2*np.pi*np.linalg.inv(A).T # reciprocal Lattice constant
        nbmax,npmax = self.setgrid(B,ecut) 
        Ggrid = self.getGgrid(nbmax) 

        wave.seek(nrecl*2)
        data = np.fromfile(wave,dtype='float64',count=4)
        nplane = data[0].astype(int)
        k = data[1:4]
        ind = (((Ggrid+k)@B)**2).sum(axis=1)/c <ecut # pick out plane wave within cutoff sphere
        spinor = 1 if Ggrid[ind].shape[0]==nplane else 2

        klist = np.zeros([nk,3])
        occlist = np.zeros([nk,nspin,nband])
        elist = np.zeros([nk,nspin,nband])
        wavelist = np.zeros([nk,nspin,nband,spinor*npmax],dtype=nq) 
        nplist = np.zeros([nk]) #number of plane waves
        Gindx = []

        for isp in range(nspin):
            for ik in range(nk):
                irec=2+isp*nk*(nband+1)+ik*(nband+1)
                wave.seek(nrecl*irec)
                data = np.fromfile(wave,dtype='float64',count=4+3*nband)

                nplane = data[0].astype(int)
                klist[ik,:]=data[1:4]
                view = data[4:].reshape([nband,3])
                elist[ik,isp,:]=view[:,0]
                occlist[ik,isp,:] = view[:,2]

                ind = (((Ggrid+klist[ik])@B)**2).sum(axis=1)/c <ecut
                if isp ==0:
                    Gindx.append(np.arange(Ggrid.shape[0])[ind])
                if Gindx[-1].shape[0]*spinor!=nplane:
                    print("number of plane wave mismatch at k=",klist[ik,:])
                for ib in range(nband):
                    irec=2+isp*nk*(nband+1)+ik*(nband+1)+1+ib
                    wave.seek(nrecl*irec)
                    wavelist[ik,isp,ib,0:nplane] = np.fromfile(wave,dtype=nq,count=nplane)
                nplist[ik]=nplane
        wave.close()
        
        self.nk=nk
        self.nband=nband
        self.ecut=ecut
        self.A=A
        self.B=B
        self.Efermi=Efermi
        self.spinor=spinor
        self.klist=klist
        self.elist=elist
        self.occlist=occlist 
        self.Ggrid = Ggrid 
        self.Gindx=Gindx 
        self.wavelist=wavelist
        self.nplist=nplist


    def setgrid(self,B,ecut):
        c=0.26246582250211
        bmag = np.linalg.norm(B,axis=1)
        nbmax = np.zeros([3,3],dtype=int)

        phi12 = np.arccos(B[0,:]@B[1,:]/bmag[0]/bmag[1])
        vtmp = np.cross(B[0,:],B[1,:])
        vmag = norm(vtmp)
        sinphi123 = B[2,:]@vtmp/bmag[2]/vmag
        nbmax[0,0] = int(np.sqrt(ecut*c)/abs(np.sin(phi12))/bmag[0])+1
        nbmax[0,1] = int(np.sqrt(ecut*c)/abs(np.sin(phi12))/bmag[1])+1
        nbmax[0,2] = int(np.sqrt(ecut*c)/abs(sinphi123)/bmag[2])+1

        phi13 = np.arccos(B[0,:]@B[2,:]/bmag[0]/bmag[2])
        vtmp = np.cross(B[0,:],B[2,:])
        vmag = norm(vtmp)
        sinphi123 = B[1,:]@vtmp/bmag[1]/vmag
        nbmax[1,0] = int(np.sqrt(ecut*c)/abs(np.sin(phi13))/bmag[0])+1
        nbmax[1,1] = int(np.sqrt(ecut*c)/abs(sinphi123)/bmag[1])+1
        nbmax[1,2] = int(np.sqrt(ecut*c)/abs(np.sin(phi13))/bmag[2])+1

        phi23 = np.arccos(B[1,:]@B[2,:]/bmag[1]/bmag[2])
        vtmp = np.cross(B[1,:],B[2,:])
        vmag = norm(vtmp)
        sinphi123 = B[0,:]@vtmp/bmag[0]/vmag
        nbmax[2,0] = int(np.sqrt(ecut*c)/abs(sinphi123)/bmag[0])+1
        nbmax[2,1] = int(np.sqrt(ecut*c)/abs(np.sin(phi23))/bmag[1])+1
        nbmax[2,2] = int(np.sqrt(ecut*c)/abs(np.sin(phi23))/bmag[2])+1

        npmax = int((np.prod(nbmax,axis=1)*4/3*np.pi).min())
        nbmax = nbmax.max(axis=0)
        return nbmax,npmax
    def getGgrid(self,nbmax):
        x = np.arange(nbmax[0]*2+1)
        x = np.where(x<=nbmax[0],x,x-2*nbmax[0]-1)
        y = np.arange(nbmax[1]*2+1)
        y = np.where(y<=nbmax[1],y,y-2*nbmax[1]-1)
        z = np.arange(nbmax[2]*2+1)
        z = np.where(z<=nbmax[2],z,z-2*nbmax[2]-1)
        Z,Y,X = np.meshgrid(z,y,x,indexing='ij')
        Ggrid = np.block([X.reshape([-1,1]),Y.reshape([-1,1]),Z.reshape([-1,1])])
        return Ggrid



def decomment(str):
    return str.split("#")[0].split("!")[0]
class POSCAR:
    def __init__(self,file):
        pos = open(file, 'r')
        items=pos.readlines()
        pos.close()
        self.lattconst = float(decomment(items[1]))
        self.A = np.zeros([3,3])
        self.A[0,:]=np.asarray(decomment(items[2]).split(),dtype=float)*self.lattconst
        self.A[1,:]=np.asarray(decomment(items[3]).split(),dtype=float)*self.lattconst
        self.A[2,:]=np.asarray(decomment(items[4]).split(),dtype=float)*self.lattconst
        self.B = 2 * np.pi * (np.linalg.inv(self.A).T)
        self.elements=np.asarray(decomment(items[5]).split())
        self.nums =np.asarray(decomment(items[6]).split(),dtype=int)
        total = self.nums.sum()
        self.pos = np.zeros([total,3])
        start = 8
        if items[7][0]=='s' or items[7][0]=='S':
            start=start+1
        for i in range(total):
            self.pos[i,:]=np.asarray(items[i+start].split()[0:3],dtype=float)
        

class PAWSetting():
    def __init__(self,pos,pot):
        self.A=pos.A
        self.B=pos.B 
        self.vol=abs(np.linalg.det(self.A))
        self.elements=pos.elements
        self.atompos=pos.pos
        nchannel=0
        atomlabel=[]
        atomidx=[]
        lmidx = {}
        lmmax = {}
        ls ={}
        for itype in range(len(self.elements)):
            atomlabel += [pos.elements[itype]]*pos.nums[itype]
            lmmax_ = 0
            lmidx_=[]
            ls[pos.elements[itype]]=pot.pps[pos.elements[itype]].ls
            for l in pot.pps[pos.elements[itype]].ls:
                lmidx_.append(slice(lmmax_,lmmax_+2*l+1,1))
                lmmax_ +=2*l+1
            lmidx[pos.elements[itype]]=lmidx_
            lmmax[pos.elements[itype]]=lmmax_
            for iatoms in range(pos.nums[itype]):
                atomidx.append(slice(nchannel,nchannel+lmmax_,1))
                nchannel+=lmmax_
        self.nchannel=nchannel
        self.atomlabel=atomlabel
        self.atomidx=atomidx
        self.lmidx=lmidx
        self.lmmax=lmmax
        self.ls=ls

        recipprojs={}
        for iat in pos.elements:
            pps = pot.pps[iat]
            x = np.arange(100)/100*pps.Gmax
            recipprojs[iat]=[CubicSpline(x,pps.recipprojs[i],extrapolate=False) for i in range(len(pps.recipprojs))]
        self.recipprojs=recipprojs

        iL=np.zeros(nchannel,dtype=complex)
        for i in range(len(atomlabel)):
            for j in range(len(lmidx[atomlabel[i]])):
                iL[atomidx[i]][lmidx[atomlabel[i]][j]]=1j**(pot.pps[atomlabel[i]].ls[j])
        self.iL=iL #i^L

        Qij={}
        for iat in pos.elements:
            Qij[iat]=np.zeros([lmmax[iat],lmmax[iat]])
            for i in range(len(pot.pps[iat].ls)):
                for j in range(len(pot.pps[iat].ls)):
                    if pot.pps[iat].ls[i]==pot.pps[iat].ls[j]:
                        l=pot.pps[iat].ls[i]
                        Qij[iat][lmidx[iat][i],lmidx[iat][j]]=pot.pps[iat].augs[i,j]*np.eye(2*l+1)
        self.Qij=Qij
        self.Tij=block_diag(*([Qij[iat] for iat in atomlabel]))

        

def cart2sph(xyz):
    rtp=np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    rtp[:,0]=np.sqrt(xy + xyz[:,2]**2) # r
    rtp[:,1]=np.arctan2(np.sqrt(xy), xyz[:,2]) # theta
    rtp[:,2]=np.arctan2(xyz[:,1], xyz[:,0]) # phi
    return rtp
def calcproj(wav,paw,ik,isp,bands): #bands is a range object
    k=wav.klist[ik,:]
    Gvec = wav.Ggrid[wav.Gindx[ik]]
    rtp = cart2sph((Gvec+k)@wav.B)
    Gnorm=rtp[:,0]
    theta=rtp[:,1]
    phi=rtp[:,2]
    # set up spherical harmonics
    sqrt2=np.sqrt(2)
    YLM={}
    lmax=np.max([np.max(paw.ls[iat]) for iat in paw.ls.keys()])
    for l in range(lmax+1):
        YLM[l]=np.zeros([2*l+1,theta.shape[0]])
        for m in range(l+1):
            if m==0:
                YLM[l][m,:]= np.real(sph_harm(m,l,phi,theta))
            else:
                data = sph_harm(m,l,phi,theta)
                YLM[l][m,:]= sqrt2*((-1)**l)*np.real(data)
                YLM[l][-m,:]= sqrt2*((-1)**l)*np.imag(data)

    # plane wave expansion of projector  <P_nlm|k+G> = i^L*4PI*<P_ln(r)|j_l(k+G r)> Y_l^m(k+G)
    # 4PI*<P_ln(r)|j_l(k+G r)> has been tabulated in reciprojs, we only need interpolation
    Gproj={}
    fak = 1/np.sqrt(paw.vol)
    for iat in paw.elements:
        Gproj[iat]=np.zeros([paw.lmmax[iat],Gnorm.shape[0]])
        for i in range(len(paw.ls[iat])):
            l=paw.ls[iat][i]
            Gproj[iat][paw.lmidx[iat][i],:]=np.nan_to_num(paw.recipprojs[iat][i](Gnorm))[np.newaxis,:]*YLM[l]*fak

    # calculate projected amplitude  <p_i_nlm|Psi>= sum_G <p_i_nlm|k+G><k+G|Psi_k>
    nchannel=paw.nchannel
    nband=len(bands)
    proj=np.zeros([nband,wav.spinor*nchannel],dtype=complex)
    npw = Gnorm.shape[0]
    for j in range(wav.spinor):
        for i in range(len(paw.atomlabel)):
            expiGR=np.exp(2j*np.pi*Gvec@paw.atompos[i])
            iat=paw.atomlabel[i]
            proj[:,j*nchannel:(j+1)*nchannel][:,paw.atomidx[i]]\
                =(wav.wavelist[ik,isp,bands,j*npw:(j+1)*npw]*expiGR[np.newaxis,:])@Gproj[iat].T*paw.iL[paw.atomidx[i]][np.newaxis,:]

    return proj
    




def hlp():
    from termcolor import cprint
    cprint("vasp2spn  (utility)", 'green', attrs=['bold'])
    print(__doc__)


def main():
    import sys
    from scipy.io import FortranFile
    import datetime

    fwav="WAVECAR"
    fpot="POTCAR"
    fpos="POSCAR"
    fout = "wannier90.spn"
    NBout = 0
    IBstart = 1
    normalize = "norm"
    for arg in sys.argv[1:]:
        if arg == "-h":
            hlp()
            exit()
        else:
            k, v = arg.split("=")
            if   k=="fwav"  : fwav=v
            elif k=="fpot"  : fpot=v
            elif k=="fpos"  : fpos=v
            elif k == "NB": NBout = int(v)
            elif k == "IBstart": IBstart = int(v)
            elif k == "norm": normalize = v

    print("reading {0}, {1}, {2}\n writing to {3}".format(fwav,fpot,fpos,fout))

    wav=WAVECAR(fwav)
    pot=POTCAR(fpot)
    pos=POSCAR(fpos)
    paw=PAWSetting(pos,pot)

    if wav.spinor != 2: raise RuntimeError('WAVECAR does not contain spinor wavefunctions. ISPIN={0}'.format(ispin))

    NK=wav.nk 
    NBin=wav.nband

    IBstart -= 1
    if IBstart < 0: IBstart = 0
    if NBout <= 0: NBout = NBin
    if NBout + IBstart > NBin:
        print(
            ' WARNING: NB+IBstart-1=', NBout + IBstart,
            ' exceeds the number of bands in WAVECAR NBin=' + str(NBin) + '. We set NBout=' + str(NBin - IBstart))
        NBout = NBin - IBstart

    print(
        "WAVECAR contains {0} k-points and {1} bands.\n Writing {2} bands in the output starting from".format(
            NK, NBin, NBout))

    SPN=FortranFile(fout, 'w')
    header = "Created from wavecar/potcar/poscar at {0}".format(datetime.datetime.now().isoformat())
    header = header[:60]
    header += " " * (60 - len(header))
    SPN.write_record(bytearray(header, encoding='ascii'))
    SPN.write_record(np.array([NBout, NK], dtype=np.int32))

    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    for ik in range(NK):
        bands=range(IBstart,IBstart+NBout)
        proj=calcproj(wav,paw,ik,0,bands)
        npw = wav.Gindx[ik].shape[0]
        Id = np.eye(npw)
        print("k-point {0:3d} : {1:6d} plane waves".format(ik, npw*2))
        # SX = wav.wavelist[ik,0,bands,0:2*npw].conj()@np.kron(sx,Id)@wav.wavelist[ik,0,bands,0:2*npw].T+proj.conj()@np.kron(sx,paw.Tij)@proj.T
        # SY = wav.wavelist[ik,0,bands,0:2*npw].conj()@np.kron(sy,Id)@wav.wavelist[ik,0,bands,0:2*npw].T+proj.conj()@np.kron(sy,paw.Tij)@proj.T
        # SZ = wav.wavelist[ik,0,bands,0:2*npw].conj()@np.kron(sz,Id)@wav.wavelist[ik,0,bands,0:2*npw].T+proj.conj()@np.kron(sz,paw.Tij)@proj.T
        SX = wav.wavelist[ik,0,bands,0:2*npw].conj()*np.diag(np.kron(sx,Id))[np.newaxis,:]@wav.wavelist[ik,0,bands,0:2*npw].T\
             +proj.conj()@np.kron(sx,paw.Tij)@proj.T
        SY = wav.wavelist[ik,0,bands,0:2*npw].conj()*np.diag(np.kron(sy,Id))[np.newaxis,:]@wav.wavelist[ik,0,bands,0:2*npw].T\
             +proj.conj()@np.kron(sy,paw.Tij)@proj.T
        SZ = wav.wavelist[ik,0,bands,0:2*npw].conj()*np.diag(np.kron(sz,Id))[np.newaxis,:]@wav.wavelist[ik,0,bands,0:2*npw].T\
             +proj.conj()@np.kron(sz,paw.Tij)@proj.T
        A = np.array([s[n, m] for m in range(NBout) for n in range(m + 1) for s in (SX, SY, SZ)], dtype=np.complex128)
        SPN.write_record(A)


if __name__ == "__main__":
    main()
