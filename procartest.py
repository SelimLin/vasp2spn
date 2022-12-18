import numpy as np
from numpy.linalg import inv,norm
import sys
from scipy.io import FortranFile
import datetime
import re


from scipy.interpolate import CubicSpline
from scipy.special import sph_harm
from scipy.linalg import block_diag
from scipy.integrate import simpson

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
        #     corechgstr, kenstr = corechgstr.split("kinetic elist-density", 1)
        #     kenstr, pspotstr = kenstr.split("pspotential", 1)
        # except:
        #     kenstr = "0 0"
        #     corechgstr, pspotstr = corechgstr.split("pspotential", 1)
        # pspotstr, pscorechgstr = pspotstr.split("core charge-density (pseudized)", 1)
        self.grid = self.make_nums(gridstr)
        self.r_h = np.log(self.grid[-1]/self.grid[0])/(self.grid.shape[0]-1)
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
            self.pswaves.append(self.make_nums(lst[0])) # Psi(r,theta,phi)=R(r)Y_lm(theta,phi)=u(r)Y_lm(theta,phi)/r
            self.aewaves.append(self.make_nums(lst[1])) # pswave and aewave are actually u(r)=R(r)r instead
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
        self.rdep=float(re.search("RDEP\s+=\s+[0-9]+.[0-9]+",settingstr).group().split()[-1])*0.529177249
        self.ridx=self.grid<self.rdep+5E-3
        # if "gradient corrections used for XC" in localstr:
        #     localstr, self.gradxc = localstr.split("gradient corrections used for XC", 1)
        #     self.gradxc = int(self.gradxc)
        # else:
        #     self.gradxc = None
        # self.localpart = self.make_nums(localstr)
        # self.localnum = self.localpart[0]
        # self.localpart = self.localpart[1:]
        # self.coredensity = self.make_nums(corechgstr)
        # self.atomicdensity = self.make_nums(atpschgstr)

        for projstr in projstrs:
            lst = projstr.split("Reciprocal Space Part")
            nonlocalvals, projs = lst[0], lst[1:]
            self.rmax = self.make_nums(nonlocalvals.split()[2])[0]
            nonlocalvals = self.make_nums(nonlocalvals.replace('D', 'E'))
            l = int(nonlocalvals[0])
            count = int(nonlocalvals[1])
            self.nonlocalprojs.append(nonlocalvals[2:])
            for proj in projs:
                recipproj, realproj = proj.split("Real Space Part")
                self.recipprojs.append(np.zeros(100+1))
                self.recipprojs[-1][1:101]=self.make_nums(recipproj)
                if np.mod(l,2)==0:
                    self.recipprojs[-1][0]=self.recipprojs[-1][2]
                else:
                    self.recipprojs[-1][0]=-self.recipprojs[-1][2]
                # self.recipprojs.append(self.make_nums(recipproj))
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
            element = potsingle[0:20].split()[1].split('_')[0]
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
def fixminus(str):
    return re.sub('-',' -',str)
def loadtxt(strs):
    nrow = len(strs)
    data = np.fromstring("".join(strs),sep=' ')
    ncol = round(len(data)/nrow)
    return data.reshape(nrow,ncol)
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
        
class procar:
    def __init__(self,file,nhse=0):
        pro = open(file,'r')
        text = pro.read()
        self.ispin = len(re.findall('# of k-points',text))
        ntot = len(re.findall('tot',text))
        pro.seek(0,0)
        items=pro.readlines()
        

        self.lphase = 1 if items[0].split()[-1]=='phase' else 0
        temp = items[1].split("#")
        self.nkpt=int(temp[1].split()[-1])-nhse
        self.nband=int(temp[2].split()[-1])
        self.nion=int(temp[3].split()[-1])
        self.ncomp = round(ntot/self.ispin/self.nband/(self.nkpt+nhse))-1
        self.channels = items[7].split()[1:-1]
        self.projlist = np.zeros([self.nkpt,self.nband,self.ispin*self.ncomp,self.nion,len(self.channels)])
        self.phase=np.zeros([self.nkpt,self.nband,self.ispin,self.nion,len(self.channels)],dtype=complex)
        self.klist = np.zeros([self.nkpt,3])
        self.kwlist = np.zeros([self.nkpt])
        self.elist = np.zeros([self.nkpt,self.nband,self.ispin])
        self.occlist =  np.zeros([self.nkpt,self.nband,self.ispin])

        #structure of PROCAR
        #length = 1(description)+per_spin*ispin
        #per_spin=2(#kpt+\n)+per_k*nkpt-1(at the end)
        #per_k=2(kptinfo+\n)+per_band*nband+1
        #per_band=2(bandinfo+\n)+1(ion)+ncomp*(nion+2)+lphase(nion+2)+1
        per_band=2+1+(self.nion+2)*self.ncomp+self.lphase*(self.nion+2+1)+1
        per_k=2+per_band*self.nband+1
        per_spin=2+per_k*self.nkpt-1
        

        for isp in range(self.ispin): 
            for ik in range(nhse,nhse+self.nkpt):
                term = fixminus(items[3+isp*per_spin+ik*per_k]).split()
                self.klist[ik-nhse,:]=np.asarray(term[4:7],dtype=float)
                self.kwlist[ik-nhse]=float(term[-1])
                for ib in range(self.nband):
                    self.elist[ik-nhse,ib,isp]=float(items[5+isp*per_spin+ik*per_k+ib*per_band].split()[4])
                    self.occlist[ik-nhse,ib,isp]=float(items[5+isp*per_spin+ik*per_k+ib*per_band].split()[-1])
                    skip = 8+per_spin*isp+per_k*ik+per_band*ib
                    for ic in range(self.ncomp):
                        pro.seek(0,0)
                        # print('skip=',skip+ic*(self.nion+1))
                        # self.projlist[ik-nhse,ib,isp*self.ncomp+ic,:,:]=np.loadtxt(pro,skiprows=skip+ic*(self.nion+2),max_rows=self.nion)[:,1:1+len(self.channels)]
                        il1 = skip+ic*(self.nion+2)
                        il2 = il1+self.nion
                        self.projlist[ik-nhse,ib,isp*self.ncomp+ic,:,:] = loadtxt(items[il1:il2])[:,1:1+len(self.channels)]
                    if self.lphase: #untested
                        pro.seek(0,0)
                        # phasedata = np.loadtxt(pro,skiprows=skip+self.ncomp*(self.nion+2)+1,max_rows=self.nion)[:,1:1+2*len(self.channels)]
                        il1 = skip+self.ncomp*(self.nion+2)+1
                        il2 = il1+self.nion
                        phasedata = loadtxt(items[il1:il2])[:,1:1+2*len(self.channels)]
                        self.phase[ik-nhse,ib,isp,:,:] = phasedata[:,0::2] + 1j*phasedata[:,1::2] 
        
        pro.close()

    def __str__(self):
        print('nkpt=',self.nkpt)
        print('nband=',self.nband)
        print('nion=',self.nion)
        print('ispin=',self.ispin)
        print('ncomp=',self.ncomp)
        print('channels=',self.channels)
        print('lphase=',self.lphase)
        print('size(projlist)',self.projlist.shape)
        print('size(phase)',self.phase.shape)
        print('size(klist)=',self.klist.shape)
        print('size(kwlist)=',self.kwlist.shape)
        print('size(elist)=',self.elist.shape)
        print('size(occlist)=',self.occlist.shape)
        return "\n"        

class PAWSetting():
    def __init__(self,pos,pot,calcO=False):
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
            x = np.arange(-1,100)/100*pps.Gmax
            recipprojs[iat]=[CubicSpline(x,pps.recipprojs[i],bc_type='natural',extrapolate=False) for i in range(len(pps.recipprojs))]
        self.recipprojs=recipprojs

        realprojs={}
        for iat in pos.elements:
            pps = pot.pps[iat]
            x = pps.projgrid
            realprojs[iat]=[]
            for i in range(len(pps.realprojs)):
                boundary= 0 if pps.ls[i]!=1 else (pps.realprojs[i][1]-pps.realprojs[i][0])/(x[1]-x[0])
                realprojs[iat].append(CubicSpline(x,pps.realprojs[i],bc_type=((1, boundary), (2, 0.0)),extrapolate=False))
        self.realprojs=realprojs


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

        if calcO:   #Oij=<\phi_i|\phi_j> #tOij=<\tilde{\phi_i}|\tilde{\phi_j}>
            Oij={}
            tOij={}
            for iat in pos.elements:
                Oij[iat]=np.zeros([len(ls[iat]),len(ls[iat])])
                tOij[iat]=np.zeros([len(ls[iat]),len(ls[iat])])
                pps = pot.pps[iat]
                for i in range(len(ls[iat])):
                    for j in range(len(ls[iat])):
                        if ls[iat][i]==ls[iat][j]:
                            Oij[iat][i,j]=simpson(pps.aewaves[i][pps.ridx]*pps.aewaves[j][pps.ridx]*pps.r_h*pps.grid[pps.ridx])
                            tOij[iat][i,j]=simpson(pps.pswaves[i][pps.ridx]*pps.pswaves[j][pps.ridx]*pps.r_h*pps.grid[pps.ridx])
            self.Oij=Oij
            self.tOij=tOij

        

        

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
#                 YLM[l][m,:]= sqrt2*((-1)**l)*np.real(data)
#                 YLM[l][-m,:]= sqrt2*((-1)**l)*np.imag(data)
                YLM[l][m,:]= sqrt2*((-1)**m)*np.real(data)
                YLM[l][-m,:]= sqrt2*((-1)**m)*np.imag(data)

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
  
  
pot=POTCAR("POTCAR") 
pos=POSCAR('POSCAR')
wav=WAVECAR('WAVECAR')
paw=PAWSetting(pos,pot,calcO=True)
print("reading PROCAR")
pro=vasp.procar('PROCAR')

# for spin non-collinear calculation, check total contribution
print("check lm-resolved total contribution")
keys = list(up.keys())
res = np.zeros([pro.nband,6,16])
isp=0
for ik in range(pro.nkpt):
    proj=calcproj(wav,paw,ik,isp,range(pro.nband))
    for ia in range(6):
        for i in range(16):
            res[:,ia,i]=np.diag((proj.conj()@(Projector(paw,[ia],up[keys[i]])+Projector(paw,[ia],dn[keys[i]]))@proj.T)).real
    if not np.allclose(pro.projlist[ik,:,0,:,:],res,atol=1e-3): # 
        print("kpt:",ik," total contribution mismatch")
        
# for spin non-collinear calculation, check Sz contribution
print("check lm-resolved Sz contribution")
keys = list(up.keys())
res = np.zeros([pro.nband,6,16])
isp=0
for ik in range(pro.nkpt):
    proj=calcproj(wav,paw,ik,isp,range(pro.nband))
    for ia in range(6):
        for i in range(16):
            res[:,ia,i]=np.diag((proj.conj()@(Projector(paw,[ia],up[keys[i]])-Projector(paw,[ia],dn[keys[i]]))@proj.T)).real
    if not np.allclose(pro.projlist[ik,:,3,:,:],res,atol=1e-3): # 
        print("kpt:",ik," Sz contribution mismatch")

print("check phase information")
s = slice(0,1)
p = slice(1,4)
d = slice(4,9)
f = slice(9,16)
ang = {0:s,1:p,2:d,3:f}
isp=0
res = np.zeros([pro.nband,6,len(pro.channels)],dtype=complex)
for ik in range(pro.nband):
    res[:,:]=0
    for iat in range(len(paw.atomlabel)):
        atom = paw.atomlabel[iat]
        L_at = np.array(paw.ls[atom])
        for l in range(4):
            if not l in L_at:
                continue
            idx = np.where(L_at==l)[0]
            EV=diag_and_sort(paw.Oij[atom][idx[:,np.newaxis],idx[np.newaxis,:]])
            proj=calcproj(wav,paw,ik,isp,range(pro.nband))
            RANGE = np.arange(paw.lmmax[atom])
            for spn in range(wav.spinor):
                data=proj[:,spn*paw.nchannel:(spn+1)*paw.nchannel][:,paw.atomidx[iat]]\
                      [:,np.array([RANGE[sl] for sl in np.array(paw.lmidx[atom])[idx]]).T]
                phas = data@EV
                maxidx=np.argmax(np.linalg.norm(phas,axis=1),axis=1)
                res[:,iat,ang[l]][:,np.arange(-l-1,l)]+=phas[np.arange(pro.nband),:,maxidx]
    if not np.allclose(pro.phase[ik,:,0,:,:],res,atol=1e-3): 
        print("kpt:",ik," phase information mismatch")
                 
        
