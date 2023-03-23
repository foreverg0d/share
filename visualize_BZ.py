import numpy as np, numpy.linalg as npl
from scipy.spatial import Voronoi
from itertools import product
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib as mpl
mpl.rcParams['font.size'] = 12.

def read_poscar(poscar, species=None):
    poscar = open(poscar,'r')
    title = poscar.readline().strip()
    scale = float(poscar.readline().strip())
    s = float(scale)
    lattice_vectors = [[ float(v) for v in poscar.readline().split() ],
            [ float(v) for v in poscar.readline().split() ],
            [ float(v) for v in poscar.readline().split() ]]
    lattice_vectors = np.array(lattice_vectors)
    reciprocal_lattice_vectors= np.linalg.inv(lattice_vectors).T
    reciprocal_lattice_vectors=reciprocal_lattice_vectors*np.pi*2
    return reciprocal_lattice_vectors

def read_high_symmetry_points():                  
    f = open("HIGH_SYMMETRY_POINTS",'r') 
    fa=f.readline()                      
    n=0                                  
    hpts=[]                              
    while True:                          
        fa=f.readline()                  
        hpts.append(fa)                  
        n=n+1                            
        if fa.find('use') > 0:           
            break     
    f.close()                             
    kpts=[]                              
    klabels=[]                            
    for i in range(0,n-3):               
        kpts.append(hpts[i].split()[0:3])
        klabels.append(hpts[i].split()[3])
    kpts=np.array(kpts,dtype=np.float64)
    return kpts, klabels                

def is_greek_alphabets(klabels):
    Greek_alphabets=['Alpha','Beta','Gamma','Delta','Epsilon','Zeta','Eta','Theta', 'Iota','Kappa','Lambda','Mu','Nu','Xi','Omicron','Pi','Rho','Sigma','Tau','Upsilon','Phi','Chi','Psi','Pega']
    group_labels=[]
    for i in range(len(klabels)): 
        klabel=klabels[i] 
        for j in range(len(Greek_alphabets)):
            if (klabel.find(Greek_alphabets[j].upper())>=0):
                latex_exp=r''+'$\\'+str(Greek_alphabets[j])+'$'
                klabel=klabel.replace(str(Greek_alphabets[j].upper()),str(latex_exp))
        if (klabel.find('_')>0):
           n=klabel.find('_')
           klabel=klabel[:n]+'$'+klabel[n:n+2]+'$'+klabel[n+2:]
        group_labels.append(klabel)
    klabels=group_labels
    return klabels

def read_kpath():
  kpath=np.loadtxt("KPATH.in", dtype=np.string_,skiprows=4)
  #print(kpath)
  kpath_labels = kpath[:,3].tolist()
  kpath_labels = [i.decode('utf-8','ignore') for i in kpath_labels]
  for i in range(len(kpath_labels)):
           if kpath_labels[i]=="Gamma":
                   kpath_labels[i]=u"Γ"
  kpaths=np.zeros((len(kpath_labels),3),dtype=float)
  for i in range(len(kpath_labels)):
      kpaths[i,:]=\
      [float(x) for x in kpath[i][0:3]]
  return kpath_labels, kpaths

def get_Wigner_Seitz_BZ(lattice_vectors):
# Inspired by http://www.thp.uni-koeln.de/trebst/Lectures/SolidState-2016/wigner_seitz_3d.py 
# Inspired by https://github.com/QijingZheng/VASP_FermiSurface/blob/master/fs.py 
    latt = []
    prefactors = [0., -1., 1.]
    for p in prefactors:
        for u in lattice_vectors:
            latt.append(p * u)
    lattice = []
    for vs in product(latt, latt, latt):
        a = vs[0] + vs[1] + vs[2]
        if not any((a == x).all() for x in lattice):
            lattice.append(a)
    voronoi = Voronoi(lattice)
    bz_facets = []
    bz_ridges = []
    bz_vertices = []
    for pid, rid in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        if(pid[0] == 0 or pid[1] == 0):
            bz_ridges.append(voronoi.vertices[np.r_[rid, [rid[0]]]])
            bz_facets.append(voronoi.vertices[rid])
            bz_vertices += rid
    bz_vertices = list(set(bz_vertices))
    return voronoi.vertices[bz_vertices], bz_ridges, bz_facets

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def visualize_BZ_matplotlib(points,ridges,facets,reciprocal_lattice_vectors,kpts,klabels,kpaths):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    basis_vector_clrs = ['r', 'g', 'b']
    basis_vector_labs = ['x', 'y', 'z']
    for ii in range(3):
        arrow = Arrow3D([0,reciprocal_lattice_vectors[ii, 0]], [0,reciprocal_lattice_vectors[ii, 1]], [0,reciprocal_lattice_vectors[ii, 2]],
                color="Cyan", mutation_scale=20,lw=1.5,arrowstyle="-|>")
        ax.add_artist(arrow)
        ax.text(reciprocal_lattice_vectors[ii, 0], reciprocal_lattice_vectors[ii, 1],reciprocal_lattice_vectors[ii, 2],
                basis_vector_labs[ii],size=16)
        for ir in ridges:
            ax.plot(ir[:, 0], ir[:, 1], ir[:, 2], color='k', lw=1.5,alpha=0.5)
    for i in range(len(klabels)):
        kpt=np.dot(kpts[i,:],reciprocal_lattice_vectors)
        ax.scatter(kpt[0], kpt[1], kpt[2],c='k', marker='o',s=20,alpha=0.8)     
        ax.text(kpt[0], kpt[1], kpt[2],klabels[i],size=16)
    for i in range(kpaths.shape[0]):
        kpaths[i,:]=np.dot(kpaths[i,:],reciprocal_lattice_vectors)
    for i in range(0,kpaths.shape[0],2):
        arrow = Arrow3D([kpaths[i,0],kpaths[i+1,0]],[kpaths[i,1],kpaths[i+1,1]],[kpaths[i,2],kpaths[i+1,2]],mutation_scale=20,lw=1,arrowstyle="->", color="gray")
        ax.add_artist(arrow)
    ax.set_axis_off()
    ax.view_init(elev=12, azim=23)
    plt.savefig('Brillouin_Zone.png',dpi=300)
    plt.show()
    
def welcome():
    print('')
    print('+---------------------------------------------------------------+')
    print('| A VASPKIT Plugin to Visualize Brillouin Zone Using Matplotlib |')
    print('|           Written by Vei WANG (wangvei@icloud.com)            |')
    print('+---------------------------------------------------------------+')
    print('')
    
if __name__ == "__main__":   
   welcome()
   reciprocal_lattice_vectors=read_poscar('POSCAR')   
   lattice_vectors=[np.array(reciprocal_lattice_vectors[0,:]),np.array(reciprocal_lattice_vectors[1,:]),np.array(reciprocal_lattice_vectors[2,:])]
   kpts,klabels=read_high_symmetry_points()
   klabels=is_greek_alphabets(klabels)
   kpath_labels,kpaths=read_kpath()
   points, ridges, facets = get_Wigner_Seitz_BZ(lattice_vectors)
   visualize_BZ_matplotlib(points, ridges, facets, reciprocal_lattice_vectors,kpts,klabels,kpaths)
