#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import datetime
from astropy.time import Time
from fermipy.gtanalysis import GTAnalysis
from fermipy.plotting import ROIPlotter
import glob,os,shutil,sys

def analyze(timebin):
    #time in config file is in MET
    #Setup gta class
    #for i in range(1,16):
    mysource='4FGL J1806.8+6949'

    print("STARTING {}th Time Bin".format(timebin))
    folderout="dataAll/data{}".format(timebin)
    if not os.path.exists(folderout):
        os.mkdir(folderout) 
        shutil.copyfile("dataAll/events.txt","{}/events.txt".format(folderout))
        shutil.copyfile("dataAll/SC00.fits","{}/SC00.fits".format(folderout))

    gta = GTAnalysis('config{}.yaml'.format(timebin),logging={'verbosity' : 3}) 
    gta.setup() #creat all counts map and fits and download all files needed ...

    #now lets optimize before doing fit to make sure if we can get rid of sources that are not strong or don't have enough counts
    opt1 = gta.optimize()
    print("ROI after optimization")
    gta.print_roi()

    deleted_sources = gta.delete_sources(minmax_ts=[-1,3])
    deleted_sources = gta.delete_sources(minmax_npred=[0,3])
    print("ROI after deleting the weak sources optimization")
    gta.print_roi()

    # Free Normalization of all Sources within 3 deg of ROI center
    gta.free_sources(distance=3.0,pars='norm')

    # Free sources with TS > 10
    gta.free_sources(minmax_ts=[10,None],pars='norm')

    # Free all parameters of isotropic and galactic diffuse components
    if gta.roi.sources[-2]['name']=="isodif":
        gta.free_source('isodiff')
    
    gta.free_source('galdiff')
    

    #Free shape parameters of my target
    gta.free_source(mysource)#, pars=['Index','Prefactor','Scale'])

    fit1 = gta.fit(optimizer='NEWMINUIT',reoptimize=True, max_iter=1000)

    print("ROI after fitting")
    gta.print_roi()
    print(gta.roi[mysource])


    #save roi
    gta.write_roi('fit1',make_plots=True)

    #now creat residual map
    fixed_sources = gta.free_sources(free=False)
    resid = gta.residmap('3C 371',model={'SpatialModel' : 'PointSource', 'Index' : 2.0})#usually we put the index from the best fit here

    o = resid
    plt.clf()
    fig = plt.figure(figsize=(14,6))
    ROIPlotter(o['excess'],roi=gta.roi).plot(vmin=-50,vmax=50,subplot=122,cmap='RdBu_r')
    plt.gca().set_title('Excess Counts')
    plt.savefig("{}/residualmap.png".format(folderout))
    #plt.show()

    n, _, _ = plt.hist( resid["sigma"].data.flatten(), bins=np.linspace(-3,3,61), \
                                density=True, histtype="step", label = "Data"  )
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    x = np.linspace(-3,3, 601)
    y = np.exp(-x**2 / 2.0) #gaussian with mean 0 and width 1
    norm = n.sum() * 0.1 / (y.sum() * 0.01) #norm factor
    plt.plot(x, norm*y, label = "Expectation")
    plt.xlabel("Significance")
    plt.ylabel("Frequency [a.u.]")
    plt.title("Significance distribution")
    plt.legend()
    plt.savefig("{}/Significance.png".format(folderout))

    #generat TS map
    tsmap = gta.tsmap('3C 371',model={'SpatialModel' : 'PointSource', 'Index' : 2.0})

    o = tsmap
    fig = plt.figure(figsize=(6,6))
    ROIPlotter(o['sqrt_ts'],roi=gta.roi).plot(vmin=0,vmax=5,levels=[3,5,7,9],subplot=111,cmap='magma')
    plt.gca().set_title('sqrt(TS)')
    plt.savefig("{}/resTS.png".format(folderout))
    #plt.show()

    #butterfly plot
    source = gta.roi.get_source_by_name(mysource) #model for the source
    E = np.array(source['model_flux']['energies'])
    dnde = np.array(source['model_flux']['dnde']) #model plot, Central value of spectral band (cm−2 s−1 MeV−1)
    dnde_hi = np.array(source['model_flux']['dnde_hi']) #+1 sigma plot upper 1-sigma bound of spectral band (cm−2 s−1 MeV−1)
    dnde_lo = np.array(source['model_flux']['dnde_lo']) #-1 sigma plot

    ene_arr= gta.log_energies[::3]  ## create SED with a 1/4 of the bins stated in config-file
    sed = gta.sed(mysource, make_plots=True,loge_bins=ene_arr)

    plt.loglog(E, (E**2)*dnde, 'k--')
    plt.loglog(E, (E**2)*dnde_hi, 'k')
    plt.loglog(E, (E**2)*dnde_lo, 'k')
    plt.errorbar(np.array(sed['e_ctr']),
                 sed['e2dnde'], 
                 yerr=sed['e2dnde_err'], fmt ='o') #1sigma error on E^2dN/dE evaluated from likelihood curvature.
    plt.xlabel('E [MeV]')
    plt.ylabel(r'E$^{2}$ dN/dE [MeV cm$^{-2}$ s$^{-1}$]')
    plt.savefig("{}/butterfly.png".format(folderout))
    #plt.show()

    #butterfly plot
    plt.loglog(E, (E**2)*dnde, 'k--')
    plt.loglog(E, (E**2)*dnde_hi, 'k')
    plt.loglog(E, (E**2)*dnde_lo, 'k')

    TS_thresh=23
    measurements=(sed['ts'] >= TS_thresh)
    limits = (sed['ts'] < TS_thresh)

    plt.errorbar(sed['e_ctr'][measurements],
                 sed['e2dnde'][measurements], 
                 yerr=sed['e2dnde_err'][measurements], fmt ='o') #1sigma error on E^2dN/dE evaluated from likelihood curvature.
    plt.errorbar(np.array(sed['e_ctr'][limits]), 
             sed['e2dnde_ul95'][limits], yerr=0.2*sed['e2dnde_ul95'][limits], 
                 fmt='o', uplims=True)
    plt.xlabel('E [MeV]')
    plt.ylabel(r'E$^{2}$ dN/dE [MeV cm$^{-2}$ s$^{-1}$]')
    plt.savefig("{}/butterflyupp.png".format(folderout))
    #plt.show()

    #save average photon flux in energy range that we defined in config file
    np.savetxt("{}/flux.txt".format(folderout),[gta.roi.sources[0]._data['flux'],gta.roi.sources[0]._data['flux_err'],gta.roi.sources[0]['ts']], newline=" ")

    
if __name__=='__main__': 
    timebin=int(sys.argv[1])
    #print("")
    analyze(timebin)
    
    