import numpy as np, uproot, awkward as ak, matplotlib.pyplot as plt
import branches as branches
import math as math 
#adapted from https://github.com/jaimeleonh/L1NetMET/blob/main/netMET/utils/plotting.py 


#l1MET = X['methf_0_hwPt'] / 2
#L1MET_bkg = bkg['methf_0_hwPt'] / 2
#l1NetMET_bkg = Yp_bkg.flatten()
plot_dir = '/Users/sena/grad_school/Research/netmet/plots/'

def get_netmet(L1EmulEtSum_hwPt, L1EmulEtSum_etSumType):
    netmet =ak.to_numpy(ak.flatten(L1EmulEtSum_hwPt[L1EmulEtSum_etSumType == 9]))
    netmet = netmet / 2 # convert to GeV 
    return netmet 

def get_l1met(L1EmulEtSum_hwPt, L1EmulEtSum_etSumType):
    l1met = ak.to_numpy(ak.flatten(L1EmulEtSum_hwPt[L1EmulEtSum_etSumType == 8]))
    l1met = l1met / 2 # convert to GeV
    return l1met

def get_puppimet(file_path):
    data = uproot.open(file_path    + ':Events')
    puppiMET = data.arrays(branches.puppiMETBranches, library='ak')
    puppiMET = ak.with_field(puppiMET, puppiMET['PuppiMET_pt']*np.cos(puppiMET['PuppiMET_phi']), "PuppiMET_ptx")
    puppiMET = ak.with_field(puppiMET, puppiMET['PuppiMET_pt']*np.sin(puppiMET['PuppiMET_phi']), "PuppiMET_pty")


    #get offline muons
    muons = data.arrays(branches.muonBranches)
    muons = muons[muons["Muon_isPFcand"] == 1]
    del muons["Muon_isPFcand"]
    muons = ak.with_field(muons, muons['Muon_pt']*np.cos(muons['Muon_phi']), "Muon_ptx")
    muons = ak.with_field(muons, muons['Muon_pt']*np.sin(muons['Muon_phi']), "Muon_pty")
    
    #substract muons 
    puppiMET_noMu = ak.copy(puppiMET)
    puppiMET_noMu['PuppiMET_ptx'] = puppiMET['PuppiMET_ptx'] + np.sum(muons['Muon_ptx'], axis=1)
    puppiMET_noMu['PuppiMET_pty'] = puppiMET['PuppiMET_pty'] + np.sum(muons['Muon_pty'], axis=1)
    puppiMET_noMu['PuppiMET_pt'] = np.sqrt(puppiMET_noMu['PuppiMET_ptx']**2 + puppiMET_noMu['PuppiMET_pty']**2)
    del puppiMET['PuppiMET_phi'], puppiMET['PuppiMET_ptx'], puppiMET['PuppiMET_pty']
    del puppiMET_noMu['PuppiMET_phi'], puppiMET_noMu['PuppiMET_ptx'], puppiMET_noMu['PuppiMET_pty']

    return ak.to_numpy(puppiMET['PuppiMET_pt']), ak.to_numpy(puppiMET_noMu['PuppiMET_pt'])

def efficiency(data_on, data_off, threshold, binwidth, xmax):
    
    num = ak.zip({'on': data_on})
    num['off'] = data_off
    num = num[num['on'] > threshold]

    numHist = np.histogram(num['off'], bins=int(xmax/binwidth), range=(0,xmax))[0] #puppi
    denomHist  = np.histogram(data_off, bins=int(xmax/binwidth), range=(0,xmax))[0] #l1met 
    effs = numHist/denomHist
    errors = [math.sqrt((((k + 1) * (k + 2)) / ((n + 2) * (n + 3))) - (((k + 1) * (k + 1)) / ((n + 2) * (n + 2))))
        for (k, n) in zip(numHist, denomHist)]
    xvals = [x+(binwidth/2) for x in range(0,xmax,binwidth)]

    return effs, xvals, errors


def getThreshForRate(rates, bins, target_rate):
    
    finalThresh = 0
    for rate, thresh in zip(rates, range(bins)):
            if rate < target_rate:
                finalThresh = thresh
                break
    return finalThresh

def plot_rate(sig_path, bkg_path, title, include_puppi=False):
    import math as math
    #normally this is done on background samples in Jaimes code, unlike resolution plots
    l1_met_threshold = 80 
    f = uproot.open(sig_path + ':Events')
    f_bkg = uproot.open(bkg_path + ':Events')
    L1EmulEtSum_hwPt = f["L1EmulEtSum_hwPt"].array(library='ak')
    L1EmulEtSum_etSumType = f["L1EmulEtSum_etSumType"].array(library='ak')
    L1EmulEtSum_hwPt_bkg = f_bkg["L1EmulEtSum_hwPt"].array(library='ak')
    L1EmulEtSum_etSumType_bkg = f_bkg["L1EmulEtSum_etSumType"].array(library='ak')
    netmet = get_netmet(L1EmulEtSum_hwPt, L1EmulEtSum_etSumType) # l1NetMET
    netmet_bkg = get_netmet(L1EmulEtSum_hwPt_bkg, L1EmulEtSum_etSumType_bkg) # l1NetMET_bkg
    l1met = get_l1met(L1EmulEtSum_hwPt, L1EmulEtSum_etSumType) # l1MET
    l1met_bkg = get_l1met(L1EmulEtSum_hwPt_bkg, L1EmulEtSum_etSumType_bkg) # l1MET_bkg
    puppimet, puppimet_noMu = get_puppimet(sig_path)
    title = 'Rate with 2025C ZMu data, ' + title

    ax = plt.subplot()
    xrange = [0, 200]
    bins = xrange[1]
    rateHist = plt.hist(l1met_bkg, bins=bins, range=xrange, histtype='step', label='L1 MET Rate', cumulative=-1, log=True,
                weights=[1./l1met.shape[0] for i in range(l1met.shape[0])])
    rateHist_netMET = plt.hist(netmet_bkg, bins=bins, range=xrange, histtype='step', label='L1 NET MET Rate', cumulative=-1, log=True, 
                weights=[1./netmet.shape[0] for i in range(netmet.shape[0])])
    if include_puppi:
        rateHist_puppiMET = plt.hist(puppimet, bins=bins, range=xrange, histtype='step', label='Puppi MET Rate', cumulative=-1, log=True,
                weights=[1./puppimet.shape[0] for i in range(puppimet.shape[0])])
        title += ' with Puppi MET'
    
    plt.title(title)
    plt.xlabel('MET [GeV]')
    plt.ylabel('Normalized Events')
    #calculate threshold 
    l1met_fixed_rate = rateHist[0][int(l1_met_threshold) * int((xrange[1] / bins))]
    netMET_thresh = getThreshForRate(rateHist_netMET[0], bins, l1met_fixed_rate)
    ax.scatter(l1_met_threshold, l1met_fixed_rate, color='red', label='L1 MET Threshold = ' + str(l1_met_threshold))
    ax.scatter(netMET_thresh, l1met_fixed_rate, color='blue', label='NET MET Threshold = ' + str(netMET_thresh))
    plt.axhline(y=l1met_fixed_rate, color='purple', linestyle='--', label='Fixed Rate = ' + str(l1met_fixed_rate)[:4])
    print('L1 MET Threshold: ', l1_met_threshold)
    print('NET MET Threshold: ', netMET_thresh)
    print('L1 MET Fixed Rate: ', l1met_fixed_rate)
    #write netmet threshold to a text file 
    #with open('/Users/sena/grad_school/Research/netmet/netMET_thresholds.txt', 'a') as f:
        #f.write(str(netMET_thresh) + '\n')
        #f.write(str(l1_met_threshold) + '\n')
        #f.write(str(l1met_fixed_rate) + '\n')
    
    plt.legend()
    plt.savefig(plot_dir+ title.replace(' ', '_') + '.pdf')
    plt.show()


def plot_efficiency(file_path, title):
    l1_met_threshold = 80 
    f = uproot.open(file_path + ':Events')
    L1EmulEtSum_hwPt = f["L1EmulEtSum_hwPt"].array(library='ak')
    L1EmulEtSum_etSumType = f["L1EmulEtSum_etSumType"].array(library='ak')
    netmet = get_netmet(L1EmulEtSum_hwPt, L1EmulEtSum_etSumType) # l1NetMET_bkg
    l1met = get_l1met(L1EmulEtSum_hwPt, L1EmulEtSum_etSumType) # l1MET_bkg
    puppimet_pt, puppimet_pt_noMu = get_puppimet(file_path)
    #read netmet threshold from the text file
    l1_met_threshold = 90
    netMET_thresh = 120 
    l1met_fixed_rate = 0.03661971830985916
    

    ### Efficiency ###

    fig = plt.figure()
    ax = plt.subplot()
    eff_data, xvals, eff_errors = efficiency(l1met, puppimet_pt, l1_met_threshold, 10, 400)
    print('eff_data: ', eff_data)
    plt.axhline(0.95, linestyle='--', color='black')
    plt.errorbar(xvals, eff_data, eff_errors, label="L1 MET > " + str(l1_met_threshold),
            marker='o', capsize=7, linestyle='none')
    ax.set_xlabel('PUPPI MET No Mu [GeV]')
    ax.set_ylabel('Efficiency')
    netMET_eff_data, _, netMET_eff_errors = efficiency(netmet, puppimet_pt, netMET_thresh, 10, 400)
    plt.errorbar(xvals, netMET_eff_data, netMET_eff_errors, label="L1 NETMET > " + str(netMET_thresh),
            marker='o', capsize=7, linestyle='none')
    plt.legend()
    plt.title(title)
    plt.savefig('/Users/sena/grad_school/Research/netmet/plots/' + title.replace(' ', '_') + '_efficiency.pdf')
    plt.show()


