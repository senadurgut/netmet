sumBranches = ['EtSum_' + var for var in ['pt', 'etSumType', 'bx']]
objectBranches = ['pt', 'eta', 'phi', 'bx']
puppiMETBranches = ['PuppiMET_pt', 'PuppiMET_phi']
muonBranches = ['Muon_' + var for var in ['pt', 'phi', 'isPFcand']]
recoBranches = ['PV_npvsGood']

# github.com/cms-sw/cmssw/blob/master/DataFormats/L1Trigger/interface/EtSum.h
sums = {'ett': 0, 'htt': 1, 'met': 2, 'mht': 3, 'metx': 4, 'mety': 5, 'htx': 6, 'hty': 7, 'methf': 8, 'etxhf': 9, 'etyhf': 10, 
        'mbhfp0': 11, 'mbhfm0': 12, 'mbhfp1': 13, 'mbhfm1': 14, 'etthf': 15, 'ettem': 16, 'htthf': 17, 'htxhf': 18, 'htyhf': 19, 
        'mhthf': 20, 'ntt': 21}
