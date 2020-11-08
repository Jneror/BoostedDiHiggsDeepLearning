import os

col_names = {
    "region": "m_region",
    "tag": "m_FJNbtagJets",
    "weight": "EventWeight",
}

selected_features = [
    'm_FJpt', 'm_FJeta', 'm_FJphi', 'm_FJm', 'm_DTpt', 'm_DTeta', 'm_DTphi', 'm_DTm',
    'm_dPhiFTwDT', 'm_dRFJwDT', 'm_dPhiDTwMET', 'm_MET', 'm_hhm', 'm_bbttpt',
]

tags = [0, 1, 2]

package_directory = os.path.dirname(os.path.abspath(__file__))