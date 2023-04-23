import uproot
import argparse
import awkward as ak
import hist
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import re

from utils import check_outputpath

name_title_dicts = {
    "diffZ": r'$\Delta Z[mm]$',
}

def calculate_differenceZ(
        vtx_vz : ak.highlevel.Array
        ):
    
    differenceZ = []
    for vtx_vz_event in vtx_vz:
        distance_all_pairs = ak.flatten(vtx_vz_event[:, None] - vtx_vz_event)
        differenceZ.append(distance_all_pairs[distance_all_pairs!=0])
    
    return differenceZ

def plot_hist(_hist:hist.hist.Hist, output_path=None, output_name=None, pu_number=None, is_truth = False):
    fig, ax = plt.subplots()
    bin_contents, bin_edges = _hist.to_numpy()
    ax.stairs(values=bin_contents, edges=bin_edges)
    ax.set_ylabel('Number of Vertices')
    ax.set_xlabel(name_title_dicts[_hist.axes.name[0]]) # assume it's 1d, so [0] is used. otherwise it's a tuple 
    if is_truth: 
        identifier = "Truth"
    else:
        identifier = "AMVF"
    ax.set_title('Distribution of ' + r'$\Delta Z[mm]$' + f' for {identifier} PU{pu_number} Sample')

    fig.savefig(output_path / output_name)

def plot_hist_compare(
        truth_hist:hist.hist.Hist, 
        reco_hist:hist.hist.Hist, 
        output_path=None, output_name=None, pu_number=None):
    
    bin_contents_truth, bin_edges_truth = truth_hist.to_numpy()
    bin_contents_reco, bin_edges_reco = reco_hist.to_numpy()
    assert (bin_edges_truth == bin_edges_reco).all()

    fig, ax = plt.subplots()
    ax.stairs(values=bin_contents_truth, edges=bin_edges_truth, label='Truth')
    ax.stairs(values=bin_contents_reco, edges=bin_edges_reco, label='AMVF')
    ax.legend()
    ax.set_ylabel('Number of Vertices')
    ax.set_xlabel(name_title_dicts[truth_hist.axes.name[0]])
    ax.set_title('Distribution of ' + r'$\Delta Z[mm]$' + f' for PU{pu_number} Sample')

    fig.savefig(output_path / output_name)
    pass

def performance_differenceZ(
        input_root_path:Path,
        output_path:Path,
        reco_tree_name='Reco_Vertex',
        reco_vtz_vz_name='reco_vtx_vz',
        truth_tree_name='Truth_Vertex_PV_Selected',
        truth_vtx_vz_name='truth_vtx_vz'
        ):
    
    output_path = output_path / "diffZ"
    check_outputpath(output_path)
    ### Reading the information from TTree
    root_file =uproot.open(input_root_path)
    file_name = input_root_path.stem # for example, vertexperformance_AMVF_pu100.root
    pu_search_pattern = r'pu(.+)$'
    pu_number = re.search(pu_search_pattern, file_name).group((1))

    reco_tree = root_file[reco_tree_name]
    reco_vtx_vz = reco_tree[reco_vtz_vz_name].array(library="ak")
    truth_tree = root_file[truth_tree_name]
    truth_vtx_vz = truth_tree[truth_vtx_vz_name].array(library="ak")

    ### Plot the reco vtx
    hist_differenceZ = hist.Hist(hist.axis.Regular(bins=50, start=-5, stop=5, name="diffZ"))
    differenceZ = calculate_differenceZ(reco_vtx_vz)
    # flatten the list
    differenceZ = np.concatenate(differenceZ)
    hist_differenceZ.fill(differenceZ)
    plot_hist(hist_differenceZ, output_path, output_name=f'diffZ_amvf_pu{pu_number}', pu_number = pu_number, is_truth=False)

    ### Plot the truth vtx 
    differenceZ_truth = calculate_differenceZ(truth_vtx_vz)
    hist_differenceZ_truth = hist_differenceZ.copy()
    hist_differenceZ_truth.reset()
    differenceZ_truth = np.concatenate(differenceZ_truth)
    hist_differenceZ_truth.fill(differenceZ_truth)
    plot_hist(hist_differenceZ_truth, output_path, output_name=f'diffZ_truth_pu{pu_number}', pu_number = pu_number, is_truth=True)
    
    ### Plot the truth and reco together 
    plot_hist_compare(truth_hist=hist_differenceZ_truth, reco_hist=hist_differenceZ, output_path=output_path,
                      output_name=f'diffZ_comp_pu{pu_number}', pu_number=pu_number)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root-path', help='the input root file path', type=str)
    parser.add_argument('--output-path', help='the output path for performance plot', type=str)
    args = parser.parse_args()

    input_root_path = Path(args.input_root_path)
    output_path = args.output_path

    if output_path is None:
        output_path = input_root_path.parent
    else:
        output_path = Path(args.output_path)

    performance_differenceZ(input_root_path, output_path)