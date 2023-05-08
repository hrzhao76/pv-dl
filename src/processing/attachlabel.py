# flake8: noqa: C901
import uproot
import pandas as pd
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import ROOT
from array import array
from tqdm import tqdm
import argparse

root_path = "/global/cfs/projectdirs/atlas/hrzhao/sample/SPVCNN/acts_v24/IVF/n10k_npu50_IVF_04302023/performance_vertexing.root"
preds_folder = "/global/cfs/projectdirs/atlas/hrzhao/spvnas-dev/test_IVF_10k/preds/tbeta=0.75_td=0.4_pkls/"
output_file_name = "vertex_spvcnn_decorated.root"


def create_branchesdict(branch_list: list):
    return_dict = dict.fromkeys(branch_list)
    for branch_name in branch_list:
        if branch_name == "event_nr":
            return_dict[branch_name] = array("i", [0])
            continue
        if branch_name.startswith("reco_trk_reco_vtx_"):
            return_dict[branch_name] = ROOT.std.vector[ROOT.std.vector["double"]]()
            continue

        return_dict[branch_name] = ROOT.std.vector["double"]()

    return_dict["reco_trk_spvcnn_vtx_ins_idx"] = ROOT.std.vector["double"]()
    return_dict["reco_trk_spvcnn_vtx_sem_idx"] = ROOT.std.vector["double"]()

    return return_dict


def attach_spvcnn_label_vertex(root_file_path, spvcnn_output_folder, output_path, output_name):
    root_file_path = Path(root_file_path)

    output_path = Path(output_path) if output_path is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = root_file_path.parent.as_posix()

    file = uproot.open(root_file_path)
    spvcnn_output_files = sorted(Path(spvcnn_output_folder).glob("*.pkl"))
    if len(spvcnn_output_files) == 0:
        raise ValueError(f"Cannot find any spvcnn pred output files in {spvcnn_output_folder}")

    ML_tracks = file["ML_tracks"].arrays()
    output_file_path = Path(output_path) / output_name
    print(f"Writing to {output_file_path.as_posix()}")

    outputFile = ROOT.TFile(output_file_path.as_posix(), "recreate")
    outputTree = ROOT.TTree("spvcnn_decorated", "spvcnn_decorated_tree")

    branches_dict = create_branchesdict(ML_tracks.fields)

    for k, v in branches_dict.items():
        if k == "event_nr":
            outputTree.Branch(k, v, f"{k}/I")
            continue
        outputTree.Branch(k, v)

    event_idx_pattern = re.compile(r"pred_event_(\d+).pkl")

    # Loop over events
    for file in tqdm(spvcnn_output_files):
        for branch_name, branch in branches_dict.items():
            if branch_name == "event_nr":
                continue
            branch.clear()

        matched = re.match(event_idx_pattern, file.name)
        if matched:
            event_idx = int(matched.group(1))
        else:
            raise ValueError(f"Cannot find event index in file name {file.name}")

        event = ML_tracks[event_idx]
        spvcnn_pd = pd.read_pickle(file)

        for branch_name, branch in branches_dict.items():
            if branch_name == "event_nr":
                branch[0] = event.event_nr
                continue

            # for std::vector<std::vector<double>>
            if branch_name.startswith("reco_trk_reco_vtx_"):
                branch.resize(len(event[branch_name]))
                for reco_trk_idx, reco_trk_reco_vtx_values in enumerate(event[branch_name]):
                    for value in reco_trk_reco_vtx_values:
                        branch[reco_trk_idx].push_back(value)
                continue

            # for std::vector<double>
            if branch_name in event.fields:
                for value in event[branch_name]:
                    branch.push_back(value)
                continue

            # for reco_trk_spvcnn_vtx_ins_idx and reco_trk_spvcnn_vtx_sem_idx
            if branch_name == "reco_trk_spvcnn_vtx_ins_idx":
                for reco_trk_idx, reco_trk_spvcnn_vtx_ins_idx in enumerate(
                    spvcnn_pd["pred_instance_labels"]
                ):
                    branch.push_back(reco_trk_spvcnn_vtx_ins_idx)
                continue

            if branch_name == "reco_trk_spvcnn_vtx_sem_idx":
                for reco_trk_idx, reco_trk_spvcnn_vtx_sem_idx in enumerate(
                    spvcnn_pd["pred_semantic_labels"]
                ):
                    branch.push_back(reco_trk_spvcnn_vtx_sem_idx)
                continue

        outputTree.Fill()

    outputTree.Write()
    outputFile.Close()

    ### do a test of the output file and the original file to see if they are the same
    random_file_idx = np.random.randint(low=0, high=len(spvcnn_output_files), size=1)[0]
    random_file = spvcnn_output_files[random_file_idx]
    matched = re.match(event_idx_pattern, random_file.name)
    if matched:
        event_idx = int(matched.group(1))
    else:
        raise ValueError(f"Cannot find event index in file name {file.name}")

    original_event = ML_tracks[event_idx]
    decorated_event = uproot.open(output_file_path)["spvcnn_decorated"].arrays()[random_file_idx]

    check_fields = []
    for field in original_event.fields:
        check_fields.append(np.allclose(original_event[field], decorated_event[field]))

    if np.all(check_fields):
        print("Test one file, and the original and decorated file are the same! Done. ")
    else:
        raise ValueError("The original and decorated files are not the same")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root-path", help="the input root file path", type=str, default=root_path)
    parser.add_argument(
        "--input-preds-folder", help="the input spvcnn prediction folder", type=str, default=preds_folder
    )
    parser.add_argument("--output-path", help="the output folder path", type=str)
    parser.add_argument("--output-name", help="the output file name", type=str, default=output_file_name)

    args = parser.parse_args()

    if args.output_path is None:
        output_path = Path(args.input_preds_folder).parent.as_posix()

    attach_spvcnn_label_vertex(
        root_file_path=args.input_root_path,
        spvcnn_output_folder=args.input_preds_folder,
        output_path=output_path,
        output_name=args.output_name,
    )
