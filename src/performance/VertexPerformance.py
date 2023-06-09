from utils import name_title_dicts
from enum import Enum
import uproot
from typing import Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import hist
import mplhep as hep
import ROOT
import logging
from tqdm import tqdm
import pandas as pd

from utils.io import check_inputpath, check_outputpath, logging_setup
from utils import (
    vxMatchWeight,
    cutMinTruthRecoRadialDiff,
    VertexMatchType,
    HardScatterType,
)


def calculate_component(track_at_cluster):
    avg_vtx_x0 = np.average(track_at_cluster["x0"], weights=1.0 / (track_at_cluster["err_d0"] ** 2))
    avg_vtx_y0 = np.average(track_at_cluster["y0"], weights=1.0 / (track_at_cluster["err_d0"] ** 2))
    avg_vtx_z0 = np.average(track_at_cluster["z0"], weights=1.0 / (track_at_cluster["err_z0"] ** 2))
    return avg_vtx_x0, avg_vtx_y0, avg_vtx_z0


class VertexPerformance:
    def __init__(
        self,
        file_path: Union[Path, str],
        ttree_name: str,
        eval_type: str,
        algo: str = "ivf",
        use_fitted_vtx: bool = False,
        if_save_fig: bool = False,
        output_path: Union[Path, str] = None,
        verbosity: int = 3,
        entry_stop: int = None,
    ) -> None:
        self.input_path = check_inputpath(file_path)

        try:
            root_file = uproot.open(self.input_path)
        except Exception:
            raise Exception(f"File {self.input_path} cannot be opened.")

        self.ttree_name = ttree_name
        try:
            self.events = root_file[self.ttree_name].arrays(entry_stop=entry_stop)
        except Exception:
            raise Exception(
                f"Tree {self.ttree_name} not found in {self.input_path}.\
                available keys are {root_file.keys()}"
            )
        self.algo = algo
        self.eval_type = eval_type
        self.use_fitted_vtx = use_fitted_vtx
        self.set_reco_vtx()
        logging_setup(verbosity=verbosity, if_write_log=False, output_path=None)

        self.if_save_fig = if_save_fig
        if self.if_save_fig:
            if output_path is None:
                raise Exception("output_path is None, but if_save_fig is True.")
            self.output_path = check_outputpath(output_path)

        hep.style.use("ATLAS")

    def set_reco_vtx(self):
        if self.use_fitted_vtx:
            if self.eval_type == "truth":
                self.key_vtx = "truth_PriVtx"
            elif self.eval_type == "reco":
                self.key_vtx = "reco_PriVtx"
            elif self.eval_type == "spvcnn":
                self.key_vtx = "spvcnn_PriVtx"
            else:
                raise NotImplementedError(f"eval_type {self.eval_type} not found.")
            self.position_type = "fitted"
            self.key_vtx_vx = self.key_vtx + "X"
            self.key_vtx_vy = self.key_vtx + "Y"
            self.key_vtx_vz = self.key_vtx + "Z"

            self.reco_vtx_vx_allevenents = self.events[self.key_vtx_vx]
            self.reco_vtx_vy_allevenents = self.events[self.key_vtx_vy]
            self.reco_vtx_vz_allevenents = self.events[self.key_vtx_vz]

        elif not self.use_fitted_vtx:
            self.position_type = "weighted_avg"
            # now we calculate the weighted average of the reco tracks as the reco vertex position
            self.reco_vtx_vx_allevenents = []
            self.reco_vtx_vy_allevenents = []
            self.reco_vtx_vz_allevenents = []

            for event_idx, event in enumerate(self.events):
                if self.eval_type == "truth":
                    reco_trk_reco_vtx_idx = event.reco_trk_truth_vtx_idx
                elif self.eval_type == "reco" and self.algo == "ivf":
                    reco_trk_reco_vtx_idx = event.reco_trk_reco_vtx_idx
                    reco_trk_reco_vtx_idx = ak.flatten(reco_trk_reco_vtx_idx, axis=1)

                elif self.eval_type == "reco" and self.algo == "spvcnn":
                    reco_trk_reco_vtx_idx = (
                        event.reco_trk_spvcnn_vtx_ins_idx - 1
                    )  # minus 1 because spvcnn instance label of 0 is noise
                    sem_mask = (
                        event.reco_trk_spvcnn_vtx_sem_idx == 0
                    )  # sem_idx of 0 is bkg by semantic segmentation
                    np.asarray(reco_trk_reco_vtx_idx)[
                        sem_mask
                    ] = (
                        -1
                    )  # set the reco vtx idx to -1 for tracks that are preded as bkg by semantic segmentation
                else:
                    raise NotImplementedError(f"eval_type {self.eval_type} for fitted vtx not found.")
                try:
                    reco_vtx_vx, reco_vtx_vy, reco_vtx_vz = self.get_weighted_avg_vtx_position(
                        event, reco_trk_reco_vtx_idx
                    )
                except ValueError:
                    print(f"event {event_idx} ")
                self.reco_vtx_vx_allevenents.append(reco_vtx_vx)
                self.reco_vtx_vy_allevenents.append(reco_vtx_vy)
                self.reco_vtx_vz_allevenents.append(reco_vtx_vz)

    def eval(self):
        self.eval_differenceZ()

    def get_difference_Z_ak(self, vtx_vz_allevents):
        assert isinstance(vtx_vz_allevents, ak.Array)
        differenceZ = []
        for vtx_vz_event in vtx_vz_allevents:
            distance_all_pairs = ak.flatten(vtx_vz_event[:, None] - vtx_vz_event)
            differenceZ.append(distance_all_pairs[distance_all_pairs != 0])
        return differenceZ

    def get_weighted_avg_vtx_position(self, event, reco_trk_reco_vtx_idx):
        assert len(reco_trk_reco_vtx_idx) == len(event.reco_trk_z0)
        mask = reco_trk_reco_vtx_idx >= 0
        reco_trk_d0 = event.reco_trk_d0[mask]
        reco_trk_z0 = event.reco_trk_z0[mask]
        reco_trk_phi = event.reco_trk_phi[mask]
        reco_trk_err_d0 = event.reco_trk_err_d0[mask]
        reco_trk_err_z0 = event.reco_trk_err_z0[mask]
        reco_trk_reco_vtx_idx = reco_trk_reco_vtx_idx[mask]

        reco_trk_x0 = reco_trk_d0 * np.cos(reco_trk_phi)
        reco_trk_y0 = reco_trk_d0 * np.sin(reco_trk_phi)
        reco_tracks_pos = {
            "x0": reco_trk_x0,
            "y0": reco_trk_y0,
            "z0": reco_trk_z0,
            "err_z0": reco_trk_err_z0,
            "err_d0": reco_trk_err_d0,
            "pred_instance_labels": reco_trk_reco_vtx_idx,
        }
        # breakpoint()
        try:
            reco_tracks_pos_df = pd.DataFrame(reco_tracks_pos)
        except ValueError:
            breakpoint()
        avg_vtx_3pos = (
            reco_tracks_pos_df.groupby("pred_instance_labels").apply(calculate_component).to_list()
        )
        avg_vtx_3pos = np.array(avg_vtx_3pos)

        return avg_vtx_3pos[:, 0], avg_vtx_3pos[:, 1], avg_vtx_3pos[:, 2]

    def get_classification_and_eff(
        self,
        total_reco_vtx_type,
        total_hs_type,
        hs_reco_eff,
        hs_sel_eff,
        hs_reco_sel_eff,
        truth_HS_idx=0,
    ):
        for event_idx, event in enumerate(tqdm((self.events))):
            # logging.debug(f"Processing event {event_idx}...")
            reco_vtx_vx = self.reco_vtx_vx_allevenents[event_idx]
            reco_vtx_vy = self.reco_vtx_vy_allevenents[event_idx]
            reco_vtx_vz = self.reco_vtx_vz_allevenents[event_idx]

            truth_vtx_vx = event.truth_PriVtxX
            truth_vtx_vy = event.truth_PriVtxY
            truth_vtx_vz = event.truth_PriVtxZ

            if not np.sum(event.reco_trk_truth_vtx_idx == 0):
                logging.warning(f"Truth HS in event {event_idx} donesn't have reco tracks, skipping!")
                continue

            RecoVertexMatchInfo = self.get_MatchInfo(event, event_idx)
            vtx_types = self.classifyRecoVertex(RecoVertexMatchInfo)
            hs_type = self.classifyHardScatter_athena(RecoVertexMatchInfo, vtx_types)

            total_reco_vtx_type += np.bincount(vtx_types, minlength=VertexMatchType.NTYPES.value)
            total_hs_type[hs_type.value] += 1

            idx_best_reco_HS_nTrk = np.argmax(RecoVertexMatchInfo[0, :, 0])
            idx_best_reco_HS_sumpt2 = np.argmax(RecoVertexMatchInfo[1].sum(axis=1))

            residual = np.array(
                [
                    reco_vtx_vx[idx_best_reco_HS_nTrk] - truth_vtx_vx[truth_HS_idx],
                    reco_vtx_vy[idx_best_reco_HS_nTrk] - truth_vtx_vy[truth_HS_idx],
                    reco_vtx_vz[idx_best_reco_HS_nTrk] - truth_vtx_vz[truth_HS_idx],
                ]
            )

            local_PU_density = self.get_localPUdensity(
                truth_vtx_vx=truth_vtx_vx,
                truth_vtx_vy=truth_vtx_vy,
                truth_vtx_vz=truth_vtx_vz,
            )

            trhth_HS_vtx_recoed = False
            trhth_HS_vtx_seled = False
            if np.square(residual).sum() <= cutMinTruthRecoRadialDiff**2:
                trhth_HS_vtx_recoed = True
            if idx_best_reco_HS_nTrk == idx_best_reco_HS_sumpt2:
                trhth_HS_vtx_seled = True
            if self.eval_type == "truth" and not trhth_HS_vtx_seled:
                logging.debug(f"Truth HS not selected in event {event_idx}")

            hs_reco_eff.Fill(trhth_HS_vtx_recoed, local_PU_density)
            hs_sel_eff.Fill(
                trhth_HS_vtx_seled,
                local_PU_density,
            )
            hs_reco_sel_eff.Fill(
                trhth_HS_vtx_recoed and trhth_HS_vtx_seled,
                local_PU_density,
            )

        return (
            total_reco_vtx_type,
            total_hs_type,
            hs_reco_eff,
            hs_sel_eff,
            hs_reco_sel_eff,
        )

    def get_MatchInfo(self, event: ak.highlevel.Record, event_idx: int):
        """Event base flattening and return the performance

        Args:
            event (ak.highlevel.Record): _description_
        """

        reco_vtx_vz = self.reco_vtx_vz_allevenents[event_idx]
        truth_vtx_vz = event["truth_PriVtxZ"]

        n_reco_vtx = len(reco_vtx_vz)
        n_truth_vtx = len(truth_vtx_vz)

        reco_trk_pt2 = np.power((1.0 / event.reco_trk_qop * np.sin(event.reco_trk_theta)), 2)
        if self.eval_type == "truth":
            reco_trk_reco_vtx_idx = ak.values_astype(event.reco_trk_truth_vtx_idx, "int32")
            reco_trk_reco_vtx_trackWeight = np.ones_like(reco_trk_reco_vtx_idx)

        elif self.eval_type == "reco" and self.algo == "ivf":
            reco_trk_reco_vtx_idx = ak.values_astype(event.reco_trk_reco_vtx_idx, "int32")
            reco_trk_reco_vtx_trackWeight = event.reco_trk_reco_vtx_trackWeight

        elif self.eval_type == "reco" and self.algo == "spvcnn":
            reco_trk_reco_vtx_idx = ak.values_astype(
                event["reco_trk_spvcnn_vtx_ins_idx"] - 1, "int32"
            )  # minus 1 because spvcnn instance label of 0 is noise
            sem_mask = event["reco_trk_spvcnn_vtx_sem_idx"] == 0  # sem_idx of 0 is bkg
            np.asarray(reco_trk_reco_vtx_idx)[
                sem_mask
            ] = -1  # set the reco vtx idx to -1 for tracks that are preded as bkg

            reco_trk_reco_vtx_trackWeight = np.ones_like(reco_trk_reco_vtx_idx)

        reco_trk_truth_vtx_idx = event.reco_trk_truth_vtx_idx

        # the first row is number of tracks
        # the second row is the sum of pt2 of tracks
        # the third row is the sum of the track weights to the reco vertices

        if self.eval_type == "truth":
            assert np.all(reco_trk_reco_vtx_idx == reco_trk_truth_vtx_idx)

            reconstructable_truth_vtx_idx = np.unique(event.reco_trk_truth_vtx_idx)
            reconstructable_truth_vtx_idx = reconstructable_truth_vtx_idx[
                reconstructable_truth_vtx_idx >= 0
            ]
            if 0 not in reconstructable_truth_vtx_idx:
                logging.warning(
                    "Truth vertex 0 is not in the reconstructable truth vertex list, added it manually!"
                )
                reconstructable_truth_vtx_idx = np.insert(reconstructable_truth_vtx_idx, 0, 0, axis=0)
            reconstructable_truth_vtx_idx = reconstructable_truth_vtx_idx.to_list()
            n_reconstructable_truth_vtx = len(reconstructable_truth_vtx_idx)

            RecoVertexMatchInfo = np.zeros(
                (3, n_reconstructable_truth_vtx, n_reconstructable_truth_vtx),
                dtype=float,
            )

            for reco_trk_idx in range(len(reco_trk_truth_vtx_idx)):
                ### filter the reco tracks that are not or illy associated to any truth particles
                if reco_trk_truth_vtx_idx[reco_trk_idx] < 0:
                    continue
                truth_vtx_idx = reco_trk_truth_vtx_idx[reco_trk_idx]
                reco_vtx_idx = reco_trk_reco_vtx_idx[reco_trk_idx]

                if truth_vtx_idx not in reconstructable_truth_vtx_idx:
                    continue

                reco_vtx_idx = reconstructable_truth_vtx_idx.index(reco_vtx_idx)
                truth_vtx_idx = reconstructable_truth_vtx_idx.index(truth_vtx_idx)

                RecoVertexMatchInfo[0, reco_vtx_idx, truth_vtx_idx] += 1
                RecoVertexMatchInfo[1, reco_vtx_idx, truth_vtx_idx] += reco_trk_pt2[reco_trk_idx]
                RecoVertexMatchInfo[2, reco_vtx_idx, truth_vtx_idx] += reco_trk_reco_vtx_trackWeight[
                    reco_trk_idx
                ]

            ### TODO Add reco track matching prob check
            ### TODO Add fake reco track check
            # assert np.sum(RecoVertexMatchInfo[0, :, :]) == np.sum(ak.flatten(reco_trk_reco_vtx_idx, axis=0) >= 0)
        if self.eval_type == "reco" and self.algo == "spvcnn":
            ### reco_trk_reco_vtx_idx is not continuous, need to convert it to continuous index
            ### e.g. [1, 2, 5] -> [0, 1, 2]

            valid_reco_vertex_idx = np.unique(reco_trk_reco_vtx_idx)
            valid_reco_vertex_idx = valid_reco_vertex_idx[valid_reco_vertex_idx >= 0]
            map_spvcnn_vtx_idx = dict.fromkeys(valid_reco_vertex_idx)
            for i, vtx_idx in enumerate(valid_reco_vertex_idx):
                map_spvcnn_vtx_idx[vtx_idx] = i

            reco_trk_truth_vtx_idx = ak.values_astype(reco_trk_truth_vtx_idx, "int32")
            RecoVertexMatchInfo = np.zeros((3, n_reco_vtx, n_truth_vtx), dtype=float)
            for reco_trk_idx in range(len(reco_trk_truth_vtx_idx)):
                truth_vtx_idx = reco_trk_truth_vtx_idx[reco_trk_idx]
                reco_vtx_idx = reco_trk_reco_vtx_idx[reco_trk_idx]

                if reco_trk_truth_vtx_idx[reco_trk_idx] < 0 or reco_trk_reco_vtx_idx[reco_trk_idx] < 0:
                    continue

                RecoVertexMatchInfo[0, map_spvcnn_vtx_idx[reco_vtx_idx], truth_vtx_idx] += 1
                RecoVertexMatchInfo[1, map_spvcnn_vtx_idx[reco_vtx_idx], truth_vtx_idx] += reco_trk_pt2[
                    reco_trk_idx
                ]
                RecoVertexMatchInfo[
                    2, map_spvcnn_vtx_idx[reco_vtx_idx], truth_vtx_idx
                ] += reco_trk_reco_vtx_trackWeight[reco_trk_idx]

        if self.eval_type == "reco" and self.algo == "ivf":
            reco_trk_truth_vtx_idx = ak.values_astype(reco_trk_truth_vtx_idx, "int32")
            RecoVertexMatchInfo = np.zeros((3, n_reco_vtx, n_truth_vtx), dtype=float)
            #### FIXME Should have a better numpy way to do this instead of looping
            for reco_trk_idx in range(len(reco_trk_truth_vtx_idx)):
                truth_vtx_idx = reco_trk_truth_vtx_idx[reco_trk_idx]
                reco_vtx_idxs = reco_trk_reco_vtx_idx[reco_trk_idx]

                ### filter the reco tracks that are not or illy associated to any truth particles
                if reco_trk_truth_vtx_idx[reco_trk_idx] < 0:
                    continue
                for i, reco_vtx_idx in enumerate(reco_vtx_idxs):
                    if reco_vtx_idx < 0:
                        continue

                    RecoVertexMatchInfo[0, reco_vtx_idx, truth_vtx_idx] += 1
                    RecoVertexMatchInfo[1, reco_vtx_idx, truth_vtx_idx] += reco_trk_pt2[reco_trk_idx]
                    RecoVertexMatchInfo[2, reco_vtx_idx, truth_vtx_idx] += reco_trk_reco_vtx_trackWeight[
                        reco_trk_idx
                    ][i]

            # assert np.sum(RecoVertexMatchInfo[0, :, :]) == np.sum(ak.flatten(reco_trk_reco_vtx_idx, axis=1) >= 0)

        return RecoVertexMatchInfo

    def classifyRecoVertex(self, RecoVertexMatchInfo):
        """According to the reco vertex match info, classify the reco vertex into different types
        Reference: http://cds.cern.ch/record/2670380, Chapter 3.2
        """
        RecoVertexMatchInfo_weight = RecoVertexMatchInfo[2]
        RecoVertexMatchInfo_pt2 = RecoVertexMatchInfo[1]

        n_reco_vtx, n_truth_vtx = RecoVertexMatchInfo_weight.shape
        # normalized_RecoVertexMatchInfo_weight = RecoVertexMatchInfo_weight / np.sum(RecoVertexMatchInfo_weight, axis=0)
        normalized_RecoVertexMatchInfo_weight = (
            RecoVertexMatchInfo_weight / np.sum(RecoVertexMatchInfo_weight, axis=1)[:, None]
        )
        np.nan_to_num(normalized_RecoVertexMatchInfo_weight, copy=False)

        vtx_types = -1 * np.ones(n_reco_vtx, dtype=int)
        is_set = np.zeros(n_reco_vtx, dtype=bool)

        for reco_vtx_id in range(0, n_reco_vtx):
            max_weight_idx = np.argmax(normalized_RecoVertexMatchInfo_weight[reco_vtx_id])
            max_weight = normalized_RecoVertexMatchInfo_weight[reco_vtx_id, max_weight_idx]

            if max_weight == 0:  # this is a fake vertex, no contribution from any truth vertex
                vtx_types[reco_vtx_id] = VertexMatchType.FAKE.value

            elif max_weight > vxMatchWeight:
                if (
                    reco_vtx_id == RecoVertexMatchInfo_weight[:, max_weight_idx].argmax()
                    and reco_vtx_id == RecoVertexMatchInfo_pt2[:, max_weight_idx].argmax()
                ):
                    vtx_types[reco_vtx_id] = VertexMatchType.MATCHED.value
                    is_set[reco_vtx_id] = 1

            elif max_weight < vxMatchWeight:
                if np.count_nonzero(normalized_RecoVertexMatchInfo_weight[reco_vtx_id]) > 1:
                    vtx_types[reco_vtx_id] = VertexMatchType.MERGED.value

        ### Correct the labelling for split
        for truth_vtx_id in range(0, n_truth_vtx):
            if not np.any(RecoVertexMatchInfo_weight[:, truth_vtx_id]):
                ### This truth vertex has no contribution to any reco vertex
                continue

            if np.count_nonzero(RecoVertexMatchInfo_weight[:, truth_vtx_id]) == 1:
                ### This truth vertex has only one contribution to one reco vertex
                reco_vtx_id_nozero = np.nonzero(RecoVertexMatchInfo_weight[:, truth_vtx_id])[0][0]
                if normalized_RecoVertexMatchInfo_weight[reco_vtx_id_nozero, :].max() > vxMatchWeight:
                    vtx_types[reco_vtx_id_nozero] = VertexMatchType.MATCHED.value
                    is_set[reco_vtx_id_nozero] = 1
                else:
                    vtx_types[reco_vtx_id_nozero] = VertexMatchType.MERGED.value
                    is_set[reco_vtx_id_nozero] = 1
                continue

            ### This truth vertex has multiple contribution to multiple reco vertex
            ### Find the reco vertex that has the largest contribution from this truth vertex

            contributed_to = np.nonzero(RecoVertexMatchInfo_weight[:, truth_vtx_id])[0]
            contributed_to_max_reco_vtx_id = RecoVertexMatchInfo_pt2[contributed_to].sum(axis=1).argmax()
            max_reco_vtx_id = contributed_to[contributed_to_max_reco_vtx_id]
            if normalized_RecoVertexMatchInfo_weight[max_reco_vtx_id, truth_vtx_id] > vxMatchWeight:
                ### The largest contribution is larger than the threshold, this is a matched vertex
                vtx_types[max_reco_vtx_id] = VertexMatchType.MATCHED.value
                is_set[max_reco_vtx_id] = 1
            else:
                ### The largest contribution is smaller than the threshold, this is a split vertex
                vtx_types[max_reco_vtx_id] = VertexMatchType.MERGED.value
                is_set[max_reco_vtx_id] = 1

            others = np.delete(contributed_to, contributed_to_max_reco_vtx_id, axis=0)

            for other_reco_vtx_id in others:
                # if vtx_types[other_reco_vtx_id] == VertexMatchType.SPLIT.value:
                #     continue
                if is_set[other_reco_vtx_id]:
                    continue
                vtx_types[other_reco_vtx_id] = VertexMatchType.SPLIT.value
                is_set[other_reco_vtx_id] = 1
        if not np.all(is_set):
            logging.warning("Not all reco vtx are set!")
            breakpoint()
        assert 0 not in is_set
        return vtx_types

    def classifyHardScatter_athena(
        self, RecoVertexMatchInfo: np.ndarray, vtx_types: dict
    ) -> HardScatterType:
        """This is dumped from the Athena Code, may be different from the PubNote"""
        # count how many reco vtx the truth HS contributes to
        n_contribution_from_truth_HS = np.count_nonzero(RecoVertexMatchInfo[0, :, 0])
        if n_contribution_from_truth_HS == 0:
            return HardScatterType.NONE
        elif n_contribution_from_truth_HS == 1:
            # find the only one reco idx that truth HS contributes to
            reco_vtx_idx = np.flatnonzero(RecoVertexMatchInfo[0, :, 0] != 0)[0]
            # check if the truth HS is the largest contribution to that reco vtx
            is_largest_contribution = reco_vtx_idx == np.argmax(RecoVertexMatchInfo[1, :, 0])
            reco_vtx_type = vtx_types[reco_vtx_idx]

            if is_largest_contribution and reco_vtx_type == VertexMatchType.MATCHED.value:
                return HardScatterType.CLEAN
            elif is_largest_contribution and reco_vtx_type == VertexMatchType.MERGED.value:
                return HardScatterType.LOWPU
            else:
                return HardScatterType.HIGHPU
        else:
            # multiple reco vertices have tracks from the hard-scatter
            # count how many have hard-scatter tracks as largest contribution

            # get indexes of reco vtxs that have hard-scatter tracks as contribution
            reco_vtx_idxs = np.flatnonzero(RecoVertexMatchInfo[0, :, 0] != 0)
            largest_contributution_idxs = np.argmax(RecoVertexMatchInfo[1, reco_vtx_idxs, :], axis=1)
            n_largest_contribution_from_truth_HS = np.count_nonzero(largest_contributution_idxs == 0)
            if n_largest_contribution_from_truth_HS == 0:
                return HardScatterType.HIGHPU
            elif n_largest_contribution_from_truth_HS == 1:
                # Only one reco vtx has the largest contribution
                # identify this reco vtx
                reco_vtx_idx = reco_vtx_idxs[np.where(largest_contributution_idxs == 0)[0][0]]
                # take its vtx type
                reco_vtx_type = vtx_types[reco_vtx_idx]
                # choose the event type
                if reco_vtx_type == VertexMatchType.MATCHED.value:
                    return HardScatterType.CLEAN
                elif reco_vtx_type == VertexMatchType.MERGED.value:
                    return HardScatterType.LOWPU
                else:
                    return HardScatterType.HIGHPU
            else:
                return HardScatterType.HSSPLIT

    def get_localPUdensity(
        self,
        truth_vtx_vx,
        truth_vtx_vy,
        truth_vtx_vz,
        truth_HS_idx=0,
        xyz_dist_window=2.0,
    ):
        # Calculate the PU density around the truth HS vertex
        residual_truth_vtx_vx = truth_vtx_vx - truth_vtx_vx[truth_HS_idx]
        residual_truth_vtx_vy = truth_vtx_vy - truth_vtx_vy[truth_HS_idx]
        residual_truth_vtx_vz = truth_vtx_vz - truth_vtx_vz[truth_HS_idx]

        dist_to_truth_HS = (
            residual_truth_vtx_vx**2 + residual_truth_vtx_vy**2 + residual_truth_vtx_vz**2
        )
        n_local_truth = len(np.where(dist_to_truth_HS < xyz_dist_window**2)[0])
        return (n_local_truth - 1) / (2 * xyz_dist_window)

    def eval_differenceZ(self, start: int = -5, stop: int = 5, bins: int = 50):
        logging.info("Evaluating difference Z...")
        if self.use_fitted_vtx:
            differenceZ = self.get_difference_Z_ak(self.reco_vtx_vz_allevenents)
        else:
            differenceZ = self.get_difference_Z_ak(ak.Array(self.reco_vtx_vz_allevenents))

        differenceZ = np.concatenate(differenceZ)
        hist_differenceZ = hist.Hist(
            hist.axis.Regular(
                bins=bins,
                start=start,
                stop=stop,
                name=f"{name_title_dicts['diffZ']}",
            )
        )
        hist_differenceZ.fill(differenceZ)

        fig, ax = self.plot_hist(
            hist_differenceZ,
            xlabel=name_title_dicts["diffZ"],
            title=f"diff_Z_{self.eval_type}_{self.algo}_{self.position_type}",
            output_name=f"diffZ_{self.eval_type}_{self.algo}_{self.position_type}.png",
        )
        logging.info("Evaluating difference Z finished.")
        ### return the figure and axis object for further modification
        return differenceZ

    def eval_classification_and_eff(self):
        logging.info("Evaluating classification and efficiency...")
        total_reco_vtx_type = np.zeros(VertexMatchType.NTYPES.value, dtype=int)
        total_hs_type = np.zeros(HardScatterType.NHSTYPES.value, dtype=int)

        hs_reco_eff = ROOT.TEfficiency(
            "hs_reco_eff",
            f"HS Reconstruction Efficiency {self.eval_type}_{self.position_type}; Local PU density; eff",
            12,
            0,
            6,
        )

        hs_sel_eff = ROOT.TEfficiency(
            "hs_sel_eff",
            f"HS Selection Efficiency {self.eval_type}_{self.position_type}; Local PU density; eff",
            12,
            0,
            6,
        )

        hs_reco_sel_eff = ROOT.TEfficiency(
            "hs_reco_sel_eff",
            f"HS Reconstruction and Selection Efficiency {self.eval_type}_{self.position_type}; Local PU density; eff",
            12,
            0,
            6,
        )
        # del hs_reco_eff, hs_sel_eff, hs_reco_sel_eff
        self.get_classification_and_eff(
            total_reco_vtx_type,
            total_hs_type,
            hs_reco_eff,
            hs_sel_eff,
            hs_reco_sel_eff,
        )
        self.plot_pv_hs_classification(
            total_reco_vtx_type,
            class_type="pv",
            output_name=f"pv_classification_{self.eval_type}_{self.algo}_{self.position_type}.png",
        )
        self.plot_pv_hs_classification(
            total_hs_type,
            class_type="hs",
            output_name=f"hs_classification_{self.eval_type}_{self.algo}_{self.position_type}.png",
        )
        self.plot_eff(
            hs_reco_eff,
            class_type="hs",
            output_name=f"hs_reco_eff_{self.eval_type}_{self.algo}_{self.position_type}",
        )
        self.plot_eff(
            hs_sel_eff,
            class_type="hs",
            output_name=f"hs_sel_eff_{self.eval_type}_{self.algo}_{self.position_type}",
        )
        self.plot_eff(
            hs_reco_sel_eff,
            class_type="hs",
            output_name=f"hs_reco_sel_eff_{self.eval_type}_{self.algo}_{self.position_type}",
        )

        logging.info("Evaluating classification and efficiency finished.")
        return (
            total_reco_vtx_type,
            total_hs_type,
            hs_reco_eff,
            hs_sel_eff,
            hs_reco_sel_eff,
        )

    def plot_hist(
        self,
        _hist: hist.Hist,
        ylabel: str = "Number of Vertices",
        xlabel: str = None,
        title: str = None,
        output_name: str = None,
    ):
        fig, ax = plt.subplots()
        hep.histplot(_hist, density=False, histtype="step")
        ax.set_ylabel(ylabel=ylabel)
        ax.set_xlabel(xlabel=xlabel)
        ax.set_title(label=title)

        if self.if_save_fig:
            fig.savefig(self.output_path / output_name)
        # plt.show()
        return fig, ax

    def plot_pv_hs_classification(
        self,
        enum_classification: np.ndarray,
        class_type: Enum,
        output_name: str = None,
    ):
        if class_type == "pv":
            enum_type = VertexMatchType
            identifier_title = "Primary Vertex"

        elif class_type == "hs":
            enum_type = HardScatterType
            identifier_title = "HS Event"

        n_types = VertexMatchType.NTYPES.value
        bins_edges = np.arange(0, n_types + 1)

        bin_centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])
        x_labels = enum_type._member_names_[:-1]

        fig, ax = plt.subplots()
        ax.stairs(enum_classification)
        ax.set_xticks(bin_centers, x_labels)
        ax.set_title(
            f"{identifier_title} classification on {self.eval_type}_{self.algo}_{self.position_type}"
        )
        ax.set_xlabel(f"{identifier_title} type")
        ax.set_ylabel("Numbers")
        if self.if_save_fig:
            fig.savefig(self.output_path / output_name)

    def plot_eff(self, eff: ROOT.TEfficiency, class_type, output_name: str = None):
        # pass
        canvas_eff = ROOT.TCanvas()
        legend_eff = ROOT.TLegend(0.1, 0.2, 0.4, 0.4)

        eff.SetLineColor(2)
        eff.Draw()
        legend_eff.AddEntry(eff, f"{self.algo}")
        legend_eff.Draw("same")
        canvas_eff.Draw()
        if self.if_save_fig:
            canvas_eff.Print((self.output_path / f"{output_name}.png").as_posix())
