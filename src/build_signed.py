"""Session 4 upgrade #2: assign E/I signs from the neurotransmitter file that was
downloaded but never used in sessions 1-3.

Sign convention (documented in experiment.md §10):
  inhibitory  = {gaba, histamine}        (histamine is the fly photoreceptor transmitter)
  excitatory  = everything else, including glutamate (caveat: a subset of glutamatergic
                 synapses in insects is inhibitory via GluCl; we treat glutamate as
                 excitatory for simplicity and say so)
  label chain per body: consensus_nt -> celltype_predicted_nt -> predicted_nt -> '+1'
  edge sign = sign of the PRESYNAPTIC neuron (a neuron carries one sign to all targets)

Variants (all saved as raw signed counts; loaders row-|sum|-normalize):
  adjacency_flysigned_s0.npz       real wiring, real signs
  adjacency_shuffledsigned_s0.npz  signed values globally permuted (sign travels with value)
  adjacency_randomsigned_s0.npz    random targets, weights resampled from signed values
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repo_paths import PROCESSED, RESULTS

INHIB = {"gaba", "histamine"}


def sign_vector():
    nt = pd.read_parquet(f"{PROCESSED}/neurotransmitters.parquet")
    chain = nt[["body", "consensus_nt", "celltype_predicted_nt", "predicted_nt"]].copy()
    chain["label"] = (chain["consensus_nt"].fillna("")
                      .where(chain["consensus_nt"].fillna("") != "unclear", "")
                      )
    need = chain["label"] == ""
    chain.loc[need, "label"] = chain.loc[need, "celltype_predicted_nt"].fillna("")
    need = chain["label"] == ""
    chain.loc[need, "label"] = chain.loc[need, "predicted_nt"].fillna("")
    lab = chain.set_index("body")["label"]
    bodies = np.load(f"{PROCESSED}/annotated_body_ids.npy")
    lbl = np.array([lab.get(int(b), "") for b in bodies])
    sign = np.where(np.isin(lbl, list(INHIB)), -1.0, 1.0).astype(np.float32)
    return sign, lbl


def stats_edges(A, sign, name):
    """Edge stats by count and by synapse weight."""
    coo = A.tocoo()
    src_sign = sign[coo.row]
    inh_mask = src_sign < 0
    w = coo.data
    out = {
        "variant": name,
        "edges": int(A.nnz),
        "edges_inhibitory_frac": float(inh_mask.mean()),
        "synapses_inhibitory_absfrac": float(np.abs(w[inh_mask]).sum() / np.abs(w).sum()),
        "neurons_inhibitory_frac": float((sign < 0).mean()),
    }
    return out


def main():
    sign, lbl = sign_vector()
    A = sp.load_npz(f"{PROCESSED}/adjacency.npz").tocsr().astype(np.float32)
    N = A.shape[0]
    labeled = np.isin(lbl, list(INHIB)) | np.isin(
        lbl, ["acetylcholine", "glutamate", "dopamine", "serotonin", "octopamine"])
    print(f"neurons: {N}; labeled {labeled.mean()*100:.1f}%; "
          f"inhibitory {np.mean(sign < 0) * 100:.1f}% (labeled-only "
          f"{np.mean(sign[labeled] < 0) * 100:.1f}%)", flush=True)

    stats = {"sign_rule": "inh={gaba,histamine}; edge sign = presynaptic neuron sign",
             "neurons_labeled_frac": float(labeled.mean()),
             "neurons_inhibitory_frac": float(np.mean(sign < 0)),
             "neurons_inhibitory_frac_labeled_only": float(np.mean(sign[labeled] < 0)),
             "variants": []}

    # ---- fly_signed: real wiring, real signs
    Af = A.copy()
    Af.data = Af.data * np.repeat(sign, np.diff(Af.indptr))
    st = stats_edges(Af, sign, "flysigned")
    sp.save_npz(f"{PROCESSED}/adjacency_flysigned_s0.npz", Af)
    stats["variants"].append(st)
    print("flysigned:", st, flush=True)
    del Af

    # ---- shuffled_signed: globally permute signed values (sign travels with value)
    rng = np.random.default_rng(0)
    As = A.copy()
    As.data = As.data * np.repeat(sign, np.diff(As.indptr))
    As.data = rng.permutation(As.data)
    st = stats_edges(As, sign, "shuffledsigned")
    sp.save_npz(f"{PROCESSED}/adjacency_shuffledsigned_s0.npz", As)
    stats["variants"].append(st)
    print("shuffledsigned:", st, flush=True)
    del As

    # ---- random_signed: random targets, weights resampled from signed values
    from reservoir_lib import random_graph_like
    Ar = A.copy()
    Ar.data = Ar.data * np.repeat(sign, np.diff(Ar.indptr))
    Ar = random_graph_like(Ar, seed=0)
    st = stats_edges(Ar, sign, "randomsigned")
    sp.save_npz(f"{PROCESSED}/adjacency_randomsigned_s0.npz", Ar)
    stats["variants"].append(st)
    print("randomsigned:", st, flush=True)

    with open(f"{RESULTS}/signed_build_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("saved stats -> results/signed_build_stats.json", flush=True)


if __name__ == "__main__":
    main()
