"""Generate text samples from a trained FlyLM ESN state (fly / random / shuffled).

Loads flylm_state_*.npz (readout W, feature neurons, input wiring) + the connectome,
runs the same dynamics on a prompt, samples with temperature.
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reservoir_lib import load_corpus, load_adjacency, normalize_connectome, random_graph_like, shuffled_weights, DATA
from flylm_esn import to_torch_csr, pick_input_neurons

torch.set_num_threads(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="path to flylm_state_*.npz")
    ap.add_argument("--variant", default="fly", choices=["fly", "random", "shuffled"])
    ap.add_argument("--norm", default="global")
    ap.add_argument("--min-weight", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="First Citizen:\n")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.8)
    args = ap.parse_args()

    st = np.load(args.state, allow_pickle=True)
    W, feat_idx, in_idx, B_enc = st["W"], st["feat_idx"], st["in_idx"], st["B_enc"]
    leak, gain, lam = float(st["leak"]), float(st["gain"]), float(st["lam"])

    text, ids, stoi, itos = load_corpus()
    V = len(stoi)
    A_raw = load_adjacency()
    if args.min_weight > 1:
        import scipy.sparse as sp
        coo = A_raw.tocoo()
        keep = coo.data >= args.min_weight
        A_raw = sp.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])),
                              shape=A_raw.shape).tocsr()
    if args.variant == "fly":
        A = normalize_connectome(A_raw, args.norm, seed=args.seed)
    elif args.variant == "random":
        A = normalize_connectome(random_graph_like(A_raw, seed=args.seed), args.norm, seed=args.seed)
    else:
        A = normalize_connectome(shuffled_weights(A_raw, seed=args.seed), args.norm, seed=args.seed)
    At = to_torch_csr(A)
    N = A.shape[0]
    in_idx_t = torch.from_numpy(in_idx.astype(np.int64))
    feat_t = torch.from_numpy(feat_idx.astype(np.int64))
    B_enc_t = torch.from_numpy(B_enc)
    Wt = torch.from_numpy(W.astype(np.float32))

    rng = torch.Generator().manual_seed(args.seed + 99)
    prompt_ids = [stoi[c] for c in args.prompt if c in stoi]
    out = list(prompt_ids)
    x = torch.zeros(N, 1)

    def step(tok):
        Z = torch.sparse.mm(At, x) * gain
        Z.index_add_(0, in_idx_t, B_enc_t[torch.tensor([tok])].T)
        x.mul_(1 - leak).add_(torch.tanh(Z), alpha=leak)

    with torch.no_grad():
        for tok in prompt_ids:                    # warm on prompt
            step(tok)
        for _ in range(args.n):
            F = x[feat_t].T                       # (1, D)
            logits = (F @ Wt) / args.temp
            p = torch.softmax(logits[0], dim=0)
            nxt = torch.multinomial(p, 1, generator=rng).item()
            out.append(nxt)
            step(nxt)
    print("".join(itos[i] for i in out))


if __name__ == "__main__":
    main()
