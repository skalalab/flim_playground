"""
Author: Pinar Demetci
Principal Investigator: Ritambhara Singh, Ph.D. from Brown University
08 August 2021
Updated: 23 February 2023
SCOTv2 algorithm: Single Cell alignment using Optimal Transport version 2
Correspondence: pinar_demetci@brown.edu, ritambhara@brown.edu
"""

import numpy as np
import scipy
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, normalize


class SCOTv2(object):
    """
    Minimal SCOT v2 runtime for two-dataset alignment.

    Original source: Singh Lab at Brown University, MIT licensed.
    """

    def __init__(self, data):
        assert type(data) == list and len(data) >= 2, (
            "SCOTv2 expects a list of at least two numpy arrays."
        )
        self.data = data
        self.marginals = []
        self.graphs = []
        self.graphDists = []
        self.couplings = []
        self.gwdists = []
        self.flags = []
        self.aligned_data = []

    def _init_marginals(self):
        for i in range(len(self.data)):
            num_cells = self.data[i].shape[0]
            marginalDist = torch.ones(num_cells) / num_cells
            self.marginals.append(marginalDist)
        return self.marginals

    def _normalize(self, norm="l2", bySample=True):
        assert norm in ["l1", "l2", "max", "zscore"], (
            "Norm must be one of 'max', 'l1', 'l2', or 'zscore'."
        )

        for i in range(len(self.data)):
            if norm == "zscore":
                scaler = StandardScaler()
                self.data[i] = scaler.fit_transform(self.data[i])
            else:
                axis = 1 if (bySample is True or bySample is None) else 0
                self.data[i] = normalize(self.data[i], norm=norm, axis=axis)
        return self.data

    def construct_graph(self, k=20, mode="connectivity", metric="correlation"):
        assert mode in ["connectivity", "distance"], (
            "Mode must be either 'connectivity' or 'distance'."
        )
        include_self = True if mode == "connectivity" else False

        for i in range(len(self.data)):
            self.graphs.append(
                kneighbors_graph(
                    self.data[i],
                    n_neighbors=k,
                    mode=mode,
                    metric=metric,
                    include_self=include_self,
                )
            )

        return self.graphs

    def init_graph_distances(self):
        for i in range(len(self.data)):
            shortestPath = dijkstra(
                csgraph=csr_matrix(self.graphs[i]),
                directed=False,
                return_predecessors=False,
            )
            max_dist = np.nanmax(shortestPath[shortestPath != np.inf])
            shortestPath[shortestPath > max_dist] = max_dist
            self.graphDists.append(shortestPath / shortestPath.max())

        return self.graphDists

    def _exp_sinkhorn_solver(
        self,
        ecost,
        u,
        v,
        a,
        b,
        mass,
        eps,
        rho,
        rho2,
        nits_sinkhorn,
        tol_sinkhorn,
    ):
        if u is None or v is None:
            u, v = torch.ones_like(a), torch.ones_like(b)
        k = (a * u ** (-eps / rho)).sum() + (b * v ** (-eps / rho)).sum()
        k = k / (
            2
            * (
                u[:, None]
                * v[None, :]
                * ecost
                * a[:, None]
                * b[None, :]
            ).sum()
        )
        z = (0.5 * mass * eps) / (2.0 + 0.5 * (eps / rho) + 0.5 * (eps / rho2))
        k = k**z
        u, v = u * k, v * k

        for _ in range(nits_sinkhorn):
            u_prev = u.clone()
            v = torch.einsum("ij,i->j", ecost, a * u) ** (
                -1.0 / (1.0 + eps / rho)
            )
            u = torch.einsum("ij,j->i", ecost, b * v) ** (
                -1.0 / (1.0 + eps / rho2)
            )
            if (u.log() - u_prev.log()).abs().max().item() * eps < tol_sinkhorn:
                break
        pi = u[:, None] * v[None, :] * ecost * a[:, None] * b[None, :]
        return u, v, pi

    def exp_unbalanced_gw(
        self,
        a,
        dx,
        b,
        dy,
        eps=0.01,
        rho=1.0,
        rho2=None,
        nits_plan=3000,
        tol_plan=1e-6,
        nits_sinkhorn=3000,
        tol_sinkhorn=1e-6,
    ):
        if rho2 is None:
            rho2 = rho

        pi = a[:, None] * b[None, :] / (a.sum() * b.sum()).sqrt()
        up, vp = None, None

        for _ in range(nits_plan):
            pi_prev = pi.clone()
            mp = pi.sum()

            distxy = torch.einsum(
                "ij,kj->ik", dx, torch.einsum("kl,jl->kj", dy, pi)
            )
            kl_pi = torch.sum(pi * (pi / (a[:, None] * b[None, :]) + 1e-10).log())
            mu, nu = torch.sum(pi, dim=1), torch.sum(pi, dim=0)
            distxx = torch.einsum("ij,j->i", dx**2, mu)
            distyy = torch.einsum("kl,l->k", dy**2, nu)
            lcost = (distxx[:, None] + distyy[None, :] - 2 * distxy) + eps * kl_pi
            if rho < float("Inf"):
                lcost = lcost + rho * torch.sum(mu * (mu / a + 1e-10).log())
            if rho2 < float("Inf"):
                lcost = lcost + rho2 * torch.sum(nu * (nu / b + 1e-10).log())
            ecost = (-lcost / (mp * eps)).exp()

            up, vp, pi = self._exp_sinkhorn_solver(
                ecost, up, vp, a, b, mp, eps, rho, rho2, nits_sinkhorn, tol_sinkhorn
            )

            flag = True
            if torch.any(torch.isnan(pi)):
                flag = False

            pi = (mp / pi.sum()).sqrt() * pi
            if (pi - pi_prev).abs().max().item() < tol_plan:
                break
        return pi, flag

    def find_correspondences(
        self,
        normalize=True,
        norm="l2",
        bySample=True,
        k=20,
        mode="connectivity",
        metric="correlation",
        eps=0.01,
        rho=1.0,
        rho2=None,
    ):
        if normalize:
            self._normalize(norm=norm, bySample=bySample)
        self._init_marginals()
        self.construct_graph(k=k, mode=mode, metric=metric)
        self.init_graph_distances()
        for i in range(len(self.data) - 1):
            a, b = torch.Tensor(self.marginals[0]), torch.Tensor(self.marginals[i + 1])
            dx, dy = torch.Tensor(self.graphDists[0]), torch.Tensor(self.graphDists[i + 1])
            coupling, flag = self.exp_unbalanced_gw(
                a,
                dx,
                b,
                dy,
                eps=eps,
                rho=rho,
                rho2=rho2,
                nits_plan=3000,
                tol_plan=1e-6,
                nits_sinkhorn=3000,
                tol_sinkhorn=1e-6,
            )
            self.couplings.append(coupling)
            self.flags.append(flag)
            if flag is False:
                raise Exception(
                    f"Solver got NaN plan with params (eps, rho, rho2) = {eps, rho, rho2}. Try increasing eps."
                )
        return self.couplings

    def barycentric_projection(self):
        aligned_datasets = [self.data[0]]
        for i in range(0, len(self.couplings)):
            coupling = np.transpose(self.couplings[i].numpy())
            weights = np.sum(coupling, axis=1)
            projected_data = np.matmul((coupling / weights[:, None]), self.data[0])
            aligned_datasets.append(projected_data)
        return aligned_datasets

    def coembed_datasets(self, Lambda=1.0, out_dim=10):
        n_datasets = len(self.data)
        L = []
        for i in range(n_datasets - 1):
            self.couplings[i] = self.couplings[i] * np.shape(self.data[i])[0]

        for i in range(n_datasets):
            graph_data = self.graphs[i] + self.graphs[i].T.multiply(
                self.graphs[i].T > self.graphs[i]
            ) - self.graphs[i].multiply(self.graphs[i].T > self.graphs[i])
            W = np.array(graph_data.todense())
            index_pos = np.where(W > 0)
            W[index_pos] = 1 / W[index_pos]
            D = np.diag(np.dot(W, np.ones(np.shape(W)[1])))
            L.append(D - W)

        Sigma_x = []
        Sigma_y = []
        for i in range(n_datasets - 1):
            Sigma_y.append(
                np.diag(
                    np.dot(
                        np.transpose(np.ones(np.shape(self.couplings[i])[0])),
                        self.couplings[i],
                    )
                )
            )
            Sigma_x.append(
                np.diag(
                    np.dot(
                        self.couplings[i],
                        np.ones(np.shape(self.couplings[i])[1]),
                    )
                )
            )

        S_xy = self.couplings[0]
        S_xx = L[0] + Lambda * Sigma_x[0]
        S_yy = L[-1] + Lambda * Sigma_y[0]
        for i in range(1, n_datasets - 1):
            S_xy = np.vstack((S_xy, self.couplings[i]))
            S_xx = scipy.linalg.block_diag(S_xx, L[i] + Lambda * Sigma_x[i])
            S_yy = S_yy + Lambda * Sigma_y[i]

        v, Q = np.linalg.eig(S_xx)
        v = v + 1e-12
        V = np.diag(v ** (-0.5))
        H_x = np.dot(Q, np.dot(V, np.transpose(Q)))

        v, Q = np.linalg.eig(S_yy)
        v = v + 1e-12
        V = np.diag(v ** (-0.5))
        H_y = np.dot(Q, np.dot(V, np.transpose(Q)))

        H = np.dot(H_x, np.dot(S_xy, H_y))
        U, _, V = np.linalg.svd(H)

        num = [0]
        for i in range(n_datasets - 1):
            num.append(num[i] + len(self.data[i]))

        U, V = U[:, :out_dim], np.transpose(V)[:, :out_dim]

        fx = np.dot(H_x, U)
        fy = np.dot(H_y, V)

        integrated_data = []
        for i in range(n_datasets - 1):
            integrated_data.append(fx[num[i] : num[i + 1]])

        integrated_data.append(fy)
        return integrated_data

    def align(
        self,
        normalize=True,
        norm="l2",
        bySample=True,
        k=20,
        mode="connectivity",
        metric="correlation",
        eps=0.01,
        rho=1.0,
        rho2=None,
        projMethod="embedding",
        Lambda=1.0,
        out_dim=10,
    ):
        assert projMethod in ["embedding", "barycentric"], (
            "projMethod must be either 'embedding' or 'barycentric'."
        )
        self.find_correspondences(
            normalize=normalize,
            norm=norm,
            bySample=bySample,
            k=k,
            mode=mode,
            metric=metric,
            eps=eps,
            rho=rho,
            rho2=rho2,
        )
        if projMethod == "embedding":
            integrated_data = self.coembed_datasets(Lambda=Lambda, out_dim=out_dim)
        else:
            integrated_data = self.barycentric_projection()
        self.integrated_data = integrated_data
        return integrated_data
