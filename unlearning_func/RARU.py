import copy
import heapq
import math
import time

import torch
import torch.nn.functional as F
from torch.autograd import grad

from logger import Logger
from utils import determine_is_gnn

logger = Logger.get_logger("Unlearning(RARU)")


class RARU:
    def __init__(self, model, data, args, device):
        self.model = model
        self.data = data
        self.args = args
        self.device = device

        # Trainable parameters of the backbone model.
        self.params = [p for p in self.model.parameters() if p.requires_grad]

        # Hyperparameters (Section 5.1.6 of the paper).
        self.k = int(args.get("scope_cutoff_k", args.get("k", 10)))
        self.lam = float(args.get("damping_lambda", 1e-2))
        self.T = int(args.get("cg_steps", 30))
        self.max_scope_size = int(args.get("rarl_max_scope_size", 5000))

        # Ablation switches.
        self.no_sconn = bool(args.get("rarl_no_sconn", False))
        self.no_sim = bool(args.get("rarl_no_sim", False))
        self.no_localized = bool(args.get("rarl_no_localized", False))
        self.no_sw = bool(args.get("rarl_no_salience_weight", False))
        self.no_margin = bool(args.get("rarl_no_margin", False))
        self.no_curv = bool(args.get("rarl_no_curvature", False))

        self.num_users = data.num_users
        self.num_items = data.num_items
        self.is_gnn = determine_is_gnn(args.get("backbone"))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_embeddings(self):
        """Backbone-agnostic accessor for (user_emb, item_emb)."""
        if self.is_gnn:
            return self.model(edge_index=self.data.train_edge_index_after_remove)
        return self.model.get_embeddings()

    def _node_to_user_item(self, node_id):
        """Translate a global node id to ('user'/'item', local index)."""
        if node_id < self.num_users:
            return "user", int(node_id)
        return "item", int(node_id - self.num_users)

    def _build_adjacency(self, edge_index):
        """Build undirected adjacency list keyed by global node id."""
        adj = [[] for _ in range(self.num_users + self.num_items)]
        src = edge_index[0].cpu().tolist()
        dst = edge_index[1].cpu().tolist()
        for u, v in zip(src, dst):
            adj[u].append(v)
        return adj

    # ------------------------------------------------------------------
    # Phase 1: Localized Rank-Aware Salience (LRAS)
    # ------------------------------------------------------------------
    def _compute_entity_salience(self, candidates, target_entities,
                                 degrees, max_deg, user_emb, item_emb):
        """
        Equation (1), (2), (3) of the paper.

        Returns a dict {node_id: salience} for `candidates`.
        """
        salience = {}

        # Stack target embeddings split by entity type for the similarity term.
        t_users = [t for t in target_entities if t < self.num_users]
        t_items = [t - self.num_users for t in target_entities
                   if t >= self.num_users]
        target_user_emb = (user_emb[torch.tensor(t_users, device=self.device)]
                           if t_users else None)
        target_item_emb = (item_emb[torch.tensor(t_items, device=self.device)]
                           if t_items else None)

        # Pre-normalize for cosine similarity.
        if target_user_emb is not None:
            target_user_emb = F.normalize(target_user_emb, dim=1)
        if target_item_emb is not None:
            target_item_emb = F.normalize(target_item_emb, dim=1)

        log_dmax = math.log(1.0 + max(1, max_deg))

        for v in candidates:
            if v < self.num_users:
                v_emb = F.normalize(user_emb[v].unsqueeze(0), dim=1)
                # Same-type target set is users.
                if target_user_emb is not None and not self.no_sim:
                    sim = torch.clamp(
                        (v_emb @ target_user_emb.t()).max(), min=0.0
                    ).item()
                else:
                    sim = 0.0
            else:
                v_emb = F.normalize(item_emb[v - self.num_users].unsqueeze(0),
                                    dim=1)
                if target_item_emb is not None and not self.no_sim:
                    sim = torch.clamp(
                        (v_emb @ target_item_emb.t()).max(), min=0.0
                    ).item()
                else:
                    sim = 0.0

            if self.no_sconn:
                conn = 0.0
            else:
                conn = math.log(1.0 + degrees[v]) / log_dmax if log_dmax > 0 else 0.0

            if self.no_sconn and self.no_sim:
                # Both salience terms disabled - fall back to uniform priority.
                s = 1.0
            elif self.no_sconn:
                s = sim
            elif self.no_sim:
                s = conn
            else:
                s = 0.5 * (conn + sim)
            salience[v] = float(s)

        return salience

    def _lras_expand(self):
        """
        Phase 1 of Algorithm 1.

        Returns:
            V_S          : set of global node ids in the localized scope.
            entity_sal   : dict {node_id: S(v)} for all entities in V_S.
            target_set   : set of target entities V_t.
        """
        edges_to_remove = self.data.edges_to_remove
        if edges_to_remove.numel() == 0:
            return set(), {}, set()

        # Target entities V_t (Section 4.1).
        unl_task = self.args.get("unlearning_task", "interaction")
        if unl_task == "item":
            target_set = set(edges_to_remove[1].cpu().tolist())  # item nodes
        else:
            # Interaction unlearning: both endpoints are considered targets.
            target_set = set(edges_to_remove[0].cpu().tolist()) | \
                         set(edges_to_remove[1].cpu().tolist())

        # Build adjacency over the retained graph D_r.
        adj = self._build_adjacency(self.data.train_edge_index_after_remove)
        degrees = [len(a) for a in adj]
        max_deg = max(degrees) if degrees else 1

        with torch.no_grad():
            user_emb, item_emb = self._get_embeddings()

        # Initial frontier: retained neighbors N_0 of target entities.
        N0 = set()
        for t in target_set:
            for nb in adj[t]:
                if nb not in target_set:
                    N0.add(nb)

        V_S = set(target_set)
        entity_sal = {}

        if not N0:
            return V_S, entity_sal, target_set

        # S(v) for the initial frontier (Eq. 3) and threshold tau (Eq. 5).
        init_sal = self._compute_entity_salience(
            N0, target_set, degrees, max_deg, user_emb, item_emb
        )
        s_max = max(init_sal.values()) if init_sal else 0.0
        tau = s_max / max(self.k, 1)

        # Max-priority queue keyed by negative salience.
        heap = []
        visited = set(target_set)
        for v, s in init_sal.items():
            heapq.heappush(heap, (-s, v))
            visited.add(v)
            entity_sal[v] = s

        # Best-first expansion (Alg. 1 lines 5-7).
        while heap and len(V_S) < self.max_scope_size:
            neg_s, v = heapq.heappop(heap)
            s = -neg_s
            if s < tau:
                break
            V_S.add(v)

            # Push unvisited retained neighbors with their salience.
            new_candidates = [w for w in adj[v] if w not in visited]
            if not new_candidates:
                continue
            new_sal = self._compute_entity_salience(
                new_candidates, target_set, degrees, max_deg,
                user_emb, item_emb,
            )
            for w, sw in new_sal.items():
                visited.add(w)
                entity_sal[w] = sw
                heapq.heappush(heap, (-sw, w))

        # Target entities themselves are kept with salience 1 (S_bar in Eq. 4).
        for t in target_set:
            entity_sal[t] = 1.0

        return V_S, entity_sal, target_set

    # ------------------------------------------------------------------
    # Phase 2: Salience-Weighted Ranking Unlearning (SWRU)
    # ------------------------------------------------------------------
    def _filter_local_edges(self, V_S):
        """D_S^r: retained interactions whose endpoint is in V_S."""
        if self.no_localized or not V_S:
            return self.data.train_edge_index_after_remove

        V_S_tensor = torch.tensor(sorted(V_S), dtype=torch.long,
                                  device=self.device)
        e = self.data.train_edge_index_after_remove
        mask = torch.isin(e[0], V_S_tensor) | torch.isin(e[1], V_S_tensor)
        return e[:, mask]

    def _interaction_salience(self, edges, entity_sal, target_set):
        """Equation (4): S_ui = (S_bar(u) + S_bar(i))/2."""
        if self.no_sw:
            return torch.ones(edges.size(1), device=self.device)
        u_ids = edges[0].cpu().tolist()
        i_ids = edges[1].cpu().tolist()
        weights = []
        for u, i in zip(u_ids, i_ids):
            su = 1.0 if u in target_set else entity_sal.get(u, 0.0)
            si = 1.0 if i in target_set else entity_sal.get(i, 0.0)
            weights.append(0.5 * (su + si))
        return torch.tensor(weights, device=self.device)

    def _bpr_loss(self, edge_index, neg_items=None):
        """BPR loss used to compute g_t and the curvature term."""
        if edge_index.numel() == 0:
            return torch.zeros((), device=self.device, requires_grad=True)
        users = edge_index[0]
        pos_items = edge_index[1] - self.num_users
        if neg_items is None:
            neg_items = torch.randint(
                0, self.num_items, (users.size(0),), device=self.device
            )

        if self.is_gnn:
            u_all, i_all = self.model(
                edge_index=self.data.train_edge_index_after_remove
            )
        else:
            u_all, i_all = self.model.get_embeddings()

        u_e = u_all[users]
        pos_e = i_all[pos_items]
        neg_e = i_all[neg_items]
        pos_score = (u_e * pos_e).sum(dim=1)
        neg_score = (u_e * neg_e).sum(dim=1)
        return -F.logsigmoid(pos_score - neg_score).mean()

    def _flat_grad(self, loss, create_graph=False):
        grads = grad(loss, self.params, create_graph=create_graph,
                     allow_unused=True, retain_graph=True)
        return [g if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, self.params)]

    def _hvp(self, loss_grads, v):
        """Generic Hessian-vector product via double backward."""
        dot = sum((g * vi).sum() for g, vi in zip(loss_grads, v))
        hvp = grad(dot, self.params, retain_graph=True, allow_unused=True)
        return [h if h is not None else torch.zeros_like(p)
                for h, p in zip(hvp, self.params)]

    def _ranking_margin_grad(self, edges, comp_items, weights):
        """
        Returns the Jacobian function for the salience-weighted preservation
        matrix P_S so that we can form P_S @ v as J^T diag(S) J v.

        We avoid materializing J explicitly: instead we return a closure that
        computes P_S @ v with one forward and two backward passes.
        """
        users = edges[0]
        pos_items = edges[1] - self.num_users
        j_items = comp_items

        def margin_value():
            if self.is_gnn:
                u_all, i_all = self.model(
                    edge_index=self.data.train_edge_index_after_remove
                )
            else:
                u_all, i_all = self.model.get_embeddings()
            u_e = u_all[users]
            pos_e = i_all[pos_items]
            j_e = i_all[j_items]
            margin = (u_e * pos_e).sum(dim=1) - (u_e * j_e).sum(dim=1)
            # Aggregate to a scalar using salience weights; the Gauss-Newton
            # form P_S v = J^T diag(S) J v can be obtained via Jvp/vJp.
            return margin, weights

        def hvp_preservation(v):
            margin, w = margin_value()
            # First backward: compute u = J v, treating margin as a vector
            # whose Jacobian wrt theta is J. We need (J v).
            # Use trick: scalar s = (margin * coef).sum(), grad s = J^T coef.
            # Therefore J v requires Rop. Implement via two-step:
            #     g = grad(sum(margin), params)  yields sum_n grad m_n
            # We instead compute per-sample by chain rule via vJp:
            #     J^T (w * (J v))_n  needs (J v)_n.
            # To stay simple, approximate P_S using the Gauss-Newton outer-product
            # estimated by a scalar HVP on (sum_n w_n * m_n^2 / 2):
            #     P_S v ~ Hess of L_margin(theta) applied to v.
            l_margin = 0.5 * (w * margin.pow(2)).sum()
            g_l = self._flat_grad(l_margin, create_graph=True)
            return self._hvp(g_l, v)

        return hvp_preservation

    def _conjugate_gradient(self, b, hvp_combined):
        """Solve (A) x = b using T-step conjugate gradient (Alg. 1 line 9)."""
        x = [torch.zeros_like(p) for p in self.params]
        r = [bi.clone() for bi in b]
        p = [ri.clone() for ri in r]
        rs_old = sum((ri * ri).sum() for ri in r)
        if float(rs_old.detach()) < 1e-20:
            return x

        for _ in range(self.T):
            Ap = hvp_combined(p)
            pAp = sum((pi * ai).sum() for pi, ai in zip(p, Ap))
            alpha = rs_old / (pAp + 1e-12)
            x = [xi + alpha * pi for xi, pi in zip(x, p)]
            r = [ri - alpha * ai for ri, ai in zip(r, Ap)]
            rs_new = sum((ri * ri).sum() for ri in r)
            if float(rs_new.detach().sqrt()) < 1e-8:
                break
            beta = rs_new / (rs_old + 1e-12)
            p = [ri + beta * pi for ri, pi in zip(r, p)]
            rs_old = rs_new
        return x

    def _competing_items(self, users, k=10):
        """
        Items ranked between k and 2k for each user, per Section 5.1.6.
        Returns an item index per user (one competitor sample, kept cheap).
        """
        with torch.no_grad():
            u_all, i_all = self._get_embeddings()
            scores = u_all[users] @ i_all.t()
            topk = torch.topk(scores, 2 * k, dim=1).indices
            # Sample one item from the [k, 2k) slice for each user.
            idx = torch.randint(k, 2 * k, (users.size(0),), device=self.device)
            comp = topk[torch.arange(users.size(0), device=self.device), idx]
        return comp

    def _mask_to_scope(self, deltas, V_S):
        """
        Zero out parameter updates for entities outside V_S so that
        Theta'_{not S} = Theta_{not S} (Eq. 11).
        """
        if self.no_localized or not V_S:
            return deltas

        user_ids = sorted(v for v in V_S if v < self.num_users)
        item_ids = sorted(v - self.num_users for v in V_S
                          if v >= self.num_users)
        user_mask = torch.zeros(self.num_users, dtype=torch.bool,
                                device=self.device)
        item_mask = torch.zeros(self.num_items, dtype=torch.bool,
                                device=self.device)
        if user_ids:
            user_mask[torch.tensor(user_ids, device=self.device)] = True
        if item_ids:
            item_mask[torch.tensor(item_ids, device=self.device)] = True

        masked = []
        for p, d in zip(self.params, deltas):
            if d is None or not p.requires_grad:
                masked.append(d)
                continue
            if p.shape[0] == self.num_users and p.dim() >= 1:
                m = user_mask.view(-1, *([1] * (d.dim() - 1)))
                masked.append(d * m)
            elif p.shape[0] == self.num_items and p.dim() >= 1:
                m = item_mask.view(-1, *([1] * (d.dim() - 1)))
                masked.append(d * m)
            else:
                # Non-entity parameters (e.g. MLP weights) are left untouched.
                masked.append(torch.zeros_like(d))
        return masked

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def unlearn(self):
        logger.info("Executing RARU (Localized Rank-Aware Salience) ...")
        start = time.time()

        # ---- Phase 1: LRAS ----
        V_S, entity_sal, target_set = self._lras_expand()
        logger.info(
            f"LRAS scope: |V_t|={len(target_set)}, |V_S|={len(V_S)}, "
            f"max_scope={self.max_scope_size}"
        )

        # ---- Phase 2: SWRU ----
        # D_S^r: retained interactions in scope.
        edges_local = self._filter_local_edges(V_S)
        # Interaction salience S_ui (Eq. 4).
        S_ui = self._interaction_salience(edges_local, entity_sal, target_set)

        # Target gradient g_t (Eq. 7).
        self.model.zero_grad(set_to_none=False)
        g_t = self._flat_grad(self._bpr_loss(self.data.edges_to_remove),
                              create_graph=False)

        # Combined operator A v = (H_tilde + P_S + lambda I) v.
        # H_tilde via double-backward on BPR loss over D_S^r.
        loss_local = self._bpr_loss(edges_local) if edges_local.numel() > 0 \
            else torch.zeros((), device=self.device, requires_grad=True)
        loss_grads_local = self._flat_grad(loss_local, create_graph=True) \
            if not self.no_curv and edges_local.numel() > 0 else None

        # Preservation matrix P_S via Gauss-Newton on ranking margins.
        if self.no_margin or edges_local.numel() == 0:
            hvp_pres = None
        else:
            users = edges_local[0]
            comp = self._competing_items(users, k=self.k)
            hvp_pres = self._ranking_margin_grad(edges_local, comp, S_ui)

        def hvp_combined(v):
            out = [self.lam * vi for vi in v]
            if loss_grads_local is not None:
                hv = self._hvp(loss_grads_local, v)
                out = [a + b for a, b in zip(out, hv)]
            if hvp_pres is not None:
                hv = hvp_pres(v)
                out = [a + b for a, b in zip(out, hv)]
            return out

        # Solve (Eq. 10).
        delta = self._conjugate_gradient(g_t, hvp_combined)

        # Mask out non-scope parameters (Eq. 11).
        delta = self._mask_to_scope(delta, V_S)

        # Apply parameter update: Theta'_S = Theta_S + DeltaTheta_S.
        # Negative sign because g_t is the gradient of the loss to unlearn:
        # moving along +g_t increases that loss, which demotes the targets.
        unlearned_model = copy.deepcopy(self.model)
        if hasattr(unlearned_model, "clear_cache"):
            unlearned_model.clear_cache()
        with torch.no_grad():
            for p, d in zip(unlearned_model.parameters(), delta):
                if p.requires_grad and d is not None:
                    p.add_(d)

        elapsed = time.time() - start
        logger.info(f"RARU complete. Time: {elapsed:.2f}s.")
        return unlearned_model, self.data.edges_to_remove, elapsed
