"""RARU: Localized Rank-Aware Salience Recommendation Unlearning.

Phase 1 grows the scope V_S from the target entities V_t (Eqs. 5-9); phase 2
solves (Htilde + P_S + lambda I) Delta = g_t (Eq. 16) with T conjugate-gradient
iterations and updates the in-scope embeddings only (Eq. 17).

``--paper_exact`` uses the paper definitions; the default keeps the released
update."""

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


def _flag(args, name):
    """Read an ablation switch, accepting the legacy ``rarl_`` spelling."""
    return bool(args.get(f"raru_{name}", args.get(f"rarl_{name}", False)))


class RARU:
    """Localized Rank-Aware Salience Recommendation Unlearning (Algorithm 1)."""

    def __init__(self, model, data, args, device):
        self.model = model
        self.data = data
        self.args = args
        self.device = device

        # Trainable parameters of the backbone model: Theta in the paper.
        self.params = [p for p in self.model.parameters() if p.requires_grad]

        # k: scope cutoff (Eq. 8) and C_u window; lambda: damping (Eq. 16).
        self.k = int(args.get("scope_cutoff_k", args.get("k", 10)))
        self.lam = float(args.get("damping_lambda", 1e-2))
        self.T = int(args.get("cg_steps", 30))

        # Memory safeguard; ignored under --paper_exact.
        self.max_scope_size = int(args.get("rarl_max_scope_size", 5000))

        # --paper_exact: paper-faithful semantics (see module docstring).
        self.exact = bool(args.get("paper_exact", False))

        # Ablation switches (Table 4); rarl_* are accepted as aliases.
        self.no_sconn = _flag(args, "no_sconn")
        self.no_sim = _flag(args, "no_sim")
        self.no_localized = _flag(args, "no_localized")
        self.no_sw = _flag(args, "no_salience_weight")
        self.no_margin = _flag(args, "no_margin")
        self.no_curv = _flag(args, "no_curvature")

        self.num_users = data.num_users
        self.num_items = data.num_items

        backbone = str(args.get("backbone", "lightgcn")).lower()
        self.backbone = backbone
        self.is_gnn = determine_is_gnn(backbone)

        # Backbone objective: BCE for NeuMF/WMF, BPR otherwise.
        self.loss_type = "bce" if backbone in ("neumf", "wmf") else "bpr"

        self._user_items = None
        self._entity_specs = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_embeddings(self):
        """Return (user_emb, item_emb) of the current graph."""
        if self.is_gnn:
            return self.model(edge_index=self.data.train_edge_index_after_remove)
        return self.model.get_embeddings()

    def _node_to_user_item(self, node_id):
        """Translate a global node id to ('user'/'item', local index)."""
        if node_id < self.num_users:
            return "user", int(node_id)
        return "item", int(node_id - self.num_users)

    def _build_adjacency(self, edge_index, symmetrize=False):
        """Adjacency lists over the retained graph, keyed by global node id."""
        adj = [[] for _ in range(self.num_users + self.num_items)]
        src = edge_index[0].cpu().tolist()
        dst = edge_index[1].cpu().tolist()
        for u, v in zip(src, dst):
            adj[u].append(v)
        if symmetrize:
            for u, v in zip(src, dst):
                adj[v].append(u)
        return adj

    def _user_items_map(self):
        """{user_id: set(item_idx)} over the original training interactions."""
        if self._user_items is None:
            if hasattr(self.data, "train_edge_index_original"):
                edges = self.data.train_edge_index_original
            else:
                edges = self.data.train_edge_index
            mapping = {}
            users = edges[0].cpu().tolist()
            items = (edges[1] - self.num_users).cpu().tolist()
            for u, i in zip(users, items):
                if 0 <= i < self.num_items:
                    mapping.setdefault(u, set()).add(i)
            self._user_items = mapping
        return self._user_items

    def _entity_param_specs(self):
        """Map parameter index -> (kind, valid_rows) for embedding tables."""
        if self._entity_specs is not None:
            return self._entity_specs

        table_names = (
            "user_emb", "item_emb",
            "user_gmf_emb", "item_gmf_emb",
            "user_mlp_emb", "item_mlp_emb",
        )
        specs = {}
        for name in table_names:
            module = getattr(self.model, name, None)
            weight = getattr(module, "weight", None)
            if weight is None:
                continue
            kind = "user" if name.startswith("user") else "item"
            rows = self.num_users if kind == "user" else self.num_items
            for p_idx, p in enumerate(self.params):
                if p is weight:
                    specs[p_idx] = (kind, rows)
        self._entity_specs = specs
        return specs

    def _scope_mask(self, V_S):
        """0/1 mask over Theta that keeps the in-scope embedding rows (Eq. 17)."""
        if self.no_localized or not V_S:
            return [torch.ones_like(p) for p in self.params]

        allowed = {
            "user": torch.zeros(self.num_users, device=self.device),
            "item": torch.zeros(self.num_items, device=self.device),
        }
        for v in V_S:
            if v < self.num_users:
                allowed["user"][v] = 1.0
            else:
                allowed["item"][v - self.num_users] = 1.0

        specs = self._entity_param_specs()
        masks = []
        for p_idx, p in enumerate(self.params):
            spec = specs.get(p_idx)
            if spec is None:
                masks.append(torch.zeros_like(p))
                continue
            kind, rows = spec
            row_mask = torch.zeros(p.shape[0], device=self.device)
            row_mask[:min(rows, p.shape[0])] = allowed[kind][:min(rows, p.shape[0])]
            masks.append(
                row_mask.view(-1, *([1] * (p.dim() - 1))).expand_as(p)
            )
        return masks

    @staticmethod
    def _project(v, mask):
        """Restrict a parameter-space vector to the Theta_S subspace."""
        return [vi * mi for vi, mi in zip(v, mask)]

    def _sample_negatives(self, edges):
        """Draw one negative item per interaction, excluding seen items."""
        users = edges[0]
        pos_items = edges[1] - self.num_users
        num_edges = users.size(0)
        neg = torch.randint(
            0, self.num_items, (num_edges,), device=self.device
        )
        mapping = self._user_items_map()
        users_list = users.cpu().tolist()
        pos_list = pos_items.cpu().tolist()
        neg_list = neg.cpu().tolist()
        for idx, u in enumerate(users_list):
            seen = mapping.get(u, ())
            if neg_list[idx] in seen or neg_list[idx] == pos_list[idx]:
                for _ in range(50):
                    candidate = int(
                        torch.randint(0, self.num_items, (1,)).item()
                    )
                    if candidate not in seen and candidate != pos_list[idx]:
                        neg_list[idx] = candidate
                        break
        return torch.tensor(neg_list, dtype=torch.long, device=self.device)

    def _score_pairs(self, edges, neg_items):
        """Positive and negative scores of a set of interactions."""
        users = edges[0]
        pos_items = edges[1] - self.num_users
        u_all, i_all = self._get_embeddings()
        u_e = u_all[users]
        pos_score = (u_e * i_all[pos_items]).sum(dim=1)
        neg_score = (u_e * i_all[neg_items]).sum(dim=1)
        return pos_score, neg_score

    def _training_loss(self, edges, neg_items):
        """L(D_t; Theta) of Eq. (10) for the backbone's training objective."""
        if edges.numel() == 0:
            return torch.zeros((), device=self.device, requires_grad=True)
        pos_score, neg_score = self._score_pairs(edges, neg_items)
        if self.loss_type == "bce":
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_score, torch.ones_like(pos_score)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_score, torch.zeros_like(neg_score)
            )
            return 0.5 * (pos_loss + neg_loss)
        return -F.logsigmoid(pos_score - neg_score).mean()

    def _curvature_terms(self, pos_score, neg_score):
        """Per-term scores and Gauss-Newton weights of Eq. (14)."""
        if self.loss_type == "bce":
            sig_pos = torch.sigmoid(pos_score)
            sig_neg = torch.sigmoid(neg_score)
            terms = torch.cat([pos_score, neg_score])
            weights = torch.cat(
                [sig_pos * (1.0 - sig_pos), sig_neg * (1.0 - sig_neg)]
            ).detach()
            return terms, weights
        margin = pos_score - neg_score
        weights = (torch.sigmoid(margin) * torch.sigmoid(-margin)).detach()
        return margin, weights

    def _preservation_terms(self, edges, comp_items, weights):
        """Terms of P_S (Eq. 13); C_u is padded with -1."""
        users = edges[0]
        pos_items = edges[1] - self.num_users
        u_all, i_all = self._get_embeddings()
        u_e = u_all[users]
        pos_score = (u_e * i_all[pos_items]).sum(dim=1)

        valid = comp_items >= 0
        comp_scores = (
            u_e.unsqueeze(1) * i_all[comp_items.clamp(min=0)]
        ).sum(dim=2)
        margins = pos_score.unsqueeze(1) - comp_scores
        terms = margins[valid]
        term_weights = weights.unsqueeze(1).expand_as(margins)[valid].detach()
        return terms, term_weights

    def _gn_hvp(self, terms, weights, v):
        """Gauss-Newton matrix-vector product J^T diag(w) J v."""
        if terms.numel() == 0:
            return [torch.zeros_like(p) for p in self.params]

        phi = torch.zeros_like(terms, requires_grad=True)
        jt_phi = self._flat_grad((terms * phi).sum(), create_graph=True)
        scalar = sum((g * vi).sum() for g, vi in zip(jt_phi, v))
        # The graph is reused by every CG iteration.
        jv = grad(scalar, phi, retain_graph=True)[0]
        hvp = self._flat_grad((terms * (weights * jv)).sum())
        return hvp

    # ------------------------------------------------------------------
    # Phase 1: Salience-Guided Scope Localization (Section 4.1)
    # ------------------------------------------------------------------
    def _compute_entity_salience(self, candidates, target_entities,
                                 degrees, max_deg, user_emb, item_emb):
        """Salience S(v) of each candidate (Eqs. 5-7)."""
        salience = {}

        # Stack target embeddings split by entity type for the similarity term.
        t_users = [t for t in target_entities if t < self.num_users]
        t_items = [t - self.num_users for t in target_entities
                   if t >= self.num_users]
        target_user_emb = (user_emb[torch.tensor(t_users, device=self.device)]
                           if t_users else None)
        target_item_emb = (item_emb[torch.tensor(t_items, device=self.device)]
                           if t_items else None)

        # Pre-normalize for cosine similarity (Eq. 6).
        if target_user_emb is not None:
            target_user_emb = F.normalize(target_user_emb, dim=1)
        if target_item_emb is not None:
            target_item_emb = F.normalize(target_item_emb, dim=1)

        # deg_max in Eq. (5) is the largest degree in the retained graph.
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

    def _sgsl_expand(self):
        """Algorithm 1, phase 1: grow V_S from V_t."""
        edges_to_remove = self.data.edges_to_remove
        if edges_to_remove.numel() == 0:
            return set(), {}, set()

        # V_t (Eq. 4); the default keeps only items for entity unlearning.
        unl_task = self.args.get("unlearning_task", "interaction")
        if unl_task == "item" and not self.exact:
            target_set = set(edges_to_remove[1].cpu().tolist())  # item nodes
        else:
            target_set = set(edges_to_remove[0].cpu().tolist()) | \
                         set(edges_to_remove[1].cpu().tolist())

        # Build adjacency over the retained graph D_r.
        adj = self._build_adjacency(
            self.data.train_edge_index_after_remove,
            symmetrize=self.exact,
        )
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

        # S(v) for the initial frontier and the threshold tau = S_max / k
        # (Eq. 8).
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

        # Best-first expansion until the salience drops below tau.
        max_size = None if self.exact else self.max_scope_size
        while heap and (max_size is None or len(V_S) < max_size):
            neg_s, v = heapq.heappop(heap)
            s = -neg_s
            if s < tau:
                break
            V_S.add(v)

            # Insert unvisited retained neighbors with their salience.
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

        # Target entities carry S_bar = 1 (Eq. 7).
        for t in target_set:
            entity_sal[t] = 1.0

        return V_S, entity_sal, target_set

    # ------------------------------------------------------------------
    # Phase 2: Salience-Weighted Ranking Unlearning (Section 4.2)
    # ------------------------------------------------------------------
    def _filter_local_edges(self, V_S):
        """D_S^r: retained interactions inside the scope (Eq. 9)."""
        if self.no_localized or not V_S:
            edges = self.data.train_edge_index_after_remove
        else:
            V_S_tensor = torch.tensor(sorted(V_S), dtype=torch.long,
                                      device=self.device)
            e = self.data.train_edge_index_after_remove
            mask = torch.isin(e[0], V_S_tensor) | torch.isin(e[1], V_S_tensor)
            edges = e[:, mask]

        # Keep only the user -> item direction (Eq. 9).
        forward = (edges[0] < self.num_users) & (edges[1] >= self.num_users)
        edges = edges[:, forward]
        return edges

    def _interaction_salience(self, edges, entity_sal, target_set):
        """Eq. (7): S_ui = (S_bar(u) + S_bar(i)) / 2."""
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
        """BPR loss used by the default update (g_t and the curvature term)."""
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
        """Default-mode P_S: exact Hessian of 0.5 * sum S_ui m_uij^2."""
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
            return margin, weights

        def hvp_preservation(v):
            margin, w = margin_value()
            l_margin = 0.5 * (w * margin.pow(2)).sum()
            g_l = self._flat_grad(l_margin, create_graph=True)
            return self._hvp(g_l, v)

        return hvp_preservation

    def _conjugate_gradient(self, b, hvp_combined):
        """Solve A x = b with T conjugate-gradient iterations (Alg. 1)."""
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

    def _conjugate_gradient_scoped(self, b, hvp_combined, mask):
        """Conjugate gradient restricted to the Theta_S subspace (Eq. 16)."""
        def apply(v):
            return self._project(hvp_combined(self._project(v, mask)), mask)

        x = [torch.zeros_like(p) for p in self.params]
        r = [bi.clone() for bi in b]
        p = [ri.clone() for ri in r]
        rs_old = sum((ri * ri).sum() for ri in r)
        if float(rs_old.detach()) < 1e-20:
            return x

        for _ in range(self.T):
            Ap = apply(p)
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
        """Default-mode C_u: one random competitor per user."""
        with torch.no_grad():
            u_all, i_all = self._get_embeddings()
            scores = u_all[users] @ i_all.t()
            topk = torch.topk(scores, 2 * k, dim=1).indices
            # Sample one item from the [k, 2k) slice for each user.
            idx = torch.randint(k, 2 * k, (users.size(0),), device=self.device)
            comp = topk[torch.arange(users.size(0), device=self.device), idx]
        return comp

    def _competing_sets(self, edges):
        """C_u of Section 4.2, padded with -1."""
        k = self.k
        users = edges[0]
        uniq, inverse = torch.unique(users, return_inverse=True)
        max_c = min(2 * k, self.num_items)
        rows = torch.full(
            (uniq.numel(), max_c), -1, dtype=torch.long, device=self.device
        )
        mapping = self._user_items_map()

        with torch.no_grad():
            if self.is_gnn:
                u_all, i_all = self.model(
                    edge_index=self.data.train_edge_index_after_remove
                )
            else:
                u_all, i_all = self.model.get_embeddings()

            # Bound the score matrix to roughly 4M entries.
            batch_size = max(1, min(256, 4_000_000 // max(self.num_items, 1)))
            for start in range(0, uniq.numel(), batch_size):
                batch_users = uniq[start:start + batch_size]
                scores = u_all[batch_users] @ i_all.t()
                seen = torch.zeros(
                    batch_users.numel(), self.num_items,
                    dtype=torch.bool, device=self.device,
                )
                for row, user in enumerate(batch_users.cpu().tolist()):
                    items = mapping.get(user)
                    if items:
                        seen[row, torch.tensor(sorted(items),
                                               device=self.device)] = True
                scores = scores.masked_fill(seen, float("-inf"))
                top = torch.topk(scores, max_c, dim=1)
                valid = torch.isfinite(top.values)
                rows[start:start + batch_users.numel()] = \
                    top.indices.masked_fill(~valid, -1)

        return rows[inverse]

    def _mask_to_scope(self, deltas, V_S):
        """Zero out the updates of entities outside V_S (Eq. 17)."""
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
            if p.dim() >= 1 and p.shape[0] == self.num_users:
                m = user_mask.view(-1, *([1] * (d.dim() - 1)))
                masked.append(d * m)
            elif p.dim() >= 1 and p.shape[0] == self.num_items:
                m = item_mask.view(-1, *([1] * (d.dim() - 1)))
                masked.append(d * m)
            else:
                # Non-entity parameters (e.g. MLP weights) are left untouched.
                masked.append(torch.zeros_like(d))
        return masked

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def _default_delta(self, V_S, edges_local, S_ui):
        """Parameter update of the released implementation (see module doc)."""
        # Target gradient g_t (Eq. 10), computed on D_t.
        self.model.zero_grad(set_to_none=False)
        g_t = self._flat_grad(self._bpr_loss(self.data.edges_to_remove),
                              create_graph=False)

        # Htilde of the default path: full BPR Hessian over D_S^r.
        loss_local = self._bpr_loss(edges_local) if edges_local.numel() > 0 \
            else torch.zeros((), device=self.device, requires_grad=True)
        loss_grads_local = self._flat_grad(loss_local, create_graph=True) \
            if not self.no_curv and edges_local.numel() > 0 else None

        # P_S of the default path.
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

        # Solve (Eq. 16).
        delta = self._conjugate_gradient(g_t, hvp_combined)

        # Theta'_{\bar S} = Theta_{\bar S} (Eq. 17).
        return self._mask_to_scope(delta, V_S)

    def _exact_delta(self, V_S, edges_local, S_ui):
        """Parameter update of the paper (Gauss-Newton, Theta_S subspace)."""
        mask = self._scope_mask(V_S)

        # g_t = grad_{Theta_S} L(D_t; Theta) of Eq. (10).
        target_neg = self._sample_negatives(self.data.edges_to_remove)
        g_t = self._flat_grad(
            self._training_loss(self.data.edges_to_remove, target_neg)
        )

        # Htilde_{r,S} (Eq. 14), formed from the retained interactions of D_S.
        curv_terms = curv_weights = None
        if not self.no_curv and edges_local.numel() > 0:
            local_neg = self._sample_negatives(edges_local)
            pos_score, neg_score = self._score_pairs(edges_local, local_neg)
            curv_terms, curv_weights = self._curvature_terms(
                pos_score, neg_score
            )

        # P_S (Eq. 13), the Gauss-Newton matrix of the ranking penalty.
        pres_terms = pres_weights = None
        if not self.no_margin and edges_local.numel() > 0:
            comp = self._competing_sets(edges_local)
            pres_terms, pres_weights = self._preservation_terms(
                edges_local, comp, S_ui
            )

        def hvp_combined(v):
            out = [self.lam * vi for vi in v]
            if curv_terms is not None:
                hv = self._gn_hvp(curv_terms, curv_weights, v)
                out = [a + b for a, b in zip(out, hv)]
            if pres_terms is not None:
                hv = self._gn_hvp(pres_terms, pres_weights, v)
                out = [a + b for a, b in zip(out, hv)]
            return out

        # Solve (Eq. 16) on the Theta_S subspace.
        rhs = self._project(g_t, mask)
        return self._conjugate_gradient_scoped(rhs, hvp_combined, mask)

    def unlearn(self):
        logger.info(
            "Executing RARU (%s) ...",
            "paper-exact" if self.exact else "default",
        )
        start = time.time()

        # Deterministic model: dropout would make g_t stochastic.
        was_training = self.model.training
        self.model.eval()

        # ---- Phase 1: SGSL (Section 4.1, Algorithm 1 lines 1-7) ----
        V_S, entity_sal, target_set = self._sgsl_expand()
        total_entities = max(self.num_users + self.num_items, 1)
        total_edges = max(self.data.train_edge_index_after_remove.size(1), 1)
        logger.info(
            f"SGSL scope: |V_t|={len(target_set)}, |V_S|={len(V_S)} "
            f"({len(V_S) / total_entities:.4f} of entities)"
        )

        # ---- Phase 2: SWRU (Section 4.2, Algorithm 1 lines 8-12) ----
        # Drop the cache left by the no_grad forward pass.
        if hasattr(self.model, "clear_cache"):
            self.model.clear_cache()

        # D_S^r: retained interactions of the localized scope (Eq. 9).
        edges_local = self._filter_local_edges(V_S)
        logger.info(
            f"Localized scope: |D_S^r|={edges_local.size(1)} "
            f"({edges_local.size(1) / total_edges:.4f} of retained edges)"
        )
        # Interaction salience S_ui (Eq. 7).
        S_ui = self._interaction_salience(edges_local, entity_sal, target_set)

        if self.exact:
            delta = self._exact_delta(V_S, edges_local, S_ui)
        else:
            delta = self._default_delta(V_S, edges_local, S_ui)

        # Theta'_S = Theta_S + Delta; +g_t raises the loss on D_t.
        # A cached graph cannot be deep-copied and is stale after the update.
        if hasattr(self.model, "clear_cache"):
            self.model.clear_cache()
        unlearned_model = copy.deepcopy(self.model)
        if hasattr(unlearned_model, "clear_cache"):
            unlearned_model.clear_cache()
        with torch.no_grad():
            for p, d in zip(unlearned_model.parameters(), delta):
                if p.requires_grad and d is not None:
                    p.add_(d)

        elapsed = time.time() - start
        if was_training:
            self.model.train()
        logger.info(f"RARU complete. Time: {elapsed:.2f}s.")
        return unlearned_model, self.data.edges_to_remove, elapsed
