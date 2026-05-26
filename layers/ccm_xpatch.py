import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CCMClusterAssigner(nn.Module):
    """
    CCM-style cluster assigner for xPatch with prototype learning.

    Input:
        x: [B, L, C]

    Output:
        prob:        [B, C, K]  final clustering probability
        membership:  [B, C, K]  approximate binary membership from final prob
        sim_matrix:  [B, C, C]
        channel_emb: [B, C, D]
        cluster_emb: [B, K, D]  updated prototype embedding

    This version is closer to the CCM paper:
        1. compute channel embeddings H from normalized historical series;
        2. compute initial P using learnable cluster embeddings C;
        3. sample initial membership M from initial P;
        4. update cluster prototypes with masked cross-attention:
              C_hat = Normalize(exp(QK^T/sqrt(d)) * M^T) V
        5. compute final P again using updated prototypes C_hat;
        6. compute final membership and RBF similarity for ClusterLoss.
    """

    def __init__(
        self,
        seq_len,
        n_cluster=2,
        d_model=64,
        sigma=5.0,
        epsilon=0.2,
        gumbel_temp=0.5,
        use_gumbel=False,
        dropout=0.0,
    ):
        super(CCMClusterAssigner, self).__init__()

        self.seq_len = seq_len
        self.n_cluster = n_cluster
        self.d_model = d_model
        self.sigma = sigma
        self.epsilon = epsilon
        self.gumbel_temp = gumbel_temp
        self.use_gumbel = use_gumbel

        # Paper Table 10 says ETTh1 uses 1 linear layer in MLP.
        # Your previous implementation used 2 Linear layers.
        # To avoid changing too much at once, we keep the same encoder structure.
        self.channel_encoder = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Initial learnable cluster embeddings C: [K, D]
        self.cluster_emb = nn.Parameter(torch.randn(n_cluster, d_model))
        nn.init.xavier_uniform_(self.cluster_emb)

        self.norm = nn.LayerNorm(d_model)

        # Prototype learning cross-attention projections.
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.prototype_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def _standardize_time(self, x):
        """
        Standardize each channel along temporal dimension.

        x: [B, L, C]
        """
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + 1e-5)

    def _compute_rbf_similarity(self, x):
        """
        S_ij = exp(-||Xi - Xj||^2 / (2 sigma^2))

        x: [B, L, C]
        return: [B, C, C]
        """
        x = self._standardize_time(x)
        x = x.permute(0, 2, 1).contiguous()  # [B, C, L]

        diff = x.unsqueeze(2) - x.unsqueeze(1)  # [B, C, C, L]
        dist2 = (diff ** 2).mean(dim=-1)        # [B, C, C]

        sim = torch.exp(-dist2 / (2.0 * self.sigma * self.sigma + 1e-8))
        return sim

    def _gumbel_sigmoid(self, prob):
        """
        Your previous implementation used independent Gumbel-Sigmoid per cluster.
        We keep it for compatibility with existing configs.

        Note:
            Since P is a softmax distribution over K, this does not enforce
            one-hot membership exactly. It gives an approximate binary mask.
        """
        eps = 1e-8
        prob = prob.clamp(min=eps, max=1.0 - eps)

        if not self.training or not self.use_gumbel:
            return prob

        u = torch.rand_like(prob).clamp(min=eps, max=1.0 - eps)
        noise = torch.log(u) - torch.log(1.0 - u)
        logit = torch.log(prob) - torch.log(1.0 - prob)

        return torch.sigmoid((logit + noise) / self.gumbel_temp)

    def _compute_prob(self, channel_emb, cluster_emb):
        """
        Compute normalized inner-product probability.

        Args:
            channel_emb:
                [B, C, D]
            cluster_emb:
                [K, D] or [B, K, D]

        Returns:
            prob:
                [B, C, K]
        """
        channel_emb = F.normalize(channel_emb, p=2, dim=-1)

        if cluster_emb.dim() == 2:
            cluster_emb = F.normalize(cluster_emb, p=2, dim=-1)
            logits = torch.einsum("bcd,kd->bck", channel_emb, cluster_emb)
        elif cluster_emb.dim() == 3:
            cluster_emb = F.normalize(cluster_emb, p=2, dim=-1)
            logits = torch.einsum("bcd,bkd->bck", channel_emb, cluster_emb)
        else:
            raise ValueError("cluster_emb must be [K, D] or [B, K, D]")

        prob = torch.softmax(logits / max(self.epsilon, 1e-6), dim=-1)
        return prob

    def _prototype_learning(self, channel_emb, init_membership):
        """
        Masked cross-attention prototype update.

        Paper Eq.3:
            C_hat = Normalize(
                exp((W_Q C)(W_K H)^T / sqrt(d)) * M^T
            ) W_V H

        Args:
            channel_emb:
                H, [B, C, D]
            init_membership:
                M, [B, C, K]

        Returns:
            updated_cluster_emb:
                C_hat, [B, K, D]
        """
        B, C, D = channel_emb.shape

        # Broadcast initial cluster embeddings to batch.
        cluster_emb = self.cluster_emb.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]

        q = self.w_q(cluster_emb)    # [B, K, D]
        k = self.w_k(channel_emb)    # [B, C, D]
        v = self.w_v(channel_emb)    # [B, C, D]

        # [B, K, C]
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)

        # exp(attn_logits) * M^T
        # M^T: [B, K, C]
        mask = init_membership.transpose(-1, -2).clamp(min=0.0)
        attn = torch.exp(attn_logits) * mask

        # Normalize over channels for each cluster.
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
        attn = self.attn_dropout(attn)

        # [B, K, D]
        updated_cluster_emb = torch.matmul(attn, v)

        # Residual connection is not explicitly written in the paper,
        # but helps avoid empty/weak cluster prototypes early in training.
        # If you want stricter reproduction, remove "+ cluster_emb".
        updated_cluster_emb = self.prototype_norm(updated_cluster_emb + cluster_emb)
        updated_cluster_emb = F.normalize(updated_cluster_emb, p=2, dim=-1)

        return updated_cluster_emb

    def forward(self, x):
        """
        x: [B, L, C]
        """
        # normalized raw input for channel embedding
        x_norm = self._standardize_time(x)             # [B, L, C]
        z = x_norm.permute(0, 2, 1).contiguous()       # [B, C, L]

        channel_emb = self.channel_encoder(z)          # [B, C, D]
        channel_emb = self.norm(channel_emb)
        channel_emb = F.normalize(channel_emb, p=2, dim=-1)

        # 1. Initial probability from learnable cluster embeddings.
        init_prob = self._compute_prob(channel_emb, self.cluster_emb)  # [B, C, K]

        # 2. Initial membership for masked prototype learning.
        init_membership = self._gumbel_sigmoid(init_prob)              # [B, C, K]

        # 3. Update prototypes by masked cross-attention.
        updated_cluster_emb = self._prototype_learning(
            channel_emb=channel_emb,
            init_membership=init_membership,
        )  # [B, K, D]

        # 4. Final probability using updated prototypes.
        prob = self._compute_prob(channel_emb, updated_cluster_emb)     # [B, C, K]

        # 5. Final membership for cluster loss.
        membership = self._gumbel_sigmoid(prob)

        # 6. Similarity matrix for ClusterLoss.
        sim_matrix = self._compute_rbf_similarity(x)

        return prob, membership, sim_matrix, channel_emb, updated_cluster_emb


class ClusterAwareLinear(nn.Module):
    """
    Cluster-aware linear projection.

    For each cluster k, we have an independent Linear layer.
    For each channel i, output is weighted by prob[i,k]:

        y_i = sum_k p_{i,k} * Linear_k(x_i)

    Input:
        x:    [B, C, in_dim]
        prob: [B, C, K]

    Output:
        y:    [B, C, out_dim]
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        n_cluster=2,
        dropout=0.0,
        bias=True,
    ):
        super(ClusterAwareLinear, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_cluster = n_cluster

        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=bias)
            for _ in range(n_cluster)
        ])

    def forward(self, x, prob):
        """
        x: [B, C, in_dim]
        prob: [B, C, K]
        """
        x = self.dropout(x)

        outs = []
        for head in self.heads:
            outs.append(head(x))  # [B, C, out_dim]

        out = torch.stack(outs, dim=2)  # [B, C, K, out_dim]
        y = (prob.unsqueeze(-1) * out).sum(dim=2)

        return y


class ClusterAwareSeasonalHead(nn.Module):
    """
    Cluster-aware seasonal prediction head.

    Original xPatch seasonal head:
        flatten -> fc3 -> GELU -> fc4

    This version:
        flatten -> cluster-aware fc3 -> GELU -> cluster-aware fc4
    """

    def __init__(
        self,
        in_dim,
        pred_len,
        n_cluster=2,
        dropout=0.0,
    ):
        super(ClusterAwareSeasonalHead, self).__init__()

        self.fc1 = ClusterAwareLinear(
            in_dim=in_dim,
            out_dim=pred_len * 2,
            n_cluster=n_cluster,
            dropout=dropout,
        )
        self.act = nn.GELU()
        self.fc2 = ClusterAwareLinear(
            in_dim=pred_len * 2,
            out_dim=pred_len,
            n_cluster=n_cluster,
            dropout=dropout,
        )

    def forward(self, x, prob):
        """
        x: [B, C, flatten_dim]
        prob: [B, C, K]
        """
        x = self.fc1(x, prob)
        x = self.act(x)
        x = self.fc2(x, prob)
        return x


class CCMClusterLoss(nn.Module):
    """
    Cluster loss for CCM-style training.

    same_cluster = M M^T
    intra = mean similarity among same-cluster pairs
    inter = mean similarity among different-cluster pairs

    loss = -intra + inter
    """

    def __init__(self):
        super(CCMClusterLoss, self).__init__()

    def forward(self, sim_matrix, membership):
        """
        sim_matrix: [B, C, C]
        membership: [B, C, K]
        """
        B, C, _ = sim_matrix.shape

        M = membership
        same_cluster = torch.matmul(M, M.transpose(-1, -2))  # [B, C, C]

        eye = torch.eye(C, device=sim_matrix.device, dtype=sim_matrix.dtype).unsqueeze(0)
        pair_mask = 1.0 - eye

        S = sim_matrix * pair_mask
        same_cluster = same_cluster * pair_mask

        intra = (same_cluster * S).sum(dim=(-1, -2)) / (
            same_cluster.sum(dim=(-1, -2)) + 1e-6
        )

        diff_cluster = (1.0 - same_cluster) * pair_mask
        inter = (diff_cluster * S).sum(dim=(-1, -2)) / (
            diff_cluster.sum(dim=(-1, -2)) + 1e-6
        )

        loss = -intra + inter
        return loss.mean()
