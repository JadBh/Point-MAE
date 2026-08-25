import torch
import torch.nn as nn


class KnnMinimal(nn.Module):
    """
    Finds the k nearest reference points for each query point.
    """

    def __init__(self, k, transpose_mode=True):
        """
        k: Neighbors to return per query.
        transpose_mode: If True, points are (B, N, C). If False, they're
            channels-first (B, C, N) and outputs are transposed to match.

        """
        super().__init__()
        self.k = k
        self.transpose_mode = transpose_mode

    @torch.no_grad()
    def forward(self, ref, query):
        """
        ref   (B, N, C): points to search over
        query (B, M, C): points to find neighbors for (clusters centers)

        returns: dist, idx — both (B, M, k), sorted nearest-first.
        'idx' indexes into N, so 'ref[b, idx[b, m]]' are query (b, m)'s neighbors
        """
        if not self.transpose_mode:
            ref = ref.transpose(-2, -1)  # (B, C, N) -> (B, N, C)
            query = query.transpose(-2, -1)

        dist = torch.cdist(query, ref)  # (B, M, N)
        dist, idx = dist.topk(self.k, dim=-1, largest=False)  # k smallest, ascending

        if not self.transpose_mode:
            dist = dist.transpose(-2, -1)  # (B, M, k) -> (B, k, M)
            idx = idx.transpose(-2, -1)
        return dist, idx



class BallQuerySampler(nn.Module):
    """
    For each query point (patch center), samples k reference points uniformly at
    random from those lying within a radius of the center.

    Unlike plain k-NN, the patch extent is set by geometry (how far apart the
    centers are) rather than by point density, so patches keep covering the same
    amount of space as the cloud grows and the gaps between patches close up.
    """

    def __init__(self, k, radius_scale=1.0, num_neighbor_centers=3, transpose_mode=True):
        """
        k: points returned per center.
        radius_scale: radius as a fraction of the mean spacing between centers.
        num_neighbor_centers: how many nearby centers the spacing is averaged over.
        """
        super().__init__()
        self.k = k
        self.radius_scale = radius_scale
        self.num_neighbor_centers = num_neighbor_centers
        self.transpose_mode = transpose_mode

    def _estimate_radius(self, centers):
        """Mean distance from each center to its nearest other centers, scaled. (B, M, 1)"""
        center_distances = torch.cdist(centers, centers)  # (B, M, M)
        num_neighbors = min(self.num_neighbor_centers, centers.shape[1] - 1)
        # +1 slot because the nearest "other" center is the center itself, at distance 0
        nearest, _ = center_distances.topk(num_neighbors + 1, dim=-1, largest=False)
        mean_spacing = nearest[..., 1:].mean(dim=-1, keepdim=True)
        return self.radius_scale * mean_spacing

    @torch.no_grad()
    def forward(self, ref, query):
        """
        ref   (B, N, C): points to search over
        query (B, M, C): patch centers
        returns: dist, idx — both (B, M, k), sorted nearest-first.
        """
        if not self.transpose_mode:
            ref = ref.transpose(-2, -1)
            query = query.transpose(-2, -1)

        point_distances = torch.cdist(query, ref)  # (B, M, N)
        radius = self._estimate_radius(query)  # (B, M, 1)
        inside_ball = point_distances <= radius

        # Rank candidates by a sort key, then take the k smallest keys:
        #   - points inside the ball get a uniform random key in [0, 1), so the
        #     k smallest are a uniform sample without replacement;
        #   - points outside get a key >= 1 ordered by distance, so they only
        #     fill leftover slots (nearest first) when a ball holds fewer than k.
        max_distance = point_distances.amax(dim=-1, keepdim=True).clamp(min=1e-12)
        random_key = torch.rand_like(point_distances)
        fallback_key = 1.0 + point_distances / max_distance
        sampling_key = torch.where(inside_ball, random_key, fallback_key)

        _, idx = sampling_key.topk(self.k, dim=-1, largest=False)  # (B, M, k)
        dist = point_distances.gather(-1, idx)

        # keep the nearest-first ordering the k-NN version guaranteed
        dist, order = dist.sort(dim=-1)
        idx = idx.gather(-1, order)

        if not self.transpose_mode:
            dist = dist.transpose(-2, -1)
            idx = idx.transpose(-2, -1)
        return dist, idx