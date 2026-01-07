import torch

def kabsch(A, B):
        """
        Kabsch algorithm for optimal rotation/translation.

        Parameters
        ----------
        A, B : torch.Tensor
            Coordinate sets, shape [batch, n_atoms, 3]

        Returns
        -------
        A_aligned : torch.Tensor
            Aligned coords of set A
        R : torch.Tensor
            Rotation matrix
        t : torch.Tensor
            Translation matrix
        """
        a_mean = A.mean(dim=1, keepdims=True).type('torch.DoubleTensor')
        b_mean = B.mean(dim=1, keepdims=True).type('torch.DoubleTensor')
        A_c = A - a_mean
        B_c = B - b_mean
        # Covariance matrix
        H = torch.bmm(A_c.transpose(1,2), B_c)  # [B, 3, 3]
        U, S, V = torch.svd(H)
        # Rotation matrix
        R = torch.bmm(V, U.transpose(1,2))  # [B, 3, 3]
        # Translation vector
        t = b_mean - torch.bmm(R, a_mean.transpose(1,2)).transpose(1,2)
        A_aligned = torch.bmm(R, A.transpose(1,2)).transpose(1,2) + t
        return A_aligned, R, t
