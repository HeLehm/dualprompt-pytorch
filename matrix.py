import torch
import torch.nn as nn
import warnings

class BatchSymmetricPositiveDefiniteMatrix(nn.Module):
    """
    Stored a batch of symmetric positive definite matrices
    """
    def __init__(self,batch_size, n, jitter = 1e-6 ,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n = n
        self.batch_size = batch_size
        self.jitter_value = jitter
        self.jitter = nn.Parameter(jitter * torch.eye(self.n).expand(self.batch_size, self.n, self.n), requires_grad=False)
        self.tril = nn.Parameter(
            torch.randn(batch_size, n * (n + 1) // 2)
        )

    def set_matrix_at(self, matrix, i):
        """
        Set the matrix for a ceratin batch index
        """
        matrix = matrix.to(self.tril.device)
        # check if matri is square
        assert matrix.shape[-1] == matrix.shape[-2], f"Expected matrix to be square"
        # check if shape is correct
        assert matrix.shape[-1] == self.n, f"Expected matrix to have shape (batch_size, {self.n}, {self.n}), got {matrix.shape}"
        # check if matrix is symmetric
        assert torch.allclose(matrix, matrix.transpose(-1, -2)), f"Expected matrix to be symmetric"
        # check if matrix is positive definite
        if not torch.all(torch.linalg.eigvalsh(matrix) > 0):
            warnings.warn("Matrix is not positive definite, adding jitter to diagonal")
            matrix = matrix + torch.eye(self.n).to(matrix.device) * self.jitter_value
        
        # set the matrix
        # use cholesky decomposition, so tril @ tril.transpose(-2, -1) will be the matrix
        matrix = torch.linalg.cholesky(matrix)

        new_trill_data = self._matrix_to_tril(matrix)

        # set the right index
        self.tril.data[i] = new_trill_data

    
    def set_matrix(self, matrix):
        """
        Set the matrix to be the given matrix
        matrix should have shape (batch_size, n, n)
        """
        matrix = matrix.to(self.tril.device)
        # check if matri is square
        assert matrix.shape[-1] == matrix.shape[-2], f"Expected matrix to be square"
        # check if shape is correct
        assert matrix.shape[-1] == self.n, f"Expected matrix to have shape (batch_size, {self.n}, {self.n}), got {matrix.shape}"
        assert matrix.shape[0] == self.batch_size, f"Expected matrix to have shape ({self.batch_size}, {self.n}, {self.n}), got {matrix.shape}"
        # check if matrix is symmetric
        assert torch.allclose(matrix, matrix.transpose(-1, -2)), f"Expected matrix to be symmetric"

        # check if matrix is positive definite
        if not torch.all(torch.linalg.eigvalsh(matrix) > 0):
            warnings.warn("Matrix is not positive definite, adding jitter to diagonal")
            matrix = matrix + self.jitter
            
        # set the matrix
        # use cholesky decomposition, so tril @ tril.transpose(-2, -1) will be the matrix

        matrix = torch.linalg.cholesky(matrix)

        new_trill_data = self._matrix_to_tril(matrix)

        assert new_trill_data.shape == self.tril.shape, f"Expected tril to have shape {self.tril.shape}, got {new_trill_data.shape}"
        
        self.tril.data = new_trill_data
        
        return self
    
    def get_unbroadcasted_scale_tril(self):
        """
        Get the scale tril without broadcasting
        """
        return self._tril_to_matrix(self.tril)

    def forward(self):
        tril = self._tril_to_matrix(self.tril)
        res = (tril @ tril.transpose(-2, -1)) + self.jitter
        return res
    
    def _matrix_to_tril(self, matrix):
        """
        Convert a matrix to a tril
        matrix : (batch_size, n, n)
        return : (batch_size, n * (n + 1) // 2)
        """
        tril = torch.tril(matrix)
        tril_i = torch.tril_indices(row=self.n, col=self.n, offset=0)
        tril = tril[:,tril_i[0], tril_i[1]]
        return tril
    
    def _tril_to_matrix(self, tril):
        """
        Convert a tril to a matrix
        NOTE: doesnt fill upper triangle
        tril : (batch_size, n * (n + 1) // 2)
        return : (batch_size, n, n)
        """
        b = tril.shape[0]
        n = self.n
        tril_matrix = torch.zeros(b, n, n, device=tril.device)
        tril_i = torch.tril_indices(row=n, col=n, offset=0, device=tril.device)
        tril_matrix[:, tril_i[0], tril_i[1]] = tril
        return tril_matrix