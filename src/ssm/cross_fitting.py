# src/ssm/cross_fitting.py

import torch
import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any


class CrossFittingManager:
    """
    Cross-fitting manager for time series data.

    Splits the time series into contiguous blocks and performs
    out-of-fold estimation per block to prevent temporal leakage and self-fitting.

    Implements Equations (17)-(20) in the paper.
    """

    def __init__(self, T: int, n_blocks: int = 5, min_block_size: int = 10):
        """
        Args:
            T: Time series length
            n_blocks: Number of blocks K
            min_block_size: Minimum block size for numerical stability
        """
        self.T = T
        self.n_blocks = min(n_blocks, T // min_block_size)
        self.min_block_size = min_block_size
        self.blocks = self._create_contiguous_blocks()
        
    def _create_contiguous_blocks(self) -> List[List[int]]:
        """
        Split time series into contiguous blocks {B_k}^K_{k=1}.

        Returns:
            List[List[int]]: Index list for each block
        """
        block_size = self.T // self.n_blocks
        remainder = self.T % self.n_blocks
        
        blocks = []
        start_idx = 0
        
        for k in range(self.n_blocks):
            # Distribute remainder across the first few blocks
            current_size = block_size + (1 if k < remainder else 0)
            end_idx = start_idx + current_size
            
            blocks.append(list(range(start_idx, min(end_idx, self.T))))
            start_idx = end_idx
            
        return blocks
    
    def get_out_of_fold_indices(self, block_k: int) -> List[int]:
        """
        Get out-of-fold indices I_{-k} for block k.

        Args:
            block_k: Block number (0-indexed)

        Returns:
            List[int]: I_{-k} = {0,...,T} \\ B_k
        """
        if not (0 <= block_k < self.n_blocks):
            raise ValueError(f"block_k must be in [0, {self.n_blocks}), got {block_k}")
            
        in_fold = set(self.blocks[block_k])
        return [t for t in range(self.T) if t not in in_fold]
    
    def get_block_indices(self, block_k: int) -> List[int]:
        """
        Get indices B_k for block k.

        Args:
            block_k: Block number (0-indexed)

        Returns:
            List[int]: B_k
        """
        if not (0 <= block_k < self.n_blocks):
            raise ValueError(f"block_k must be in [0, {self.n_blocks}), got {block_k}")
            
        return self.blocks[block_k].copy()
    
    def get_block_info(self) -> Dict[str, Any]:
        """
        Return block partition info.

        Returns:
            Dict: Block partition metadata
        """
        return {
            'T': self.T,
            'n_blocks': self.n_blocks, 
            'block_sizes': [len(block) for block in self.blocks],
            'blocks': self.blocks.copy()
        }


class TwoStageCrossFitter:
    """
    Cross-fitting executor for two-stage regression.

    The regression algorithm is injected externally;
    this class manages only the temporal split and out-of-fold strategy.
    """

    def __init__(self, cf_manager: CrossFittingManager):
        """
        Args:
            cf_manager: CrossFittingManager instance
        """
        self.cf_manager = cf_manager
        
    def cross_fit_stage1(
        self,
        X_input: torch.Tensor,
        Y_target: torch.Tensor,
        stage1_estimator: Callable,
        **estimator_kwargs
    ) -> List[torch.Tensor]:
        """
        Execute Stage-1 cross-fitting.

        For each block k, estimate V^{(-k)} on out-of-fold data.
        The estimation method is delegated to stage1_estimator.

        Args:
            X_input: Explanatory variables (T, d_X)
            Y_target: Target variables (T, d_Y)
            stage1_estimator: Stage-1 estimation function
                signature: V = estimator(X_oof, Y_oof, **kwargs)
            **estimator_kwargs: Additional arguments for estimator

        Returns:
            List[torch.Tensor]: [V^{(-0)}, V^{(-1)}, ..., V^{(-K-1)}]
        """
        if X_input.size(0) != self.cf_manager.T or Y_target.size(0) != self.cf_manager.T:
            raise ValueError(f"Input size mismatch: X={X_input.size(0)}, Y={Y_target.size(0)}, T={self.cf_manager.T}")
        
        V_list = []

        for k in range(self.cf_manager.n_blocks):
            oof_indices = self.cf_manager.get_out_of_fold_indices(k)

            if len(oof_indices) < self.cf_manager.min_block_size:
                raise ValueError(f"Out-of-fold size {len(oof_indices)} too small for block {k}")

            X_oof = X_input[oof_indices]
            Y_oof = Y_target[oof_indices]

            V_k = stage1_estimator(X_oof, Y_oof, **estimator_kwargs)
            V_list.append(V_k)
            
        return V_list
        
    def compute_out_of_fold_features(
        self,
        X_input: torch.Tensor,
        V_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute out-of-fold features H^{(cf)}.

        For each block k, compute H^{(cf)}_{B_k} = V^{(-k)} X_{B_k}
        and reassemble in original temporal order.

        Args:
            X_input: Input features (T, d_X)
            V_list: List of V^{(-k)} from cross_fit_stage1

        Returns:
            torch.Tensor: H^{(cf)} (T, d_V) - cross-fitted features in original order
        """
        if len(V_list) != self.cf_manager.n_blocks:
            raise ValueError(f"V_list length {len(V_list)} != n_blocks {self.cf_manager.n_blocks}")
        
        if X_input.size(0) != self.cf_manager.T:
            raise ValueError(f"Input size mismatch: X={X_input.size(0)}, T={self.cf_manager.T}")
        
        device = X_input.device
        d_V = V_list[0].size(0)
        H_cf = torch.zeros(self.cf_manager.T, d_V, device=device, dtype=X_input.dtype)

        for k in range(self.cf_manager.n_blocks):
            block_indices = self.cf_manager.get_block_indices(k)
            X_block = X_input[block_indices]
            H_block = (V_list[k] @ X_block.T).T

            H_cf[block_indices] = H_block
            
        return H_cf
        
    def cross_fit_stage2(
        self,
        H_cf: torch.Tensor,
        Y_target: torch.Tensor,
        stage2_estimator: Callable,
        detach_features: bool = True,
        **estimator_kwargs
    ) -> torch.Tensor:
        """
        Stage-2 estimation (readout).

        Perform final estimation using cross-fitted features H_cf.
        The estimation method is delegated to stage2_estimator.

        Args:
            H_cf: Out-of-fold features (T, d_V)
            Y_target: Target (T, d_Y)
            stage2_estimator: Stage-2 estimation function
                signature: U = estimator(H, Y, **kwargs)
            detach_features: Whether to detach gradients (prevents self-fitting)
            **estimator_kwargs: Additional arguments for estimator

        Returns:
            torch.Tensor: Stage-2 estimation result U
        """
        if H_cf.size(0) != Y_target.size(0):
            raise ValueError(f"Feature-target size mismatch: H_cf={H_cf.size(0)}, Y={Y_target.size(0)}")
        
        if detach_features:
            H_cf = H_cf.detach()
            
        return stage2_estimator(H_cf, Y_target, **estimator_kwargs)

    def cross_fit_stage2_matrix(
        self,
        H_cf: torch.Tensor,
        M_target: torch.Tensor,
        stage2_estimator: Callable,
        detach_features: bool = True,
        **estimator_kwargs
    ) -> List[torch.Tensor]:
        """
        Stage-2 estimation (multivariate readout matrix).

        Perform out-of-fold estimation per block and return U_B list.

        Args:
            H_cf: Out-of-fold features (T, d_B)
            M_target: Multivariate target (T, m)
            stage2_estimator: Stage-2 estimation function
                signature: U_B = estimator(H, M, **kwargs) -> (d_B, m)
            detach_features: Whether to detach gradients
            **estimator_kwargs: Additional arguments for estimator

        Returns:
            List[torch.Tensor]: U_B list per block [(d_B, m), ...]
        """
        if H_cf.size(0) != M_target.size(0):
            raise ValueError(f"Feature-target size mismatch: H_cf={H_cf.size(0)}, M={M_target.size(0)}")

        U_B_list = []

        for k in range(self.cf_manager.n_blocks):
            oof_indices = self.cf_manager.get_out_of_fold_indices(k)

            H_oof = H_cf[oof_indices]
            M_oof = M_target[oof_indices]

            if detach_features:
                H_oof = H_oof.detach()

            U_B_k = stage2_estimator(H_oof, M_oof, **estimator_kwargs)
            U_B_list.append(U_B_k)

        return U_B_list


class CrossFittingError(Exception):
    """Error during cross-fitting."""
    pass


def validate_cross_fitting_setup(
    T: int,
    n_blocks: int,
    min_samples_per_fold: int = 10
) -> Dict[str, Any]:
    """
    Validate cross-fitting configuration.

    Args:
        T: Time series length
        n_blocks: Number of blocks
        min_samples_per_fold: Minimum samples per fold

    Returns:
        Dict: Validation results and metadata

    Raises:
        CrossFittingError: If configuration is invalid
    """
    if T < n_blocks * min_samples_per_fold:
        raise CrossFittingError(
            f"Time series length {T} too short. n_blocks={n_blocks}, "
            f"min_samples={min_samples_per_fold} requires at least {n_blocks * min_samples_per_fold}"
        )

    avg_block_size = T // n_blocks
    min_oof_size = T - avg_block_size - (T % n_blocks)

    if min_oof_size < min_samples_per_fold:
        raise CrossFittingError(
            f"Out-of-fold size {min_oof_size} too small. "
            f"Requires at least {min_samples_per_fold}"
        )
    
    return {
        'valid': True,
        'T': T,
        'n_blocks': n_blocks,
        'avg_block_size': avg_block_size,
        'min_oof_size': min_oof_size,
        'max_oof_size': T - (T // n_blocks)
    }