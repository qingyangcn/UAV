"""
Multi-Objective PSO utilities: dominance, archive management, crowding distance.
"""
import numpy as np
from typing import List, Tuple, Dict


def dominates(f_a: np.ndarray, f_b: np.ndarray) -> bool:
    """
    Check if fitness vector f_a dominates f_b (for maximization).
    f_a dominates f_b if f_a >= f_b in all objectives and f_a > f_b in at least one.
    
    Args:
        f_a: Fitness vector (maximize all objectives)
        f_b: Fitness vector (maximize all objectives)
    
    Returns:
        True if f_a dominates f_b
    """
    return np.all(f_a >= f_b) and np.any(f_a > f_b)


def pareto_filter(solutions: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Filter solutions to keep only non-dominated ones.
    
    Args:
        solutions: List of (position, fitness) tuples
    
    Returns:
        List of non-dominated (position, fitness) tuples
    """
    if not solutions:
        return []
    
    front = []
    for i, (x_i, f_i) in enumerate(solutions):
        dominated = False
        for j, (x_j, f_j) in enumerate(solutions):
            if i == j:
                continue
            if dominates(f_j, f_i):
                dominated = True
                break
        if not dominated:
            front.append((x_i.copy(), f_i.copy()))
    
    return front


def crowding_distance(archive: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Calculate crowding distance for each solution in archive.
    Higher crowding distance means more isolated (better diversity).
    
    Args:
        archive: List of (position, fitness) tuples
    
    Returns:
        Array of crowding distances, same length as archive
    """
    n = len(archive)
    if n <= 2:
        return np.full(n, np.inf)
    
    # Extract fitness values
    fitness_matrix = np.array([f for _, f in archive])  # shape (n, num_objectives)
    num_objectives = fitness_matrix.shape[1]
    
    # Initialize distances
    distances = np.zeros(n)
    
    # For each objective
    for m in range(num_objectives):
        # Sort by objective m
        obj_values = fitness_matrix[:, m]
        sorted_indices = np.argsort(obj_values)
        
        # Boundary solutions get infinite distance
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf
        
        # Calculate distance for middle solutions
        obj_range = obj_values[sorted_indices[-1]] - obj_values[sorted_indices[0]]
        if obj_range < 1e-10:
            continue
        
        for i in range(1, n - 1):
            idx = sorted_indices[i]
            idx_prev = sorted_indices[i - 1]
            idx_next = sorted_indices[i + 1]
            
            distances[idx] += (obj_values[idx_next] - obj_values[idx_prev]) / obj_range
    
    return distances


def truncate_archive(archive: List[Tuple[np.ndarray, np.ndarray]], 
                     max_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Truncate archive using crowding distance to maintain diversity.
    
    Args:
        archive: List of (position, fitness) tuples
        max_size: Maximum archive size
    
    Returns:
        Truncated archive
    """
    if len(archive) <= max_size:
        return archive
    
    # Calculate crowding distances
    distances = crowding_distance(archive)
    
    # Sort by crowding distance (descending) and keep top max_size
    sorted_indices = np.argsort(distances)[::-1]
    selected_indices = sorted_indices[:max_size]
    
    return [archive[i] for i in selected_indices]


def select_leader(archive: List[Tuple[np.ndarray, np.ndarray]], 
                  weights: np.ndarray,
                  rng: np.random.RandomState) -> np.ndarray:
    """
    Select a leader from archive using weighted sum or diversity-based selection.
    
    Args:
        archive: List of (position, fitness) tuples
        weights: Objective weights for scalarization
        rng: Random number generator
    
    Returns:
        Position vector of selected leader
    """
    if not archive:
        raise ValueError("Archive is empty")
    
    if len(archive) == 1:
        return archive[0][0].copy()
    
    # Use crowding distance for diversity-based selection
    distances = crowding_distance(archive)
    
    # Select from top diverse solutions with some randomness
    # Replace infinite distances with a large finite value for normalization
    has_finite = np.any(~np.isinf(distances))
    if has_finite:
        max_finite = np.max(distances[~np.isinf(distances)])
        finite_distances = np.where(np.isinf(distances), max_finite * 2.0, distances)
    else:
        finite_distances = np.ones_like(distances)
    
    # Normalize for probability
    prob = finite_distances.astype(np.float64)
    prob_sum = prob.sum()
    
    if prob_sum > 1e-10 and not np.any(np.isnan(prob)):
        prob = prob / prob_sum
        
        # Sample based on diversity
        idx = rng.choice(len(archive), p=prob)
        return archive[idx][0].copy()
    
    # Fallback to random selection
    idx = rng.randint(len(archive))
    return archive[idx][0].copy()


def select_best_solution(archive: List[Tuple[np.ndarray, np.ndarray]], 
                        weights: np.ndarray,
                        normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select the best solution from archive using weighted sum.
    
    Args:
        archive: List of (position, fitness) tuples
        weights: Objective weights for scalarization
        normalize: Whether to normalize fitness before scalarization
    
    Returns:
        (position, fitness) of best solution
    """
    if not archive:
        raise ValueError("Archive is empty")
    
    if len(archive) == 1:
        return archive[0]
    
    # Extract fitness vectors
    fitness_matrix = np.array([f for _, f in archive])
    
    # Normalize if requested
    if normalize:
        f_min = fitness_matrix.min(axis=0)
        f_max = fitness_matrix.max(axis=0)
        f_range = f_max - f_min
        
        # Avoid division by zero
        f_range = np.where(f_range < 1e-10, 1.0, f_range)
        fitness_norm = (fitness_matrix - f_min) / f_range
    else:
        fitness_norm = fitness_matrix
    
    # Calculate weighted sum
    scores = np.dot(fitness_norm, weights)
    
    # Select best
    best_idx = np.argmax(scores)
    return archive[best_idx]


def grid_based_selection(archive: List[Tuple[np.ndarray, np.ndarray]], 
                        max_size: int,
                        num_divisions: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Alternative to crowding distance: grid-based selection for archive truncation.
    
    Args:
        archive: List of (position, fitness) tuples
        max_size: Maximum archive size
        num_divisions: Number of divisions per objective
    
    Returns:
        Truncated archive
    """
    if len(archive) <= max_size:
        return archive
    
    # Extract fitness
    fitness_matrix = np.array([f for _, f in archive])
    num_objectives = fitness_matrix.shape[1]
    
    # Normalize to [0, 1]
    f_min = fitness_matrix.min(axis=0)
    f_max = fitness_matrix.max(axis=0)
    f_range = f_max - f_min
    f_range = np.where(f_range < 1e-10, 1.0, f_range)
    fitness_norm = (fitness_matrix - f_min) / f_range
    
    # Assign to grid cells
    grid_indices = (fitness_norm * num_divisions).astype(int)
    grid_indices = np.clip(grid_indices, 0, num_divisions - 1)
    
    # Convert to tuple for hashing
    grid_cells = [tuple(idx) for idx in grid_indices]
    
    # Count solutions per cell
    cell_counts = {}
    solution_cells = {}
    for i, cell in enumerate(grid_cells):
        if cell not in cell_counts:
            cell_counts[cell] = 0
        cell_counts[cell] += 1
        solution_cells[i] = cell
    
    # Select solutions, preferring less crowded cells
    selected = []
    cell_selected = {cell: 0 for cell in cell_counts}
    
    # Sort by cell density (ascending)
    solution_indices = list(range(len(archive)))
    solution_indices.sort(key=lambda i: (cell_counts[solution_cells[i]], i))
    
    for idx in solution_indices:
        if len(selected) >= max_size:
            break
        selected.append(archive[idx])
    
    return selected
