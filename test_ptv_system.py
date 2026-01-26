"""
test_ptv_system.py
Simple test to verify the refactored PTV system works
"""
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def setup_directories():
    """Create required directory structure."""
    os.makedirs("../PTLibrary2D/aux_matchpair", exist_ok=True)
    os.makedirs("../PTLibrary2D/res_matchpair", exist_ok=True)
    logger.info("Directory structure created")


def test_basic_matching():
    """Test 1: Basic particle matching with synthetic data."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Particle Matching")
    print("=" * 60)

    from trackBeads import trackBeads

    # Generate synthetic particle positions
    np.random.seed(42)
    n_particles = 50

    # Frame 1
    x1 = np.random.rand(n_particles) * 100
    y1 = np.random.rand(n_particles) * 100
    positions1 = np.column_stack([x1, y1])

    # Frame 2: add small displacement with noise
    mean_dx, mean_dy = 0.5, 0.3
    x2 = x1 + mean_dx + np.random.randn(n_particles) * 0.1
    y2 = y1 + mean_dy + np.random.randn(n_particles) * 0.1
    positions2 = np.column_stack([x2, y2])

    # Create tracker
    tracker = trackBeads(
        positionArray1=positions1,
        positionArray2=positions2,
        dataName="test_basic",
        maxBeadDisplacement=2.0
    )

    # Get results
    match_data = tracker.matchArray[0]
    matchInfo = match_data['matchInfo']

    print(f"\nResults:")
    print(f"  Total particles frame 1: {n_particles}")
    print(f"  Total particles frame 2: {n_particles}")
    print(f"  Matched particles: {len(matchInfo['I1'])}")
    print(f"  Match ratio: {matchInfo['matchRatio']:.1%}")
    print(f"  Unmatched frame 1: {len(matchInfo['I1u'])}")
    print(f"  Unmatched frame 2: {len(matchInfo['I2u'])}")

    # Compute displacement statistics
    displacement = match_data['displacement']
    if len(displacement) > 0:
        mean_disp_x = np.mean(displacement[:, 0])
        mean_disp_y = np.mean(displacement[:, 1])
        print(f"\nDisplacement statistics:")
        print(f"  Mean dx: {mean_disp_x:.3f} (expected: {mean_dx})")
        print(f"  Mean dy: {mean_disp_y:.3f} (expected: {mean_dy})")
        print(f"  Error dx: {abs(mean_disp_x - mean_dx):.3f}")
        print(f"  Error dy: {abs(mean_disp_y - mean_dy):.3f}")

    assert len(matchInfo['I1']) > 0, "No matches found!"
    assert matchInfo['matchRatio'] > 0.5, "Match ratio too low!"

    print("\n✓ Test 1 PASSED")
    return tracker


def test_trajectory_linking():
    """Test 2: Trajectory linking across frames."""
    print("\n" + "=" * 60)
    print("TEST 2: Trajectory Linking")
    print("=" * 60)

    from LinkPairsF2L import LinkPairsF2L
    from crack_Link2FrameID import crack_Link2FrameId

    # Create synthetic multi-frame matching data
    n_particles = 30
    n_frames = 4

    # Create perfect tracking (each particle maintains same index)
    pair_matches = []
    for i in range(n_frames - 1):
        matches = np.column_stack([
            np.arange(n_particles),
            np.arange(n_particles)
        ])
        pair_matches.append(matches)

    # Link trajectories
    trajectories, broken = LinkPairsF2L(pair_matches)

    print(f"\nResults:")
    print(f"  Number of trajectories: {len(trajectories)}")
    print(f"  Broken chains: {len(broken)}")
    print(f"  Expected trajectories: {n_particles}")

    # Check trajectory lengths
    traj_lengths = [len(t) for t in trajectories]
    print(f"  Trajectory length range: {min(traj_lengths)} - {max(traj_lengths)}")
    print(f"  Expected length: {n_frames}")

    # Filter by length
    link_lengths = [len(t) for t in trajectories]
    first_ids, last_ids = crack_Link2FrameId(trajectories, link_lengths, n_frames)

    print(f"\nFiltered trajectories (length >= {n_frames}):")
    print(f"  Count: {len(first_ids)}")

    assert len(trajectories) == n_particles, "Wrong number of trajectories!"
    assert len(broken) == 0, "Unexpected broken chains!"
    assert all(l == n_frames for l in traj_lengths), "Wrong trajectory lengths!"

    print("\n✓ Test 2 PASSED")
    return trajectories


def test_mls_interpolation():
    """Test 3: MLS interpolation."""
    print("\n" + "=" * 60)
    print("TEST 3: MLS Interpolation")
    print("=" * 60)

    from MLS_interpolation import MLS_interpolation

    # Create synthetic data with known linear field
    np.random.seed(42)
    n_points = 25

    # Scattered measurement points
    xi = np.random.rand(n_points, 2) * 10

    # Known linear field: u = 2*x + 3*y
    ui = 2.0 * xi[:, 0] + 3.0 * xi[:, 1]

    # Test point
    x_test = np.array([5.0, 5.0])
    expected_u = 2.0 * 5.0 + 3.0 * 5.0  # = 25.0

    # Interpolate
    u, grad_u, laplacian_u = MLS_interpolation(x_test, xi, ui, order=2)

    print(f"\nResults:")
    print(f"  Query point: {x_test}")
    print(f"  Interpolated value: {u:.3f}")
    print(f"  Expected value: {expected_u:.3f}")
    print(f"  Error: {abs(u - expected_u):.3f}")
    print(f"  Gradient: {grad_u}")
    print(f"  Expected gradient: [2.0, 3.0]")
    print(f"  Laplacian: {laplacian_u:.3f}")

    # For linear field, gradient should be close to [2, 3]
    grad_error = np.linalg.norm(grad_u - np.array([2.0, 3.0]))
    print(f"  Gradient error: {grad_error:.3f}")

    assert abs(u - expected_u) < 0.5, "Interpolation error too large!"
    assert grad_error < 0.5, "Gradient error too large!"

    print("\n✓ Test 3 PASSED")
    return u, grad_u


def test_preprocessing():
    """Test 4: Preprocessing and nearest neighbor search."""
    print("\n" + "=" * 60)
    print("TEST 4: Preprocessing")
    print("=" * 60)

    from match_pair_preprocess import match_pair_preprocess

    # Create synthetic data
    np.random.seed(42)
    n1 = 40
    n2 = 40

    x1 = np.random.rand(n1) * 100
    y1 = np.random.rand(n1) * 100
    r1 = np.random.rand(n1) * 5

    x2 = np.random.rand(n2) * 100
    y2 = np.random.rand(n2) * 100
    r2 = np.random.rand(n2) * 5

    # Run preprocessing
    result = match_pair_preprocess(x1, y1, r1, x2, y2, r2)

    dist_min1a, dist_min1b, dist_min2a, dist_min2b = result[:4]
    index1a, index1b, index2a, index2b = result[4:]

    print(f"\nResults:")
    print(f"  Frame 1 particles: {n1}")
    print(f"  Frame 2 particles: {n2}")
    print(f"  dist_min1a shape: {dist_min1a.shape}")
    print(f"  dist_min1b shape: {dist_min1b.shape}")
    print(f"  index1a shape: {index1a.shape}")
    print(f"  index1b shape: {index1b.shape}")

    # Check that we have 3 nearest neighbors for each particle
    assert dist_min1a.shape == (n1, 3), "Wrong shape for dist_min1a!"
    assert dist_min1b.shape == (n1, 3), "Wrong shape for dist_min1b!"
    assert index1a.shape == (n1, 3), "Wrong shape for index1a!"
    assert index1b.shape == (n1, 3), "Wrong shape for index1b!"

    # Check that distances are non-negative
    assert np.all(dist_min1a >= 0), "Negative distances found!"
    assert np.all(dist_min1b >= 0), "Negative distances found!"

    print(f"\nSample nearest neighbor distances (frame 1 -> frame 2):")
    print(f"  Particle 0: {dist_min1b[0]}")
    print(f"  Particle 1: {dist_min1b[1]}")

    print("\n✓ Test 4 PASSED")
    return result


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  REFACTORED PTV SYSTEM - COMPREHENSIVE TESTS")
    print("=" * 70)

    # Setup
    setup_directories()

    try:
        # Run tests
        tracker = test_basic_matching()
        trajectories = test_trajectory_linking()
        u, grad_u = test_mls_interpolation()
        preprocess_result = test_preprocessing()

        print("\n" + "=" * 70)
        print("  ALL TESTS PASSED! ✓")
        print("=" * 70)
        print("\nThe refactored PTV system is working correctly!")
        print("\nYou can now use:")
        print("  - trackBeads for particle tracking")
        print("  - LinkPairsF2L for trajectory linking")
        print("  - MLS_interpolation for velocity field interpolation")
        print("  - match_pair_preprocess for preprocessing")

        return True

    except Exception as e:
        print("\n" + "=" * 70)
        print("  TEST FAILED! ✗")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)