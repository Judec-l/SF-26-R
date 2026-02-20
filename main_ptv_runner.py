"""
main_ptv_runner.py

Main script to run PTV analysis on your image directory.
This is the file you run to process your data.

Usage:
    python main_ptv_runner.py

Configure your input/output folders in the CONFIG section below.
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ptv_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import your PTV modules
try:
    from beadPosition import beadPosition
    from trackBeads import trackBeads
    from LinkPairsF2L import LinkPairsF2L
    from crack_Link2FrameID import crack_Link2FrameId
    from MLS_interpolation import MLS_interpolation
except ImportError as e:
    logger.error(f"Failed to import PTV modules: {e}")
    logger.error("Make sure all .py files are in the same directory as this script!")
    sys.exit(1)

# ============================================================================
# CONFIGURATION - EDIT THIS SECTION
# ============================================================================

CONFIG = {
    # Input/Output Directories
    'input_dir': r'C:\ExtractData\Hydrogel',  # Your images are here
    'output_dir': r'C:\ExtractData\Hydrogel\output',

    # Image Settings
    'image_pattern': '*.png',  # File pattern (*.tif, *.png, *.jpg, etc.)
    'image_sort': 'name',  # How to sort: 'name' or 'time'

    # Detection Parameters
    'threshold': 0.5,  # Intensity threshold (0-1) for bead detection
    'min_bead_area': 3,  # Minimum pixel area for valid bead

    # Tracking Parameters
    'max_displacement': 10.0,  # Maximum expected particle movement (pixels)
    'experiment_name': 'trial_run',  # Name for cache files

    # Processing Options
    'frame_start': 0,  # Start from frame N (0 = first frame)
    'frame_end': None,  # End at frame N (None = all frames)
    'frame_step': 10,  # Process every Nth frame

    # Output Options
    'save_visualizations': True,  # Save images with detected beads
    'save_trajectories': True,  # Save trajectory data
    'save_velocity_field': True,  # Save interpolated velocity field

    # Advanced Options
    'use_cache': True,  # Use cached preprocessing/results
    'min_trajectory_length': 3,  # Minimum frames for valid trajectory
}


# ============================================================================
# END CONFIGURATION
# ============================================================================


def setup_directories(config):
    """Create necessary directory structure."""
    logger.info("Setting up directories...")

    # Create output directories
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    subdirs = {
        'detections': output_dir / 'detections',
        'matches': output_dir / 'matches',
        'trajectories': output_dir / 'trajectories',
        'velocity_fields': output_dir / 'velocity_fields',
        'data': output_dir / 'data'
    }

    for name, path in subdirs.items():
        path.mkdir(parents=True, exist_ok=True)

    # Create PTV working directories
    Path('../PTLibrary2D/aux_matchpair').mkdir(parents=True, exist_ok=True)
    Path('../PTLibrary2D/res_matchpair').mkdir(parents=True, exist_ok=True)

    logger.info(f"✓ Output directory: {output_dir}")
    return subdirs


def get_image_files(config):
    """Get sorted list of image files."""
    input_dir = Path(config['input_dir'])

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all images matching pattern
    pattern = config['image_pattern']
    image_files = sorted(input_dir.glob(pattern))

    if len(image_files) == 0:
        raise FileNotFoundError(
            f"No images found in {input_dir} matching pattern '{pattern}'"
        )

    # Sort by modification time if requested
    if config['image_sort'] == 'time':
        image_files = sorted(image_files, key=lambda x: x.stat().st_mtime)

    # Apply frame selection
    start = config['frame_start']
    end = config['frame_end']
    step = config['frame_step']

    image_files = image_files[start:end:step]

    logger.info(f"✓ Found {len(image_files)} images to process")
    logger.info(f"  First: {image_files[0].name}")
    logger.info(f"  Last: {image_files[-1].name}")

    return image_files


def detect_beads_in_images(image_files, config, output_dirs):
    """Detect bead positions in all images."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Detecting beads in images")
    logger.info("=" * 60)

    all_positions = []
    threshold = config['threshold']

    for i, img_path in enumerate(image_files):
        logger.info(f"Processing frame {i + 1}/{len(image_files)}: {img_path.name}")

        try:
            # Detect beads
            detector = beadPosition(str(img_path), threshold)
            detector.calcBeadCenter()

            positions = detector.beadPositionLocal
            all_positions.append(positions)

            logger.info(f"  Detected {len(positions)} beads")

            # Save visualization if requested
            if config['save_visualizations']:
                output_path = output_dirs['detections'] / f"frame_{i:04d}_detected.png"
                detector.visualize(str(output_path))

        except Exception as e:
            logger.error(f"  Failed to process {img_path.name}: {e}")
            all_positions.append(np.empty((0, 2)))

    logger.info(f"✓ Detection complete: {len(all_positions)} frames processed")
    return all_positions


def track_frame_pairs(all_positions, config, output_dirs):
    """Track particles between consecutive frames."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Tracking particles between frames")
    logger.info("=" * 60)

    trackers = []
    n_frames = len(all_positions)

    for i in range(n_frames - 1):
        logger.info(f"Matching frame {i} -> {i + 1}")

        try:
            tracker = trackBeads(
                positionArray1=all_positions[i],
                positionArray2=all_positions[i + 1],
                dataName=f"{config['experiment_name']}_f{i}_{i + 1}",
                maxBeadDisplacement=config['max_displacement'],
                beadParam={
                    'threshold': config['threshold'],
                    'matchpair_option': {
                        'AuxFileSave': 'YES' if config['use_cache'] else 'NO',
                        'ResFileSave': 'YES' if config['use_cache'] else 'NO',
                    }
                }
            )

            trackers.append(tracker)

            # Get match statistics
            match_info = tracker.matchArray[0]['matchInfo']
            logger.info(f"  Matched: {len(match_info['I1'])} particles ({match_info['matchRatio']:.1%})")

            # Save match visualization if requested
            if config['save_visualizations']:
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(12, 5))

                # Plot frame i
                ax1 = fig.add_subplot(121)
                pos1 = all_positions[i]
                ax1.scatter(pos1[:, 0], pos1[:, 1], c='blue', s=20, alpha=0.6)
                ax1.set_title(f'Frame {i}')
                ax1.set_aspect('equal')

                # Plot frame i+1 with matches
                ax2 = fig.add_subplot(122)
                pos2 = all_positions[i + 1]
                ax2.scatter(pos2[:, 0], pos2[:, 1], c='red', s=20, alpha=0.6)

                # Draw match lines
                I1, I2 = match_info['I1'], match_info['I2']
                for idx1, idx2 in zip(I1, I2):
                    ax2.plot([pos1[idx1, 0], pos2[idx2, 0]],
                             [pos1[idx1, 1], pos2[idx2, 1]],
                             'k-', alpha=0.3, linewidth=0.5)

                ax2.set_title(f'Frame {i + 1} (with matches)')
                ax2.set_aspect('equal')

                output_path = output_dirs['matches'] / f"match_{i:04d}_{i + 1:04d}.png"
                plt.tight_layout()
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()

        except Exception as e:
            logger.error(f"  Failed to match frames {i}-{i + 1}: {e}")
            continue

    logger.info(f"✓ Tracking complete: {len(trackers)} frame pairs tracked")
    return trackers


def link_trajectories(trackers, config, output_dirs):
    """Link frame-pair matches into continuous trajectories."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Linking trajectories")
    logger.info("=" * 60)

    # Extract pair matches
    pair_matches = []
    for tracker in trackers:
        match_info = tracker.matchArray[0]['matchInfo']
        matches = np.column_stack([match_info['I1'], match_info['I2']])
        pair_matches.append(matches)

    # Link trajectories
    trajectories, broken_chains = LinkPairsF2L(pair_matches)

    logger.info(f"  Total trajectories: {len(trajectories)}")
    logger.info(f"  Broken chains: {len(broken_chains)}")

    # Filter by minimum length
    min_length = config['min_trajectory_length']
    traj_lengths = [len(t) for t in trajectories]
    valid_trajectories = [t for t, l in zip(trajectories, traj_lengths) if l >= min_length]

    logger.info(f"  Valid trajectories (length >= {min_length}): {len(valid_trajectories)}")

    # Save trajectory data
    if config['save_trajectories'] and valid_trajectories:
        output_path = output_dirs['data'] / 'trajectories.npz'
        np.savez(
            output_path,
            trajectories=np.array(valid_trajectories, dtype=object),
            lengths=np.array([len(t) for t in valid_trajectories]),
            broken_chains=np.array(broken_chains)
        )
        logger.info(f"  Saved trajectory data to {output_path}")

    logger.info(f"✓ Trajectory linking complete")
    return valid_trajectories, broken_chains


def compute_velocity_statistics(trackers, all_positions, config, output_dirs):
    """Compute and save velocity statistics."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Computing velocity statistics")
    logger.info("=" * 60)

    # Collect all displacements
    all_displacements = []
    all_positions_matched = []

    for i, tracker in enumerate(trackers):
        disp = tracker.matchArray[0]['displacement'][:, :2]  # x, y only
        match_info = tracker.matchArray[0]['matchInfo']
        pos = all_positions[i][match_info['I1']][:, :2]

        if len(disp) > 0:
            all_displacements.append(disp)
            all_positions_matched.append(pos)

    if not all_displacements:
        logger.warning("  No displacements to analyze!")
        return

    all_disp = np.vstack(all_displacements)
    all_pos = np.vstack(all_positions_matched)

    # Compute statistics
    speeds = np.linalg.norm(all_disp, axis=1)
    mean_velocity = np.mean(all_disp, axis=0)
    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    max_speed = np.max(speeds)

    logger.info(f"  Mean velocity: [{mean_velocity[0]:.3f}, {mean_velocity[1]:.3f}] pixels/frame")
    logger.info(f"  Mean speed: {mean_speed:.3f} pixels/frame")
    logger.info(f"  Std speed: {std_speed:.3f} pixels/frame")
    logger.info(f"  Max speed: {max_speed:.3f} pixels/frame")

    # Save velocity data
    output_path = output_dirs['data'] / 'velocity_data.npz'
    np.savez(
        output_path,
        displacements=all_disp,
        positions=all_pos,
        speeds=speeds,
        mean_velocity=mean_velocity,
        mean_speed=mean_speed,
        std_speed=std_speed,
        max_speed=max_speed
    )
    logger.info(f"  Saved velocity data to {output_path}")

    # Interpolate velocity field if requested
    if config['save_velocity_field'] and len(all_pos) >= 20:
        logger.info("  Interpolating velocity field...")

        # Create regular grid
        x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
        y_min, y_max = all_pos[:, 1].min(), all_pos[:, 1].max()

        x_grid = np.linspace(x_min, x_max, 50)
        y_grid = np.linspace(y_min, y_max, 50)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Interpolate
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    u, _, _ = MLS_interpolation(
                        np.array([X[i, j], Y[i, j]]),
                        all_pos,
                        all_disp[:, 0],
                        order=2
                    )
                    v, _, _ = MLS_interpolation(
                        np.array([X[i, j], Y[i, j]]),
                        all_pos,
                        all_disp[:, 1],
                        order=2
                    )
                    U[i, j] = u
                    V[i, j] = v
                except:
                    U[i, j] = np.nan
                    V[i, j] = np.nan

        # Save velocity field
        output_path = output_dirs['data'] / 'velocity_field.npz'
        np.savez(output_path, X=X, Y=Y, U=U, V=V)
        logger.info(f"  Saved velocity field to {output_path}")

        # Plot velocity field
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))

        # Quiver plot
        skip = 2  # Plot every 2nd point for clarity
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                  U[::skip, ::skip], V[::skip, ::skip],
                  alpha=0.7)

        # Scatter plot of measurement points
        ax.scatter(all_pos[:, 0], all_pos[:, 1], c='red', s=10, alpha=0.3)

        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title('Interpolated Velocity Field')
        ax.set_aspect('equal')

        output_path = output_dirs['velocity_fields'] / 'velocity_field.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved velocity field plot to {output_path}")

    logger.info(f"✓ Velocity analysis complete")


def save_summary_report(config, output_dirs, all_positions, trackers, trajectories):
    """Save a summary report of the analysis."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Generating summary report")
    logger.info("=" * 60)

    report_path = Path(config['output_dir']) / 'analysis_report.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PTV ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Experiment: {config['experiment_name']}\n")
        f.write(f"Input directory: {config['input_dir']}\n")
        f.write(f"Output directory: {config['output_dir']}\n\n")

        f.write("DETECTION RESULTS:\n")
        f.write(f"  Total frames processed: {len(all_positions)}\n")
        beads_per_frame = [len(pos) for pos in all_positions]
        f.write(f"  Mean beads per frame: {np.mean(beads_per_frame):.1f}\n")
        f.write(f"  Min/Max beads: {np.min(beads_per_frame)} / {np.max(beads_per_frame)}\n\n")

        f.write("TRACKING RESULTS:\n")
        f.write(f"  Frame pairs tracked: {len(trackers)}\n")
        match_ratios = [t.matchArray[0]['matchInfo']['matchRatio'] for t in trackers]
        f.write(f"  Mean match ratio: {np.mean(match_ratios):.1%}\n")
        f.write(f"  Min/Max match ratio: {np.min(match_ratios):.1%} / {np.max(match_ratios):.1%}\n\n")

        f.write("TRAJECTORY RESULTS:\n")
        f.write(f"  Total trajectories: {len(trajectories)}\n")
        traj_lengths = [len(t) for t in trajectories]
        f.write(f"  Mean trajectory length: {np.mean(traj_lengths):.1f} frames\n")
        f.write(f"  Min/Max length: {np.min(traj_lengths)} / {np.max(traj_lengths)} frames\n\n")

        f.write("PARAMETERS USED:\n")
        f.write(f"  Detection threshold: {config['threshold']}\n")
        f.write(f"  Max displacement: {config['max_displacement']} pixels\n")
        f.write(f"  Min trajectory length: {config['min_trajectory_length']} frames\n")

    logger.info(f"✓ Report saved to {report_path}")


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("  PARTICLE TRACKING VELOCIMETRY (PTV) ANALYSIS")
    print("=" * 70 + "\n")

    try:
        # Setup
        output_dirs = setup_directories(CONFIG)
        image_files = get_image_files(CONFIG)

        # Step 1: Detect beads
        all_positions = detect_beads_in_images(image_files, CONFIG, output_dirs)

        # Step 2: Track frame pairs
        trackers = track_frame_pairs(all_positions, CONFIG, output_dirs)

        # Step 3: Link trajectories
        trajectories, broken = link_trajectories(trackers, CONFIG, output_dirs)

        # Step 4: Compute velocities
        compute_velocity_statistics(trackers, all_positions, CONFIG, output_dirs)

        # Step 5: Generate report
        save_summary_report(CONFIG, output_dirs, all_positions, trackers, trajectories)

        print("\n" + "=" * 70)
        print("  ✓ ANALYSIS COMPLETE!")
        print("=" * 70)
        print(f"\nResults saved to: {CONFIG['output_dir']}")
        print("\nGenerated files:")
        print(f"  - Detections: {output_dirs['detections']}")
        print(f"  - Matches: {output_dirs['matches']}")
        print(f"  - Data: {output_dirs['data']}")
        print(f"  - Report: {CONFIG['output_dir']}/analysis_report.txt")

        return True

    except Exception as e:
        logger.error(f"\n✗ ANALYSIS FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)