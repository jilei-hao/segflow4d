"""
Batch validation CLI tool for comparing multiple segmentation images.

This tool compares segmentation images from a reference directory against
target directory based on filename patterns, computing Dice, MSD, and HD95 metrics.
"""

from __future__ import annotations
import argparse
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import asdict

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from utility.validation.segmentation_validation import (
    load_segmentation,
    evaluate_segmentation,
    ValidationResult
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_matching_files(
    directory: Path,
    pattern: str
) -> List[Path]:
    """
    Find files matching a glob pattern in the given directory.
    
    Args:
        directory: Directory to search in
        pattern: Glob pattern (e.g., 'seg3d_bav_tp*.nii.gz')
        
    Returns:
        Sorted list of matching file paths
    """
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    matches = sorted(directory.glob(pattern))
    
    if not matches:
        raise ValueError(f"No files found matching pattern '{pattern}' in {directory}")
    
    return matches


def extract_identifier(filepath: Path, pattern: str) -> str:
    """
    Extract a comparable identifier from a filename based on pattern.
    
    For pattern like 'seg3d_bav_tp*.nii.gz', this extracts the part
    that replaces the wildcard (*).
    
    Args:
        filepath: Path to the file
        pattern: Pattern with wildcard (*)
        
    Returns:
        Extracted identifier string
    """
    filename = filepath.name
    
    # Split pattern at wildcard
    parts = pattern.split('*')
    if len(parts) != 2:
        raise ValueError(f"Pattern must contain exactly one wildcard (*): {pattern}")
    
    prefix, suffix = parts
    
    # Check if filename matches pattern
    if not filename.startswith(prefix) or not filename.endswith(suffix):
        raise ValueError(
            f"File '{filename}' does not match pattern '{pattern}'"
        )
    
    # Extract the identifier
    start_idx = len(prefix)
    end_idx = len(filename) - len(suffix) if suffix else len(filename)
    identifier = filename[start_idx:end_idx]
    
    return identifier


def match_file_pairs(
    ref_files: List[Path],
    tgt_files: List[Path],
    ref_pattern: str,
    tgt_pattern: str
) -> List[Tuple[Path, Path, str]]:
    """
    Match reference and target files by sorted rank (1st with 1st, 2nd with 2nd, etc.).
    
    Files are sorted alphabetically within each directory, then paired by position.
    
    Args:
        ref_files: List of reference file paths (already sorted)
        tgt_files: List of target file paths (already sorted)
        ref_pattern: Pattern used to find reference files (used for validation)
        tgt_pattern: Pattern used to find target files (used for validation)
        
    Returns:
        List of tuples: (ref_path, tgt_path, identifier)
        where identifier is "pair_N" or derived from filenames
    """
    # Files should already be sorted from find_matching_files, but ensure it
    ref_sorted = sorted(ref_files)
    tgt_sorted = sorted(tgt_files)
    
    # Check that we have the same number of files
    if len(ref_sorted) != len(tgt_sorted):
        raise ValueError(
            f"Number of files mismatch:\n"
            f"  Reference directory has {len(ref_sorted)} files\n"
            f"  Target directory has {len(tgt_sorted)} files\n"
            f"  Both directories must have the same number of matching files."
        )
    
    # Create matched pairs by position
    pairs = []
    for idx, (ref_file, tgt_file) in enumerate(zip(ref_sorted, tgt_sorted), 1):
        # Try to extract identifier for display, fallback to pair number
        try:
            ref_id = extract_identifier(ref_file, ref_pattern)
            identifier = f"{ref_id}"
        except Exception:
            identifier = f"pair_{idx:03d}"
        
        pairs.append((ref_file, tgt_file, identifier))
    
    return pairs


def validate_images_compatible(
    ref_path: Path,
    tgt_path: Path,
    ref_arr: np.ndarray,
    tgt_arr: np.ndarray,
    ref_spacing: Tuple[float, float, float],
    tgt_spacing: Tuple[float, float, float]
) -> None:
    """
    Validate that reference and target images are compatible for comparison.
    
    Args:
        ref_path: Path to reference image
        tgt_path: Path to target image
        ref_arr: Reference image array
        tgt_arr: Target image array
        ref_spacing: Reference image spacing
        tgt_spacing: Target image spacing
        
    Raises:
        ValueError: If images are not compatible
    """
    # Check shapes match
    if ref_arr.shape != tgt_arr.shape:
        raise ValueError(
            f"Shape mismatch for {ref_path.name} vs {tgt_path.name}:\n"
            f"  Reference: {ref_arr.shape}\n"
            f"  Target: {tgt_arr.shape}"
        )
    
    # Check spacing matches (with tolerance)
    spacing_tolerance = 1e-5
    spacing_diff = [
        abs(r - t) for r, t in zip(ref_spacing, tgt_spacing)
    ]
    
    if any(diff > spacing_tolerance for diff in spacing_diff):
        logger.warning(
            f"Spacing mismatch for {ref_path.name} vs {tgt_path.name}:\n"
            f"  Reference: {ref_spacing} mm\n"
            f"  Target: {tgt_spacing} mm\n"
            f"  Using reference spacing for metrics."
        )


def _process_single_pair(
    pair_data: Tuple[Path, Path, str, int, int, Optional[List[int]], int]
) -> Tuple[str, Optional[ValidationResult], Optional[str]]:
    """
    Worker function to process a single validation pair.
    
    This function is designed to be run in a separate process.
    
    Args:
        pair_data: Tuple containing (ref_path, tgt_path, identifier, idx, total, labels, background_label)
        
    Returns:
        Tuple of (identifier, result, error_message)
        If successful, error_message is None
        If failed, result is None and error_message contains the error
    """
    ref_path, tgt_path, identifier, idx, total, labels, background_label = pair_data
    
    log_prefix = f"[{idx}/{total}] {identifier}"
    
    try:
        # Load images
        ref_arr, ref_spacing = load_segmentation(str(ref_path))
        tgt_arr, tgt_spacing = load_segmentation(str(tgt_path))
        
        # Validate compatibility
        validate_images_compatible(
            ref_path, tgt_path,
            ref_arr, tgt_arr,
            ref_spacing, tgt_spacing
        )
        
        # Compute metrics
        result = evaluate_segmentation(
            target=tgt_arr,
            ref=ref_arr,
            spacing=ref_spacing,
            labels=labels,
            background_label=background_label
        )
        
        return (identifier, result, None)
        
    except Exception as e:
        error_msg = f"{log_prefix}: {str(e)}"
        return (identifier, None, error_msg)


def batch_validate_segmentation(
    ref_dir: Path,
    ref_pattern: str,
    tgt_dir: Path,
    tgt_pattern: str,
    labels: Optional[List[int]] = None,
    background_label: int = 0,
    num_workers: int = 1
) -> Dict[str, ValidationResult]:
    """
    Batch validate segmentation images from two directories.
    
    Files are matched by sorted rank: the 1st file in the reference directory
    is compared with the 1st file in the target directory, 2nd with 2nd, etc.
    Both directories must contain the same number of files.
    
    Args:
        ref_dir: Directory containing reference images
        ref_pattern: Glob pattern for reference files (e.g., 'seg3d_*_tp*.nii.gz')
        tgt_dir: Directory containing target images
        tgt_pattern: Glob pattern for target files
        labels: Optional list of labels to evaluate
        background_label: Label value to exclude from evaluation
        num_workers: Number of parallel workers (default: 1, use -1 for all CPUs)
        
    Returns:
        Dictionary mapping identifiers to ValidationResult objects
    """
    logger.info(f"Searching for reference files in {ref_dir} with pattern '{ref_pattern}'")
    ref_files = find_matching_files(ref_dir, ref_pattern)
    logger.info(f"Found {len(ref_files)} reference files")
    
    logger.info(f"Searching for target files in {tgt_dir} with pattern '{tgt_pattern}'")
    tgt_files = find_matching_files(tgt_dir, tgt_pattern)
    logger.info(f"Found {len(tgt_files)} target files")
    
    # Match files
    logger.info("Matching reference and target files...")
    pairs = match_file_pairs(ref_files, tgt_files, ref_pattern, tgt_pattern)
    logger.info(f"Successfully matched {len(pairs)} file pairs")
    
    # Determine number of workers
    if num_workers == -1:
        num_workers = multiprocessing.cpu_count()
    num_workers = max(1, min(num_workers, len(pairs)))  # Don't use more workers than pairs
    
    logger.info(f"Processing with {num_workers} parallel workers...")
    
    # Process each pair
    results: Dict[str, ValidationResult] = {}
    
    if num_workers == 1:
        # Sequential processing (original behavior)
        for idx, (ref_path, tgt_path, identifier) in enumerate(pairs, 1):
            logger.info(f"\n[{idx}/{len(pairs)}] Processing pair: {identifier}")
            logger.info(f"  Reference: {ref_path.name}")
            logger.info(f"  Target: {tgt_path.name}")
            
            pair_data = (ref_path, tgt_path, identifier, idx, len(pairs), labels, background_label)
            identifier, result, error = _process_single_pair(pair_data)
            
            if error:
                logger.error(error)
            elif result:
                results[identifier] = result
                logger.info(
                    f"  Macro avg: Dice={result.macro_avg.dice:.4f}, "
                    f"MSD={result.macro_avg.msd:.3f} mm, "
                    f"HD95={result.macro_avg.hd95:.3f} mm"
                )
    else:
        # Parallel processing
        # Prepare work items
        work_items = [
            (ref_path, tgt_path, identifier, idx, len(pairs), labels, background_label)
            for idx, (ref_path, tgt_path, identifier) in enumerate(pairs, 1)
        ]
        
        # Process in parallel
        completed_count = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_work = {executor.submit(_process_single_pair, work): work for work in work_items}
            
            # Process results as they complete
            for future in as_completed(future_to_work):
                completed_count += 1
                work_item = future_to_work[future]
                identifier = work_item[2]  # identifier is 3rd element
                idx = work_item[3]  # idx is 4th element
                
                try:
                    identifier, result, error = future.result()
                    
                    if error:
                        logger.error(error)
                    elif result:
                        results[identifier] = result
                        logger.info(
                            f"[{completed_count}/{len(pairs)}] Completed {identifier}: "
                            f"Dice={result.macro_avg.dice:.4f}, "
                            f"MSD={result.macro_avg.msd:.3f} mm, "
                            f"HD95={result.macro_avg.hd95:.3f} mm"
                        )
                except Exception as e:
                    logger.error(f"Worker exception for {identifier}: {e}")
    
    return results


def results_to_dict(results: Dict[str, ValidationResult]) -> Dict:
    """
    Convert ValidationResult objects to a JSON-serializable dictionary.
    
    Args:
        results: Dictionary of ValidationResult objects
        
    Returns:
        JSON-serializable dictionary
    """
    output = {}
    
    for identifier, result in results.items():
        # Convert per-label metrics
        per_label = {}
        for label, metrics in result.per_label.items():
            per_label[str(label)] = {
                'dice': metrics.dice,
                'msd': metrics.msd,
                'hd95': metrics.hd95,
                'ref_voxels': result.ref_voxels[label],
                'target_voxels': result.target_voxels[label]
            }
        
        output[identifier] = {
            'macro_avg': {
                'dice': result.macro_avg.dice,
                'msd': result.macro_avg.msd,
                'hd95': result.macro_avg.hd95
            },
            'per_label': per_label
        }
    
    return output


def print_summary(results: Dict[str, ValidationResult]) -> None:
    """
    Print a summary of validation results.
    
    Args:
        results: Dictionary of ValidationResult objects
    """
    if not results:
        logger.warning("No results to display")
        return
    
    print("\n" + "="*80)
    print("BATCH VALIDATION SUMMARY")
    print("="*80)
    
    # Aggregate statistics
    all_dice = [r.macro_avg.dice for r in results.values()]
    all_msd = [r.macro_avg.msd for r in results.values()]
    all_hd95 = [r.macro_avg.hd95 for r in results.values()]
    
    print(f"\nTotal pairs processed: {len(results)}")
    print(f"\nAggregate Statistics (Macro Average across all pairs):")
    print(f"  Mean Dice:  {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    print(f"  Mean MSD:   {np.mean(all_msd):.3f} ± {np.std(all_msd):.3f} mm")
    print(f"  Mean HD95:  {np.mean(all_hd95):.3f} ± {np.std(all_hd95):.3f} mm")
    
    print(f"\nPer-Pair Results:")
    print(f"{'Identifier':<20} {'Dice':>8} {'MSD (mm)':>10} {'HD95 (mm)':>11}")
    print("-" * 80)
    
    for identifier in sorted(results.keys()):
        result = results[identifier]
        print(
            f"{identifier:<20} "
            f"{result.macro_avg.dice:>8.4f} "
            f"{result.macro_avg.msd:>10.3f} "
            f"{result.macro_avg.hd95:>11.3f}"
        )
    
    print("="*80)


def plot_metrics(results: Dict[str, ValidationResult], output_path: Path) -> None:
    """
    Create plots showing how metrics change across file numbers.
    
    Args:
        results: Dictionary of ValidationResult objects
        output_path: Path to save the plot image
    """
    if not results:
        logger.warning("No results to plot")
        return
    
    # Extract data in sorted order
    identifiers = sorted(results.keys())
    file_numbers = list(range(1, len(identifiers) + 1))
    
    dice_values = [results[id_].macro_avg.dice for id_ in identifiers]
    msd_values = [results[id_].macro_avg.msd for id_ in identifiers]
    hd95_values = [results[id_].macro_avg.hd95 for id_ in identifiers]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Segmentation Validation Metrics Over File Sequence', fontsize=14, fontweight='bold')
    
    # Plot Dice coefficient
    axes[0].plot(file_numbers, dice_values, 'o-', color='#2E86AB', linewidth=2, markersize=6)
    axes[0].axhline(y=np.mean(dice_values), color='red', linestyle='--', 
                     label=f'Mean: {np.mean(dice_values):.4f}')
    axes[0].set_ylabel('Dice Coefficient', fontsize=11, fontweight='bold')
    axes[0].set_ylim([max(0, min(dice_values) - 0.05), min(1.0, max(dice_values) + 0.05)])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best')
    axes[0].set_title('Dice Coefficient (higher is better)', fontsize=10)
    
    # Plot MSD (ASSD)
    axes[1].plot(file_numbers, msd_values, 'o-', color='#A23B72', linewidth=2, markersize=6)
    axes[1].axhline(y=np.mean(msd_values), color='red', linestyle='--',
                     label=f'Mean: {np.mean(msd_values):.3f} mm')
    axes[1].set_ylabel('MSD (ASSD) [mm]', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best')
    axes[1].set_title('Mean Surface Distance (lower is better)', fontsize=10)
    
    # Plot HD95
    axes[2].plot(file_numbers, hd95_values, 'o-', color='#F18F01', linewidth=2, markersize=6)
    axes[2].axhline(y=np.mean(hd95_values), color='red', linestyle='--',
                     label=f'Mean: {np.mean(hd95_values):.3f} mm')
    axes[2].set_ylabel('HD95 [mm]', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('File Number (sorted order)', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='best')
    axes[2].set_title('Hausdorff Distance 95th Percentile (lower is better)', fontsize=10)
    
    # Add file identifiers as x-axis labels if not too many
    if len(identifiers) <= 20:
        axes[2].set_xticks(file_numbers)
        axes[2].set_xticklabels(identifiers, rotation=45, ha='right', fontsize=8)
    else:
        axes[2].set_xlabel(f'File Number (sorted order, n={len(identifiers)})', 
                          fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plot saved to: {output_path}")


def main():
    """CLI entry point for batch segmentation validation."""
    parser = argparse.ArgumentParser(
        description=(
            "Batch validate segmentation images from two directories.\n"
            "Files are matched by sorted rank: 1st ref file vs 1st target file, "
            "2nd vs 2nd, etc."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  batch-segmentation-validation \\
      --ref-dir /path/to/reference \\
      --ref-fn "seg3d_bav_tp*.nii.gz" \\
      --tgt-dir /path/to/target \\
      --tgt-fn "seg3d_bav_tp*.nii.gz"
  
  # With output directory and parallel processing (uses all CPUs)
  batch-segmentation-validation \\
      --ref-dir ./ground_truth \\
      --ref-fn "label_*.nii.gz" \\
      --tgt-dir ./predictions \\
      --tgt-fn "pred_*.nii.gz" \\
      --output-dir ./validation_results \\
      --workers -1
  
  # With specific labels and 4 parallel workers
  batch-segmentation-validation \\
      --ref-dir ./ref \\
      --ref-fn "seg_*.nii.gz" \\
      --tgt-dir ./tgt \\
      --tgt-fn "seg_*.nii.gz" \\
      --labels 1 2 3 \\
      --background-label 0 \\
      --workers 4

Note:
  Files in each directory are sorted alphabetically and matched by position.
  Both directories must contain the same number of files matching the patterns.
  Use --workers -1 to utilize all available CPU cores for faster processing.
        """
    )
    
    parser.add_argument(
        "--ref-dir",
        type=str,
        required=True,
        help="Directory containing reference segmentation images"
    )
    parser.add_argument(
        "--ref-fn",
        type=str,
        required=True,
        help="Filename pattern for reference images (e.g., 'seg3d_bav_tp*.nii.gz')"
    )
    parser.add_argument(
        "--tgt-dir",
        type=str,
        required=True,
        help="Directory containing target segmentation images"
    )
    parser.add_argument(
        "--tgt-fn",
        type=str,
        required=True,
        help="Filename pattern for target images (e.g., 'seg3d_bav_tp*.nii.gz')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional: Directory to save results (JSON file and metric plots)"
    )
    parser.add_argument(
        "--labels",
        type=int,
        nargs="+",
        default=None,
        help="Specific labels to evaluate (e.g., 1 2 3). If not provided, evaluates all non-background labels"
    )
    parser.add_argument(
        "--background-label",
        type=int,
        default=0,
        help="Background label to exclude from evaluation (default: 0)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing (default: 1, use -1 for all CPUs)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    ref_dir = Path(args.ref_dir)
    tgt_dir = Path(args.tgt_dir)
    
    # Run batch validation
    try:
        results = batch_validate_segmentation(
            ref_dir=ref_dir,
            ref_pattern=args.ref_fn,
            tgt_dir=tgt_dir,
            tgt_pattern=args.tgt_fn,
            labels=args.labels,
            background_label=args.background_label,
            num_workers=args.workers
        )
        
        # Print summary
        print_summary(results)
        
        # Save to output directory if requested
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON results
            json_path = output_dir / "validation_results.json"
            output_data = results_to_dict(results)
            
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"\nJSON results saved to: {json_path}")
            
            # Save plot
            plot_path = output_dir / "validation_metrics.png"
            plot_metrics(results, plot_path)
            
            logger.info(f"Output directory: {output_dir}")
        
        # Exit with appropriate code
        if results:
            logger.info("\n✓ Batch validation completed successfully")
            return 0
        else:
            logger.error("\n✗ No results were generated")
            return 1
            
    except Exception as e:
        logger.error(f"\n✗ Batch validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())