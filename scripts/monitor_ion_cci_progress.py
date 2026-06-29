"""
Real-time IoN-CCI Processing Progress Monitor
"""

import sys
import subprocess
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
WSL_DISTRO = "Ubuntu-24.04"
PROJECT_ROOT_WSL = "/mnt/e/TN Omics"

def check_count_files():
    """Check how many count files exist (STAR ReadsPerGene or featureCounts)."""
    # Check for STAR ReadsPerGene files (primary)
    cmd = f"wsl -d {WSL_DISTRO} -- bash -c \"find '{PROJECT_ROOT_WSL}/results/ion_cci_processing/processed' -name '*ReadsPerGene.out.tab' 2>/dev/null | wc -l\""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        star_counts = int(result.stdout.strip())
        if star_counts > 0:
            return star_counts
    # Fallback: check for featureCounts files
    cmd = f"wsl -d {WSL_DISTRO} -- bash -c \"find '{PROJECT_ROOT_WSL}/results/ion_cci_processing/processed' -name 'counts.txt' -o -name '*featureCounts.txt' 2>/dev/null | wc -l\""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        return int(result.stdout.strip())
    return 0

def check_bam_files():
    """Check how many BAM files exist and their sizes."""
    cmd = f"wsl -d {WSL_DISTRO} -- bash -c \"find '{PROJECT_ROOT_WSL}/results/ion_cci_processing/processed' -name '*_Aligned.sortedByCoord.out.bam' -exec ls -lh {{}} \\; 2>/dev/null | grep -v ' 0 ' | wc -l\""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        return int(result.stdout.strip())
    return 0

def check_active_processes():
    """Check for active STAR or featureCounts processes."""
    cmd = f"wsl -d {WSL_DISTRO} -- bash -c \"ps aux | grep -E '[S]TAR|[f]eatureCounts' | wc -l\""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        return int(result.stdout.strip())
    return 0

def get_sample_status():
    """Get status of each sample."""
    samples = [f"SRR2515854{i}" for i in range(7, 9)] + [f"SRR2515855{i}" for i in range(0, 9)]
    
    status = {}
    for sample_id in samples:
        # Check for STAR ReadsPerGene file (primary) or featureCounts
        cmd = f"wsl -d {WSL_DISTRO} -- bash -c \"test -f '{PROJECT_ROOT_WSL}/results/ion_cci_processing/processed/{sample_id}/{sample_id}_ReadsPerGene.out.tab' && echo 'complete' || (test -f '{PROJECT_ROOT_WSL}/results/ion_cci_processing/processed/{sample_id}/counts.txt' && echo 'complete') || echo 'pending'\""
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if 'complete' in result.stdout:
            status[sample_id] = '[COMPLETE]'
        else:
            # Check for BAM file
            cmd = f"wsl -d {WSL_DISTRO} -- bash -c \"test -f '{PROJECT_ROOT_WSL}/results/ion_cci_processing/processed/{sample_id}/{sample_id}_Aligned.sortedByCoord.out.bam' && ls -lh '{PROJECT_ROOT_WSL}/results/ion_cci_processing/processed/{sample_id}/{sample_id}_Aligned.sortedByCoord.out.bam' 2>/dev/null | awk '{{print $5}}'\""
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout.strip() and result.stdout.strip() != '0':
                status[sample_id] = '[ALIGNED - quantifying...]'
            else:
                status[sample_id] = '[PROCESSING...]'
    
    return status

def main():
    """Monitor progress."""
    print("=" * 70)
    print("ION-CCI PROCESSING - REAL-TIME PROGRESS")
    print("=" * 70)
    print("\nPress Ctrl+C to stop monitoring (processing continues)\n")
    
    try:
        while True:
            count_files = check_count_files()
            bam_files = check_bam_files()
            active = check_active_processes()
            status = get_sample_status()
            
            # Clear screen (optional)
            print("\033[2J\033[H", end="")  # ANSI clear screen
            
            print("=" * 70)
            print("ION-CCI PROCESSING - REAL-TIME PROGRESS")
            print("=" * 70)
            print(f"\nOverall Progress: {count_files}/12 samples quantified")
            print(f"Aligned BAM files: {bam_files}/12")
            print(f"Active processes: {active}")
            print("\n" + "-" * 70)
            print("Sample Status:")
            print("-" * 70)
            
            for sample_id, stat in sorted(status.items()):
                print(f"  {sample_id}: {stat}")
            
            print("\n" + "-" * 70)
            print(f"Last update: {time.strftime('%H:%M:%S')}")
            print("=" * 70)
            
            if count_files == 12:
                print("\n*** ALL SAMPLES COMPLETE! ***")
                break
            
            time.sleep(10)  # Update every 10 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped. Processing continues in background.")
        print(f"Current progress: {check_count_files()}/12 samples")

if __name__ == "__main__":
    main()
