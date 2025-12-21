"""Example demonstrating HTCondor job submission with file transfers.

This example shows how to submit agent jobs to HTCondor with proper
file transfer configuration for remote execution on batch farm nodes.

For comprehensive documentation, see FILE_TRANSFER_GUIDE.md
"""

import shlex
import shutil
from pathlib import Path
from chatlas_agents.htcondor import HTCondorJobSubmitter


def example_basic_htcondor_submission():
    """Basic HTCondor job with file transfers."""
    print("\n=== Example 1: Basic HTCondor Job ===")
    
    # Setup job directory (would typically be on submit node or AFS)
    job_name = "basic_job"
    output_dir = Path("./htcondor_examples") / job_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create input files
    (output_dir / "input.txt").write_text("Process this data")
    (output_dir / "config.yaml").write_text("""
llm:
  provider: openai
  model: gpt-4
""")
    
    # Create submitter
    submitter = HTCondorJobSubmitter(
        docker_image="python:3.13-slim",
        output_dir=output_dir,
    )
    
    # Generate submit file with file transfer
    submit_file = submitter.generate_submit_file(
        job_name=job_name,
        prompt="Process the input file",
        config_file="config.yaml",
        # HTCondor file transfer parameters
        should_transfer_files="YES",
        when_to_transfer_output="ON_EXIT",
        transfer_input_files="input.txt,config.yaml",
        transfer_output_files="results/",
        request_cpus=1,
        request_memory="2GB",
        request_disk="1GB",
    )
    
    print(f"✓ Generated submit file: {submit_file}")
    print(f"\nSubmit file content:")
    print("-" * 60)
    print(submit_file.read_text())
    print("-" * 60)
    
    print(f"\nTo submit: condor_submit {submit_file}")
    print(f"Results will be in: {output_dir}/")


def example_htcondor_with_network_storage():
    """HTCondor job using network storage (AFS/EOS) instead of file transfer."""
    print("\n=== Example 2: HTCondor with Network Storage ===")
    
    job_name = "network_storage_job"
    output_dir = Path("./htcondor_examples") / job_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # For this example, we simulate network storage paths
    # In practice, these would be actual AFS/EOS paths
    # SECURITY NOTE: In production, validate and sanitize these paths
    network_input = "/eos/project/atlas/data/input.root"
    network_output = "/eos/project/atlas/results/"
    
    submitter = HTCondorJobSubmitter(
        docker_image="atlas/athanalysis:latest",
        output_dir=output_dir,
    )
    
    # Build command safely - in production, validate paths before use
    # This example uses known-safe hardcoded paths
    prompt = f"Process {network_input} and save to {network_output}"
    custom_command = f"""
# Access network storage directly (no file transfer needed)
python3 -m chatlas_agents.cli run \\
    --input {shlex.quote(prompt)} \\
    --docker-sandbox \\
    --docker-image atlas/athanalysis:latest
"""
    
    # Escape command for HTCondor submit file
    escaped_command = shlex.quote(custom_command)
    
    # Generate submit file
    submit_content = f"""# HTCondor submit file for {job_name}
# Uses network storage instead of file transfers

executable = /bin/bash
arguments = -c {escaped_command}

# Docker universe
universe = docker
docker_image = atlas/athanalysis:latest

# Output, error, and log files
output = {output_dir}/job.$(ClusterId).$(ProcId).out
error = {output_dir}/job.$(ClusterId).$(ProcId).err
log = {output_dir}/job.$(ClusterId).log

# No file transfer - using network storage
should_transfer_files = NO

# Resource requirements
request_cpus = 4
request_memory = 8GB
request_disk = 10GB

# Queue the job
queue 1
"""
    
    submit_file = output_dir / f"{job_name}.sub"
    submit_file.write_text(submit_content)
    
    print(f"✓ Generated submit file: {submit_file}")
    print(f"\nThis job uses network storage paths:")
    print(f"  Input:  {network_input}")
    print(f"  Output: {network_output}")
    print(f"\nNo file transfers needed - faster and more efficient!")


def example_htcondor_batch_processing():
    """HTCondor batch processing with multiple input files."""
    print("\n=== Example 3: Batch Processing ===")
    
    job_name = "batch_processing"
    output_dir = Path("./htcondor_examples") / job_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create multiple input files
    for i in range(3):
        (output_dir / f"input_{i}.csv").write_text(f"data,{i}\nvalue,{i*10}\n")
    
    submitter = HTCondorJobSubmitter(
        docker_image="python:3.13-slim",
        output_dir=output_dir,
    )
    
    # Submit multiple jobs (one per input file)
    for i in range(3):
        job_sub_name = f"{job_name}_{i}"
        
        submit_file = submitter.generate_submit_file(
            job_name=job_sub_name,
            prompt=f"Process input_{i}.csv",
            should_transfer_files="YES",
            when_to_transfer_output="ON_EXIT",
            transfer_input_files=f"input_{i}.csv",
            transfer_output_files=f"output_{i}.json",
        )
        
        print(f"✓ Generated submit file for job {i}: {submit_file}")
    
    print(f"\nTo submit all jobs:")
    print(f"  cd {output_dir}")
    print(f"  for f in *.sub; do condor_submit $f; done")


def example_htcondor_large_output():
    """HTCondor job with large output files - use compression."""
    print("\n=== Example 4: Large Output with Compression ===")
    
    job_name = "large_output_job"
    output_dir = Path("./htcondor_examples") / job_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    submitter = HTCondorJobSubmitter(
        docker_image="python:3.13-slim",
        output_dir=output_dir,
    )
    
    # Custom command that compresses output
    # Using set -e to fail on any error
    custom_command = """set -e
# Run agent to generate large output
python3 -m chatlas_agents.cli run \\
    --input 'Generate large analysis report' \\
    --docker-sandbox

# Compress results before transfer
cd /workspace
tar czf results.tar.gz results/

echo 'Compressed results created'
"""
    
    # Escape command for HTCondor submit file
    escaped_command = shlex.quote(custom_command)
    
    submit_content = f"""# HTCondor submit file with output compression

executable = /bin/bash
arguments = -c {escaped_command}

universe = docker
docker_image = python:3.13-slim

output = {output_dir}/job.$(ClusterId).$(ProcId).out
error = {output_dir}/job.$(ClusterId).$(ProcId).err
log = {output_dir}/job.$(ClusterId).log

# File transfer
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_output_files = results.tar.gz

# Increase disk request for large output
request_cpus = 1
request_memory = 4GB
request_disk = 20GB

queue 1
"""
    
    submit_file = output_dir / f"{job_name}.sub"
    submit_file.write_text(submit_content)
    
    print(f"✓ Generated submit file: {submit_file}")
    print(f"\nThis job compresses output before transfer:")
    print(f"  - Reduces transfer time")
    print(f"  - Saves disk space")
    print(f"  - Extract with: tar xzf results.tar.gz")


def example_htcondor_monitoring():
    """Example: monitoring HTCondor job and retrieving results."""
    print("\n=== Example 5: Job Monitoring and Result Retrieval ===")
    
    # This is a demonstration of the workflow
    # In practice, you would actually submit the job
    
    print("""
HTCondor Job Lifecycle:

1. Submit job:
   $ condor_submit job.sub
   Submitting job(s).
   1 job(s) submitted to cluster 12345.

2. Monitor job status:
   $ condor_q 12345
   ID      OWNER            SUBMITTED     RUN_TIME ST PRI SIZE CMD
   12345.0 username        12/21 10:30   0+00:05:12 R  0   2.0  job.sh

3. Wait for completion:
   - Job will disappear from condor_q when done
   - Check condor_history for completed jobs:
   $ condor_history 12345

4. Retrieve results:
""")
    
    # Simulate job completion
    job_dir = Path("./htcondor_examples/basic_job")
    cluster_id = "12345"
    
    print(f"\n   Job directory: {job_dir}/")
    print(f"   Output files:")
    print(f"     - job.{cluster_id}.0.out  (stdout)")
    print(f"     - job.{cluster_id}.0.err  (stderr)")
    print(f"     - job.{cluster_id}.log    (HTCondor log)")
    print(f"     - results/                (transferred output)")
    
    print(f"\n5. Process results:")
    print("""
   import json
   from pathlib import Path
   
   job_dir = Path("./htcondor_examples/basic_job")
   
   # Read stdout
   stdout = (job_dir / "job.12345.0.out").read_text()
   print(stdout)
   
   # Process result files
   results_dir = job_dir / "results"
   for file in results_dir.rglob("*"):
       if file.is_file():
           print(f"Result: {file}")
   """)


def example_htcondor_dry_run():
    """Dry run example - generate submit file without submitting."""
    print("\n=== Example 6: Dry Run (Generate Submit File Only) ===")
    
    job_name = "dry_run_example"
    output_dir = Path("./htcondor_examples") / job_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample input
    (output_dir / "data.csv").write_text("test,data\n1,2\n3,4\n")
    
    submitter = HTCondorJobSubmitter(
        docker_image="python:3.13-slim",
        output_dir=output_dir,
    )
    
    # Dry run - only generate submit file
    cluster_id = submitter.submit_job(
        job_name=job_name,
        prompt="Process data.csv",
        transfer_input_files="data.csv",
        transfer_output_files="results/",
        dry_run=True,  # Don't actually submit
    )
    
    print(f"✓ Dry run complete")
    print(f"  cluster_id: {cluster_id} (None because dry_run=True)")
    print(f"  Submit file created: {output_dir}/{job_name}.sub")
    print(f"\nReview the submit file before actual submission")


def main():
    """Run all examples."""
    print("=" * 70)
    print("HTCondor File Transfer Examples")
    print("=" * 70)
    print("\nThese examples demonstrate HTCondor job submission with various")
    print("file transfer strategies for remote batch execution.")
    print("\nFor comprehensive documentation, see FILE_TRANSFER_GUIDE.md")
    print("=" * 70)
    
    # Run examples (all are dry-run / demonstration)
    example_basic_htcondor_submission()
    example_htcondor_with_network_storage()
    example_htcondor_batch_processing()
    example_htcondor_large_output()
    example_htcondor_monitoring()
    example_htcondor_dry_run()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("\nNote: These examples generate submit files but don't actually")
    print("submit jobs. Use condor_submit to submit the generated files.")
    print("=" * 70)
    
    # Cleanup info
    print("\nGenerated files are in: ./htcondor_examples/")
    print("To clean up: rm -rf ./htcondor_examples/")


if __name__ == "__main__":
    # Check if HTCondor is available
    if shutil.which("condor_submit"):
        print("✓ HTCondor detected - examples will generate submit files")
    else:
        print("⚠ HTCondor not detected - examples will still run (dry run mode)")
    
    main()
