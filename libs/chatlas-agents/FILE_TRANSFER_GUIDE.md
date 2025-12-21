# File Transfer Guide for Sandboxes and HTCondor

This guide explains how to handle file transfers between host and sandbox environments, with special considerations for remote containers on HTCondor batch farm nodes.

## Table of Contents

1. [Overview](#overview)
2. [Backend Protocol File Transfer API](#backend-protocol-file-transfer-api)
3. [File Transfer Patterns](#file-transfer-patterns)
4. [Local Sandboxes (Docker, Apptainer)](#local-sandboxes-docker-apptainer)
5. [HTCondor Remote Execution](#htcondor-remote-execution)
6. [Best Practices](#best-practices)
7. [Examples](#examples)

## Overview

File transfers are essential when working with sandboxed environments because:

- **Agents run on host**, but **code executes in isolated sandbox**
- **Input files** need to be transferred from host to sandbox
- **Output files** (results, logs, artifacts) need to be retrieved from sandbox to host
- **Remote execution** (HTCondor) adds another layer of transfer complexity

### File Transfer Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      Host Machine                            │
│  ┌────────────┐                                              │
│  │   Agent    │                                              │
│  │  Process   │                                              │
│  └─────┬──────┘                                              │
│        │                                                      │
│        │ upload_files() / download_files()                   │
│        ↓                                                      │
│  ┌─────────────────┐                                         │
│  │ Sandbox Backend │ ← Implements file transfer protocol    │
│  └────────┬────────┘                                         │
└───────────┼─────────────────────────────────────────────────┘
            │
            │ Docker cp / Apptainer exec / HTTP API
            ↓
┌───────────────────────────────────────────────────────────────┐
│                    Sandbox Container                          │
│  ┌──────────────────────────────────────────────────────┐    │
│  │         /workspace/                                   │    │
│  │  ├── input_data.csv                                   │    │
│  │  ├── script.py                                        │    │
│  │  └── results/                                         │    │
│  │      └── output.json                                  │    │
│  └──────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────┘
```

For HTCondor, add another layer:

```
Host → HTCondor Submit Node → HTCondor Execute Node (Container) → Results back
```

## Backend Protocol File Transfer API

The DeepAgents `BackendProtocol` defines standardized file transfer methods:

### Upload Files

```python
def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
    """Upload multiple files to the sandbox.
    
    Args:
        files: List of (path, content) tuples to upload.
               - path: Destination path in sandbox (e.g., "/workspace/data.txt")
               - content: File content as bytes
    
    Returns:
        List of FileUploadResponse objects, one per input file.
        Response order matches input order (response[i] for files[i]).
        Check the error field to determine success/failure per file.
    """
```

**FileUploadResponse:**
```python
@dataclass
class FileUploadResponse:
    path: str  # The requested file path
    error: FileOperationError | None  # None on success, error code on failure
```

### Download Files

```python
def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
    """Download multiple files from the sandbox.
    
    Args:
        paths: List of file paths to download from sandbox.
    
    Returns:
        List of FileDownloadResponse objects, one per input path.
        Response order matches input order (response[i] for paths[i]).
        Check the error field to determine success/failure per file.
    """
```

**FileDownloadResponse:**
```python
@dataclass
class FileDownloadResponse:
    path: str  # The requested file path
    content: bytes | None  # File content on success, None on failure
    error: FileOperationError | None  # None on success, error code on failure
```

### Error Codes

Standardized `FileOperationError` codes for LLM-actionable error handling:

- `"file_not_found"` - File doesn't exist (download)
- `"permission_denied"` - Access denied
- `"is_directory"` - Tried to download directory as file
- `"invalid_path"` - Path syntax malformed

## File Transfer Patterns

### Pattern 1: Direct Transfer (Local Sandboxes)

**When to use:** Docker or Apptainer running on the same machine as the agent.

**How it works:**
- Agent calls `backend.upload_files()` or `backend.download_files()`
- Backend uses container CLI (`docker cp`, `apptainer exec`) to transfer files
- Files are transferred directly between host filesystem and container

```python
from chatlas_agents.sandbox import create_docker_sandbox
from pathlib import Path

# Read file from host
input_data = Path("data.csv").read_bytes()

with create_docker_sandbox() as backend:
    # Upload to sandbox
    responses = backend.upload_files([
        ("/workspace/input.csv", input_data),
    ])
    
    # Check for errors
    for resp in responses:
        if resp.error:
            print(f"Upload failed: {resp.path} - {resp.error}")
    
    # ... agent processes files in sandbox ...
    
    # Download results
    responses = backend.download_files([
        "/workspace/results/output.json",
        "/workspace/logs/execution.log",
    ])
    
    for resp in responses:
        if resp.error:
            print(f"Download failed: {resp.path} - {resp.error}")
        else:
            # Save to host
            output_path = Path(resp.path.lstrip("/"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(resp.content)
```

### Pattern 2: Agent-Driven Transfer

**When to use:** Agent needs to decide which files to transfer based on task requirements.

**How it works:**
- Expose file transfer as tools to the agent
- Agent can request uploads/downloads as needed
- Useful for interactive workflows

```python
from langchain_core.tools import tool

# Create tools for agent
@tool
def upload_file_to_sandbox(host_path: str, sandbox_path: str) -> str:
    """Upload a file from host to sandbox.
    
    Args:
        host_path: Path to file on host machine
        sandbox_path: Destination path in sandbox (must start with /)
    """
    content = Path(host_path).read_bytes()
    responses = backend.upload_files([(sandbox_path, content)])
    
    if responses[0].error:
        return f"Error: {responses[0].error}"
    return f"Successfully uploaded {host_path} to {sandbox_path}"

@tool
def download_file_from_sandbox(sandbox_path: str, host_path: str) -> str:
    """Download a file from sandbox to host.
    
    Args:
        sandbox_path: Path to file in sandbox
        host_path: Destination path on host machine
    """
    responses = backend.download_files([sandbox_path])
    
    if responses[0].error:
        return f"Error: {responses[0].error}"
    
    Path(host_path).write_bytes(responses[0].content)
    return f"Successfully downloaded {sandbox_path} to {host_path}"

# Add to agent
agent = create_deep_agent(
    backend=backend,
    tools=[upload_file_to_sandbox, download_file_from_sandbox],
)
```

### Pattern 3: Batch Transfer with Error Handling

**When to use:** Transferring multiple files with partial success support.

**How it works:**
- APIs support batch operations
- Individual file failures don't abort entire transfer
- Process successes and failures separately

```python
# Batch upload
files_to_upload = [
    ("/workspace/data1.csv", data1_bytes),
    ("/workspace/data2.csv", data2_bytes),
    ("/workspace/config.json", config_bytes),
]

responses = backend.upload_files(files_to_upload)

# Separate successes and failures
succeeded = [r for r in responses if r.error is None]
failed = [r for r in responses if r.error is not None]

print(f"Uploaded {len(succeeded)} files successfully")
for resp in failed:
    print(f"Failed to upload {resp.path}: {resp.error}")

# Batch download with retry logic
def download_with_retry(backend, paths, max_retries=3):
    """Download files with retry on failure."""
    results = {}
    remaining = list(paths)
    
    for attempt in range(max_retries):
        if not remaining:
            break
            
        responses = backend.download_files(remaining)
        
        # Process results
        still_failing = []
        for resp in responses:
            if resp.error is None:
                results[resp.path] = resp.content
            else:
                still_failing.append(resp.path)
        
        remaining = still_failing
        if remaining and attempt < max_retries - 1:
            print(f"Retry attempt {attempt + 1} for {len(remaining)} files...")
    
    return results, remaining

# Usage
paths = ["/workspace/result1.json", "/workspace/result2.json"]
succeeded, failed = download_with_retry(backend, paths)
```

## Local Sandboxes (Docker, Apptainer)

### Docker Sandbox Backend

**Implementation:** Uses `docker cp` command for file transfers.

**Upload Process:**
1. Write content to temporary file on host
2. Use `docker cp temp_file container:/dest/path`
3. Clean up temporary file

**Download Process:**
1. Use `docker cp container:/src/path temp_file`
2. Read content from temporary file
3. Clean up temporary file

```python
from chatlas_agents.sandbox import DockerSandboxBackend

backend = DockerSandboxBackend(
    image="python:3.13-slim",
    working_dir="/workspace",
)

# Upload example
data = b"Hello, World!"
responses = backend.upload_files([
    ("/workspace/hello.txt", data),
])

# Download example
responses = backend.download_files([
    "/workspace/hello.txt",
])
print(responses[0].content.decode())  # "Hello, World!"
```

### Apptainer Sandbox Backend

**Implementation:** Uses `apptainer exec` with stdin/stdout redirection.

**Upload Process:**
1. Use `apptainer exec instance://{name} sh -c 'cat > /dest/path'` with stdin
2. Pipe file content to stdin

**Download Process:**
1. Use `apptainer exec instance://{name} cat /src/path`
2. Capture stdout as file content

```python
from chatlas_agents.sandbox import ApptainerSandboxBackend

backend = ApptainerSandboxBackend(
    image="docker://python:3.13-slim",
    working_dir="/workspace",
)

# Same API as Docker
responses = backend.upload_files([
    ("/workspace/data.txt", b"data content"),
])
```

### Limitations

**File size limits:**
- Docker: Limited by available disk space
- Apptainer: Same as Docker, but stdin/stdout approach may have buffer limits for very large files

**Path restrictions:**
- Paths must be absolute (start with `/`)
- Parent directories must exist (create with `backend.execute("mkdir -p /parent/dir")`)

**Permissions:**
- Files created with container's default user permissions
- May need to adjust permissions after upload

## HTCondor Remote Execution

HTCondor batch jobs run on remote execute nodes, requiring a multi-stage file transfer process.

### HTCondor File Transfer Architecture

```
┌──────────────┐         ┌──────────────────┐         ┌──────────────────┐
│     Host     │────────▶│ HTCondor Submit  │────────▶│ HTCondor Execute │
│   Machine    │   (1)   │      Node        │   (2)   │   Node (Remote)  │
│              │         │                  │         │                  │
│  Agent runs  │         │  Job queued,     │         │  Job executes in │
│  here        │         │  files staged    │         │  Docker container│
└──────────────┘         └──────────────────┘         └──────────────────┘
                                   │                           │
                                   │◀──────────────────────────┘
                                   │         (3) Results
                                   │
                                   ▼
                         ┌──────────────────┐
                         │  Output files    │
                         │  Available for   │
                         │  retrieval       │
                         └──────────────────┘
```

**Transfer stages:**

1. **Host → Submit Node:** User uploads input files to submit node (manual or via network share)
2. **Submit Node → Execute Node:** HTCondor's `transfer_input_files` automatically transfers files to execute node
3. **Execute Node → Submit Node:** HTCondor's `transfer_output_files` automatically transfers results back
4. **Submit Node → Host:** User downloads results from submit node

### HTCondor Submit File Configuration

Enable file transfer in HTCondor submit files:

```bash
# Enable file transfer mechanism
should_transfer_files = YES
when_to_transfer_output = ON_EXIT

# Specify input files (comma-separated, relative to submit directory)
transfer_input_files = input_data.csv, config.yaml, scripts/

# Specify output files/directories to transfer back
# Use remaps for specific destinations
transfer_output_files = results/, logs/execution.log
transfer_output_remaps = "results/output.json=output.json"

# Working directory in the container
initialdir = /workspace
```

### HTCondor Integration Strategy

**Option 1: Pre-stage Files on Submit Node**

Best for: Known input files, batch processing

```python
from chatlas_agents.htcondor import HTCondorJobSubmitter
from pathlib import Path

# 1. Prepare files on submit node (or accessible shared filesystem)
submit_node_dir = Path("/afs/cern.ch/user/username/htcondor_jobs/job1")
submit_node_dir.mkdir(parents=True, exist_ok=True)

# Copy input files to submit directory
shutil.copy("input_data.csv", submit_node_dir / "input_data.csv")
shutil.copy("config.yaml", submit_node_dir / "config.yaml")

# 2. Generate submit file with file transfer config
submitter = HTCondorJobSubmitter(
    docker_image="python:3.13-slim",
    output_dir=submit_node_dir,
)

# 3. Add custom submit parameters for file transfer
submit_file = submitter.generate_submit_file(
    job_name="my_job",
    prompt="Process the data",
    should_transfer_files="YES",
    when_to_transfer_output="ON_EXIT",
    transfer_input_files="input_data.csv,config.yaml",
    transfer_output_files="results/",
)

# 4. Submit job
cluster_id = submitter.submit_job(...)

# 5. Wait for job to complete, then retrieve results
# Results will be in submit_node_dir/results/
```

**Option 2: Embedded Files in Command**

Best for: Small text files, configuration

```python
# Embed small files as base64 in job command
import base64

config_content = """
model: gpt-4
temperature: 0.7
"""

config_b64 = base64.b64encode(config_content.encode()).decode()

# Generate submit file with file creation command
submit_content = f"""
executable = /bin/bash
arguments = -c "echo {config_b64} | base64 -d > /workspace/config.yaml && python -m chatlas_agents.cli run --input 'Process data'"
"""
```

**Option 3: Network Storage Access**

Best for: Large datasets, shared resources

```python
# Submit file uses network storage (AFS, EOS, CVMFS)
submit_content = """
# Access network storage
transfer_input_files = 

# Command mounts or accesses network storage
arguments = -c "
    # Copy from network storage to local workspace
    cp /eos/project/atlas/data/input.csv /workspace/
    
    # Run agent
    python -m chatlas_agents.cli run --input 'Process data'
    
    # Copy results back to network storage
    cp /workspace/results/* /eos/project/atlas/results/
"
"""
```

### Retrieving Results from HTCondor Jobs

```python
def wait_for_job_completion(submitter, cluster_id, timeout=3600):
    """Poll job status until completion."""
    import time
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = submitter.query_job_status(cluster_id)
        
        # Parse status to check if completed
        if status and "0 jobs" in status:  # No jobs in queue
            return True
        
        time.sleep(30)  # Poll every 30 seconds
    
    return False

def retrieve_results(job_dir: Path) -> dict:
    """Retrieve all result files from job directory."""
    results = {}
    
    # Read output files
    output_file = job_dir / f"job.{cluster_id}.0.out"
    error_file = job_dir / f"job.{cluster_id}.0.err"
    log_file = job_dir / f"job.{cluster_id}.log"
    
    results["stdout"] = output_file.read_text() if output_file.exists() else ""
    results["stderr"] = error_file.read_text() if error_file.exists() else ""
    results["log"] = log_file.read_text() if log_file.exists() else ""
    
    # Read transferred output files
    results_dir = job_dir / "results"
    if results_dir.exists():
        results["output_files"] = {}
        for file_path in results_dir.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(results_dir)
                results["output_files"][str(rel_path)] = file_path.read_bytes()
    
    return results

# Usage
if wait_for_job_completion(submitter, cluster_id):
    results = retrieve_results(job_dir)
    
    # Process results
    for file_path, content in results["output_files"].items():
        print(f"Retrieved {file_path}: {len(content)} bytes")
```

## Best Practices

### 1. Use Absolute Paths

Always use absolute paths in sandboxes:

```python
# ✅ Good
backend.upload_files([("/workspace/data.csv", content)])

# ❌ Bad - relative paths may be misinterpreted
backend.upload_files([("data.csv", content)])
```

### 2. Create Parent Directories

Ensure parent directories exist before uploading:

```python
# Create directory structure
backend.execute("mkdir -p /workspace/data/inputs")

# Now upload to subdirectory
backend.upload_files([
    ("/workspace/data/inputs/file.csv", content),
])
```

### 3. Handle Partial Failures

Always check for errors in batch operations:

```python
responses = backend.upload_files(files)

for i, resp in enumerate(responses):
    if resp.error:
        print(f"Failed to upload {files[i][0]}: {resp.error}")
        # Implement retry or alternative strategy
```

### 4. Clean Up Temporary Files

Remove temporary files after transfer:

```python
# Upload inputs
backend.upload_files([("/workspace/temp_input.csv", data)])

# Process
backend.execute("python process.py /workspace/temp_input.csv")

# Download results
responses = backend.download_files(["/workspace/output.json"])

# Clean up
backend.execute("rm /workspace/temp_input.csv /workspace/output.json")
```

### 5. Use Appropriate Transfer Method

Choose based on file characteristics:

| File Type | Size | Method | Notes |
|-----------|------|--------|-------|
| Config files | < 1 KB | Embed in command | Base64 encode, include in script |
| Data files | < 100 MB | Direct transfer | Use upload_files/download_files |
| Large datasets | > 100 MB | Network storage | Mount shared filesystem or use URLs |
| Many small files | Any | Archive first | tar/zip before transfer |

### 6. HTCondor-Specific Best Practices

**a) Use shared filesystem when available:**
```bash
# CERN AFS, EOS
transfer_input_files = 
arguments = -c "cp /eos/project/data/input.csv /workspace/"
```

**b) Compress large outputs:**
```bash
arguments = -c "
    python process.py
    tar czf results.tar.gz results/
"
transfer_output_files = results.tar.gz
```

**c) Set appropriate resource limits:**
```bash
# Ensure enough disk for transfers
request_disk = 10GB

# Prevent transfers of unnecessary files
transfer_output_files = results/
+WantDiskTransfer = False  # Don't transfer working files
```

## Examples

### Example 1: Process Dataset in Docker Sandbox

```python
from chatlas_agents.sandbox import create_docker_sandbox
from deepagents import create_deep_agent
from pathlib import Path

# Read input data
input_csv = Path("data/input.csv").read_bytes()

with create_docker_sandbox(image="python:3.13-slim") as backend:
    # Upload input data
    print("Uploading input data...")
    responses = backend.upload_files([
        ("/workspace/input.csv", input_csv),
    ])
    
    if responses[0].error:
        raise RuntimeError(f"Upload failed: {responses[0].error}")
    
    # Create agent with sandbox backend
    agent = create_deep_agent(
        backend=backend,
        system_prompt="Process the CSV file in /workspace/input.csv and save results to /workspace/output.json",
    )
    
    # Run agent
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Process the input CSV"}]
    })
    
    # Download results
    print("Downloading results...")
    responses = backend.download_files([
        "/workspace/output.json",
    ])
    
    if responses[0].error:
        raise RuntimeError(f"Download failed: {responses[0].error}")
    
    # Save results
    Path("data/output.json").write_bytes(responses[0].content)
    print("Processing complete!")
```

### Example 2: HTCondor Batch Job with File Transfers

```python
from chatlas_agents.htcondor import HTCondorJobSubmitter
from pathlib import Path
import shutil

# Setup job directory on submit node
job_name = "atlas_analysis_job"
submit_dir = Path("/afs/cern.ch/user/u/username/jobs") / job_name
submit_dir.mkdir(parents=True, exist_ok=True)

# Copy input files to submit directory
shutil.copy("data/events.root", submit_dir / "events.root")
shutil.copy("config/analysis.yaml", submit_dir / "analysis.yaml")

# Create HTCondor submitter
submitter = HTCondorJobSubmitter(
    docker_image="atlas/athanalysis:latest",
    output_dir=submit_dir,
)

# Generate submit file with file transfer
submit_file = submitter.generate_submit_file(
    job_name=job_name,
    prompt="Run ATLAS analysis on events.root",
    should_transfer_files="YES",
    when_to_transfer_output="ON_EXIT",
    transfer_input_files="events.root,analysis.yaml",
    transfer_output_files="results/",
    initialdir="/workspace",
    request_cpus=4,
    request_memory="8GB",
    request_disk="20GB",
)

print(f"Submit file created: {submit_file}")
print(f"Submit with: condor_submit {submit_file}")

# After job completes, results will be in:
# submit_dir/results/
```

### Example 3: Agent-Managed File Transfers

```python
from langchain_core.tools import tool
from chatlas_agents.sandbox import create_apptainer_sandbox
from deepagents import create_deep_agent
from pathlib import Path

# Create sandbox
with create_apptainer_sandbox(image="docker://python:3.13-slim") as backend:
    
    # Define file transfer tools for agent
    @tool
    def upload_host_file(host_path: str, sandbox_path: str) -> str:
        """Upload a file from host filesystem to sandbox.
        
        Args:
            host_path: Path to file on host (e.g., 'data/input.csv')
            sandbox_path: Destination in sandbox (e.g., '/workspace/input.csv')
        """
        try:
            content = Path(host_path).read_bytes()
            responses = backend.upload_files([(sandbox_path, content)])
            
            if responses[0].error:
                return f"Error uploading {host_path}: {responses[0].error}"
            
            return f"Successfully uploaded {host_path} to {sandbox_path}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    @tool
    def download_sandbox_file(sandbox_path: str, host_path: str) -> str:
        """Download a file from sandbox to host filesystem.
        
        Args:
            sandbox_path: Source file in sandbox (e.g., '/workspace/output.csv')
            host_path: Destination on host (e.g., 'results/output.csv')
        """
        try:
            responses = backend.download_files([sandbox_path])
            
            if responses[0].error:
                return f"Error downloading {sandbox_path}: {responses[0].error}"
            
            # Save to host
            dest = Path(host_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(responses[0].content)
            
            return f"Successfully downloaded {sandbox_path} to {host_path}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Create agent with file transfer tools
    agent = create_deep_agent(
        backend=backend,
        tools=[upload_host_file, download_sandbox_file],
        system_prompt="""You can upload and download files between host and sandbox.
        
        Available tools:
        - upload_host_file: Transfer files from host to sandbox
        - download_sandbox_file: Transfer files from sandbox to host
        
        Use these when you need to work with files from the host system.
        """,
    )
    
    # Agent can now manage file transfers
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Upload data/experiment_data.csv to the sandbox, process it, and download the results"
        }]
    })
```

## Troubleshooting

### Upload Fails with "permission_denied"

**Cause:** Container user doesn't have write permission to destination directory.

**Solution:**
```python
# Create directory with proper permissions first
backend.execute("mkdir -p /workspace/data && chmod 777 /workspace/data")
backend.upload_files([("/workspace/data/file.txt", content)])
```

### Download Fails with "file_not_found"

**Cause:** File doesn't exist or path is wrong.

**Solution:**
```python
# List directory first to verify file exists
result = backend.execute("ls -la /workspace/")
print(result.output)

# Then download with correct path
```

### HTCondor Job Doesn't Transfer Files

**Cause:** `transfer_input_files` paths are relative to submit directory, not absolute.

**Solution:**
```bash
# Ensure input files are in submit directory
# or use absolute paths with shared filesystem

# Submit directory
initialdir = /path/to/submit/dir

# Files relative to initialdir
transfer_input_files = data.csv, config.yaml

# OR use shared filesystem
transfer_input_files = 
arguments = -c "cp /eos/shared/data.csv /workspace/"
```

### Large File Transfers Timeout

**Cause:** Network or I/O bottleneck.

**Solution:**
```python
# For large files, use compression
import gzip

# Compress before upload
compressed = gzip.compress(large_data)
backend.upload_files([("/workspace/data.gz", compressed)])

# Decompress in sandbox
backend.execute("gunzip /workspace/data.gz")

# Or use network storage instead
backend.execute("cp /eos/atlas/large_file.root /workspace/")
```

## Summary

Key takeaways for file transfer handling:

1. **Use the standardized API**: `upload_files()` and `download_files()` work across all sandbox backends
2. **Check for errors**: Always inspect `FileUploadResponse` and `FileDownloadResponse` error fields
3. **Handle batch operations**: Process successes and failures separately
4. **HTCondor requires staging**: Files must be on submit node, then HTCondor transfers to execute node
5. **Choose appropriate method**: Direct transfer for small files, network storage for large datasets
6. **Clean up**: Remove temporary files to avoid filling up sandbox storage

For more information:
- [SANDBOX.md](SANDBOX.md) - Sandbox backend documentation
- [htcondor.py](chatlas_agents/htcondor.py) - HTCondor integration implementation
- [sandbox.py](chatlas_agents/sandbox.py) - Docker and Apptainer backend implementations
