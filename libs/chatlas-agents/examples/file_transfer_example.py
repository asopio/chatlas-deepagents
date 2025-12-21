"""Example demonstrating file transfers between host and sandbox.

This example shows various patterns for uploading and downloading files
when working with Docker or Apptainer sandboxes.

For comprehensive documentation, see FILE_TRANSFER_GUIDE.md
"""

from pathlib import Path
import tempfile
import json

from chatlas_agents.sandbox import create_docker_sandbox, create_apptainer_sandbox
from deepagents import create_deep_agent


def example_basic_file_transfer():
    """Basic example: upload input, process, download output."""
    print("\n=== Example 1: Basic File Transfer ===")
    
    # Create sample input data
    input_data = "apple,5\nbanana,3\norange,7\n"
    
    with create_docker_sandbox(image="python:3.13-slim") as backend:
        # Upload input file
        print("Uploading input.csv...")
        responses = backend.upload_files([
            ("/workspace/input.csv", input_data.encode()),
        ])
        
        if responses[0].error:
            print(f"Upload failed: {responses[0].error}")
            return
        
        print("✓ Uploaded successfully")
        
        # Process the file in sandbox
        print("Processing data...")
        result = backend.execute("""
            python3 -c "
import csv

# Read input
total = 0
with open('/workspace/input.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        total += int(row[1])

# Write output
with open('/workspace/output.txt', 'w') as f:
    f.write(f'Total: {total}\\n')

print('Processing complete')
"
        """)
        
        print(result.output)
        
        # Download result
        print("Downloading output.txt...")
        responses = backend.download_files(["/workspace/output.txt"])
        
        if responses[0].error:
            print(f"Download failed: {responses[0].error}")
            return
        
        output = responses[0].content.decode()
        print(f"✓ Downloaded successfully")
        print(f"Result: {output.strip()}")


def example_batch_transfer():
    """Example: batch upload/download with error handling."""
    print("\n=== Example 2: Batch File Transfer ===")
    
    # Create multiple files
    files = [
        ("/workspace/file1.txt", b"Content 1"),
        ("/workspace/file2.txt", b"Content 2"),
        ("/workspace/data/file3.txt", b"Content 3"),  # Requires directory
    ]
    
    with create_docker_sandbox() as backend:
        # Create directory first
        backend.execute("mkdir -p /workspace/data")
        
        # Batch upload
        print(f"Uploading {len(files)} files...")
        responses = backend.upload_files(files)
        
        # Check results
        succeeded = [r for r in responses if r.error is None]
        failed = [r for r in responses if r.error is not None]
        
        print(f"✓ Uploaded {len(succeeded)} files successfully")
        if failed:
            for resp in failed:
                print(f"✗ Failed to upload {resp.path}: {resp.error}")
        
        # Batch download
        print("Downloading all files...")
        paths = [f[0] for f in files]
        responses = backend.download_files(paths)
        
        for resp in responses:
            if resp.error:
                print(f"✗ Failed to download {resp.path}: {resp.error}")
            else:
                print(f"✓ Downloaded {resp.path}: {len(resp.content)} bytes")


def example_agent_with_file_tools():
    """Example: agent with file transfer tools."""
    print("\n=== Example 3: Agent with File Transfer Tools ===")
    
    from langchain_core.tools import tool
    
    # Create temp directory for host files
    temp_dir = Path(tempfile.mkdtemp())
    (temp_dir / "input.json").write_text('{"value": 42}')
    
    with create_docker_sandbox() as backend:
        # Define file transfer tools for agent
        @tool
        def upload_file(host_path: str, sandbox_path: str) -> str:
            """Upload a file from host to sandbox.
            
            Args:
                host_path: Path to file on host
                sandbox_path: Destination path in sandbox
            """
            try:
                content = Path(host_path).read_bytes()
                responses = backend.upload_files([(sandbox_path, content)])
                
                if responses[0].error:
                    return f"Error: {responses[0].error}"
                
                return f"Uploaded {host_path} to {sandbox_path}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        @tool
        def download_file(sandbox_path: str, host_path: str) -> str:
            """Download a file from sandbox to host.
            
            Args:
                sandbox_path: Source path in sandbox
                host_path: Destination path on host
            """
            try:
                responses = backend.download_files([sandbox_path])
                
                if responses[0].error:
                    return f"Error: {responses[0].error}"
                
                Path(host_path).write_bytes(responses[0].content)
                return f"Downloaded {sandbox_path} to {host_path}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Create agent with file tools
        agent = create_deep_agent(
            backend=backend,
            tools=[upload_file, download_file],
            system_prompt="""You can transfer files between host and sandbox.
            
            Use upload_file to transfer files from host to sandbox.
            Use download_file to transfer files from sandbox to host.
            """,
        )
        
        # Run agent
        print("Running agent...")
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"""Upload {temp_dir}/input.json to /workspace/input.json, 
                read it, double the value, write to /workspace/output.json, 
                and download it to {temp_dir}/output.json"""
            }]
        })
        
        # Check results
        output_file = temp_dir / "output.json"
        if output_file.exists():
            output_data = json.loads(output_file.read_text())
            print(f"✓ Agent completed task")
            print(f"  Input value: 42")
            print(f"  Output value: {output_data.get('value', 'N/A')}")
        else:
            print("✗ Agent failed to complete task")


def example_apptainer_transfer():
    """Example: file transfer with Apptainer sandbox."""
    print("\n=== Example 4: Apptainer File Transfer ===")
    
    # Note: This requires Apptainer to be installed
    try:
        with create_apptainer_sandbox(image="docker://python:3.13-slim") as backend:
            # Same API as Docker
            print("Uploading to Apptainer sandbox...")
            responses = backend.upload_files([
                ("/workspace/test.txt", b"Hello from Apptainer!"),
            ])
            
            if responses[0].error:
                print(f"Upload failed: {responses[0].error}")
                return
            
            print("✓ Uploaded successfully")
            
            # Verify file exists
            result = backend.execute("cat /workspace/test.txt")
            print(f"File content: {result.output.strip()}")
            
            # Download back
            print("Downloading from Apptainer sandbox...")
            responses = backend.download_files(["/workspace/test.txt"])
            
            if responses[0].error:
                print(f"Download failed: {responses[0].error}")
                return
            
            print(f"✓ Downloaded: {responses[0].content.decode()}")
            
    except RuntimeError as e:
        print(f"Apptainer not available: {e}")
        print("Skipping Apptainer example")


def example_error_handling():
    """Example: handling file transfer errors."""
    print("\n=== Example 5: Error Handling ===")
    
    with create_docker_sandbox() as backend:
        # Try to download non-existent file
        print("Attempting to download non-existent file...")
        responses = backend.download_files(["/workspace/does_not_exist.txt"])
        
        if responses[0].error:
            print(f"✓ Error caught: {responses[0].error}")
            print("  (This is expected)")
        
        # Try to upload to non-existent directory
        print("\nAttempting upload without creating directory first...")
        responses = backend.upload_files([
            ("/workspace/deep/nested/file.txt", b"content"),
        ])
        
        # This may succeed or fail depending on implementation
        if responses[0].error:
            print(f"Upload failed: {responses[0].error}")
            print("Creating directory and retrying...")
            backend.execute("mkdir -p /workspace/deep/nested")
            responses = backend.upload_files([
                ("/workspace/deep/nested/file.txt", b"content"),
            ])
            if responses[0].error is None:
                print("✓ Upload succeeded after creating directory")
        else:
            print("✓ Upload succeeded (implementation creates directories)")


def main():
    """Run all examples."""
    print("=" * 60)
    print("File Transfer Examples")
    print("=" * 60)
    print("\nThese examples demonstrate file upload/download patterns")
    print("for sandbox backends (Docker, Apptainer).")
    print("\nFor comprehensive documentation, see FILE_TRANSFER_GUIDE.md")
    print("=" * 60)
    
    # Run examples
    example_basic_file_transfer()
    example_batch_transfer()
    example_agent_with_file_tools()
    example_apptainer_transfer()
    example_error_handling()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
