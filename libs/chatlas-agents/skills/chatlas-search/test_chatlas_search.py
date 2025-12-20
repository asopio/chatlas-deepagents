#!/usr/bin/env python3
"""Test for ChATLAS search skill.

This test queries the actual ChATLAS MCP server to ensure the skill works correctly.
"""

import os
import subprocess
import sys
from pathlib import Path


def test_chatlas_search_skill():
    """Test the chatlas-search skill against the actual MCP server."""
    print("=" * 70)
    print("Testing ChATLAS Search Skill")
    print("=" * 70)
    
    # Set up environment variables
    os.environ["CHATLAS_MCP_URL"] = os.environ.get(
        "CHATLAS_MCP_URL", "https://chatlas-mcp.app.cern.ch/mcp"
    )
    os.environ["CHATLAS_MCP_TIMEOUT"] = os.environ.get("CHATLAS_MCP_TIMEOUT", "60")
    
    print(f"\nConfiguration:")
    print(f"  MCP URL: {os.environ['CHATLAS_MCP_URL']}")
    print(f"  Timeout: {os.environ['CHATLAS_MCP_TIMEOUT']}s")
    
    # Find the skill script
    skill_dir = Path(__file__).parent
    skill_script = skill_dir / "chatlas_search.py"
    
    if not skill_script.exists():
        print(f"\n❌ ERROR: Skill script not found at {skill_script}")
        return False
    
    print(f"\n  Skill script: {skill_script}")
    
    # Test 1: Basic query to twiki_prod
    print("\n" + "=" * 70)
    print("Test 1: Query twiki_prod for 'ATLAS detector'")
    print("=" * 70)
    
    try:
        # Use python3 explicitly to ensure subprocess uses same interpreter
        python_exec = "python3"
        result = subprocess.run(
            [
                python_exec,
                str(skill_script),
                "ATLAS detector",
                "--vectorstore", "twiki_prod",
                "--ndocs", "3"
            ],
            capture_output=True,
            text=True,
            timeout=90,  # Allow time for MCP connection
            env=os.environ.copy()  # Pass environment variables
        )
        
        print(f"\nReturn code: {result.returncode}")
        
        if result.returncode != 0:
            print(f"❌ FAILED: Script exited with code {result.returncode}")
            if result.stderr:
                print(f"\nStderr:\n{result.stderr}")
            return False
        
        if result.stdout:
            print(f"\n✅ SUCCESS: Received output from MCP server")
            print(f"\nOutput preview (first 500 chars):")
            print("-" * 70)
            print(result.stdout[:500])
            if len(result.stdout) > 500:
                print(f"\n... ({len(result.stdout) - 500} more characters)")
            print("-" * 70)
            
            # Check for error messages
            if "Error:" in result.stdout:
                print(f"\n⚠️  WARNING: Output contains error message")
                print(result.stdout)
                return False
        else:
            print(f"❌ FAILED: No output received")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ FAILED: Script timed out after 90 seconds")
        return False
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {str(e)}")
        return False
    
    # Test 2: Query cds_v1
    print("\n" + "=" * 70)
    print("Test 2: Query cds_v1 for 'Higgs boson'")
    print("=" * 70)
    
    try:
        result = subprocess.run(
            [
                "python3",
                str(skill_script),
                "Higgs boson",
                "--vectorstore", "cds_v1",
                "--ndocs", "2"
            ],
            capture_output=True,
            text=True,
            timeout=90,
            env=os.environ.copy()
        )
        
        print(f"\nReturn code: {result.returncode}")
        
        if result.returncode != 0:
            print(f"❌ FAILED: Script exited with code {result.returncode}")
            if result.stderr:
                print(f"\nStderr:\n{result.stderr}")
            return False
        
        if result.stdout:
            print(f"✅ SUCCESS: Received output from MCP server")
            print(f"\nOutput preview (first 300 chars):")
            print("-" * 70)
            print(result.stdout[:300])
            if len(result.stdout) > 300:
                print(f"\n... ({len(result.stdout) - 300} more characters)")
            print("-" * 70)
            
            if "Error:" in result.stdout:
                print(f"\n⚠️  WARNING: Output contains error message")
                print(result.stdout)
                return False
        else:
            print(f"❌ FAILED: No output received")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ FAILED: Script timed out after 90 seconds")
        return False
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {str(e)}")
        return False
    
    return True


def main():
    """Run the test."""
    print("\n" + "=" * 70)
    print("ChATLAS Search Skill - Integration Test")
    print("=" * 70)
    
    success = test_chatlas_search_skill()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ ALL TESTS PASSED")
        print("   The chatlas-search skill is working correctly!")
        print("   - Can connect to MCP server")
        print("   - Can query different vectorstores")
        print("   - Returns formatted results")
    else:
        print("❌ TESTS FAILED")
        print("   See error messages above for details")
    print("=" * 70)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
