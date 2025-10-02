#!/usr/bin/env python3
"""
Test script to verify the rate limit fixes work correctly.
"""
import asyncio
import os
import sys
from dotenv import load_dotenv
load_dotenv()

# Add the current directory to path so we can import eval
sys.path.append(os.path.dirname(__file__))

from eval import make_openai_request_with_retry, score_summaries

async def test_retry_mechanism():
    """Test the retry mechanism with a simple request."""
    print("Testing retry mechanism...")
    
    # Simple test message
    test_message = [{"role": "system", "content": "Rate this summary on a scale of 1-5: This is a test summary."}]
    
    try:
        response = await make_openai_request_with_retry(test_message)
        print("✓ Retry mechanism test passed")
        return True
    except Exception as e:
        print(f"✗ Retry mechanism test failed: {e}")
        return False

async def test_rate_limiting():
    """Test the rate limiting with multiple requests."""
    print("Testing rate limiting...")
    
    # Create a few test messages
    test_messages = [
        [{"role": "system", "content": f"Rate this summary on a scale of 1-5: Test summary {i}."}]
        for i in range(3)
    ]
    
    try:
        responses = await score_summaries(test_messages, requests_per_minute=10)
        print(f"✓ Rate limiting test passed. Got {len(responses)} responses")
        return True
    except Exception as e:
        print(f"✗ Rate limiting test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("Running rate limit fix tests...\n")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("✗ OPENAI_API_KEY not found in environment variables")
        return
    
    tests = [
        test_retry_mechanism,
        test_rate_limiting
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
        print()
    
    passed = sum(results)
    total = len(results)
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All tests passed! The rate limit fixes should work correctly.")
    else:
        print("✗ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    asyncio.run(main())
