# OpenAI Rate Limit Error Fixes

## Problem
The evaluation script was hitting OpenAI's rate limits for GPT-4o with the error:
```
openai.RateLimitError: Error code: 429 - Rate limit reached for gpt-4o in organization ... on tokens per min (TPM): Limit 30000, Used 30000, Requested 452.
```

## Root Cause
- The script was making 180 requests per minute, which was too aggressive
- OpenAI's GPT-4o has a limit of 30,000 tokens per minute (TPM)
- Each request uses approximately 400-600 tokens
- No retry mechanism was in place for rate limit errors

## Solutions Implemented

### 1. Reduced Request Rate
- Changed from 180 to 30 requests per minute initially
- Added intelligent rate limiting based on both requests and estimated tokens

### 2. Exponential Backoff Retry Logic
- Added `make_openai_request_with_retry()` function
- Implements exponential backoff with jitter
- Parses wait time from OpenAI error messages when available
- Maximum 5 retry attempts per request

### 3. Token-Based Rate Limiting
- Estimates ~500 tokens per request (conservative)
- Limits to 25,000 tokens per minute (below 30k limit)
- Uses the more restrictive of request-based or token-based limits

### 4. Progress Saving and Resume
- Saves progress to `{model_name}_geval_progress.json`
- Can resume from where it left off if interrupted
- Automatically cleans up progress file on successful completion

## Key Changes Made

### New Imports
```python
import time
import random
import re
from openai import AsyncOpenAI, RateLimitError
```

### New Functions
- `make_openai_request_with_retry()`: Handles retries with exponential backoff
- Enhanced `score_summaries()`: Intelligent rate limiting
- Enhanced `evaluate_geval()`: Progress saving and resume capability

### Configuration Changes
- Default requests per minute: 30 (down from 180)
- Token limit: 25,000 TPM (conservative buffer)
- Max retries: 5 per request
- Base delay: 1 second with exponential backoff

## Usage

### Running the Fixed Script
```bash
cd code/evaluation
python eval.py
```

### Testing the Fixes
```bash
cd code/evaluation
python test_rate_limit_fix.py
```

### Monitoring Progress
- Progress files are saved in `{domain}/{split}/evaluation/{model}/`
- Look for `*_geval_progress.json` files
- These are automatically cleaned up on successful completion

## Benefits

1. **Reliability**: Automatic retry on rate limit errors
2. **Efficiency**: Intelligent rate limiting prevents wasted requests
3. **Resumability**: Can continue from where it left off if interrupted
4. **Monitoring**: Clear progress indicators and error messages
5. **Conservative**: Uses safe limits with buffers to avoid hitting limits

## Troubleshooting

### If you still hit rate limits:
1. Reduce `requests_per_minute` further (try 20 or 15)
2. Reduce `tokens_per_minute` (try 20,000 or 15,000)
3. Check your OpenAI usage dashboard for current limits

### If requests are too slow:
1. Gradually increase `requests_per_minute` (try 40, 50, etc.)
2. Monitor for rate limit errors
3. Adjust based on your specific OpenAI plan limits

### If evaluation gets interrupted:
1. Simply restart the script - it will resume from saved progress
2. Check for `*_geval_progress.json` files to see what was completed
3. Delete progress files manually if you want to start fresh

## OpenAI Plan Considerations

- **Free tier**: Very low limits, use 5-10 requests per minute
- **Pay-as-you-go**: 30k TPM for GPT-4o, current settings should work
- **Higher tiers**: Can increase limits accordingly

Check your specific limits at: https://platform.openai.com/account/rate-limits
