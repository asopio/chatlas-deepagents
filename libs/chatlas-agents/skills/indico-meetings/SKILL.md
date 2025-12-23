---
name: indico-meetings
description: Fetch ATLAS meeting agendas, slides, and abstracts from CERN Indico
---

# Indico Meetings Skill

Query CERN Indico for ATLAS meeting information, download slides, and extract abstracts. Based on the [indicomb project](https://gitlab.cern.ch/indicomb/indicomb).

## When to Use This Skill

Use this skill when you need to:
- Get a summary of upcoming ATLAS meetings (today/this week)
- Find meeting agendas and schedules for specific date ranges
- Download slides and presentation materials from conferences/workshops
- Extract talk abstracts and speaker information
- Track ATLAS meeting history or plan for future meetings

## Prerequisites

**No special setup required on LXPlus** - this skill uses only Python standard library (`urllib`).

**Authentication (optional but recommended):**
- For public meetings: No authentication needed
- For protected meetings: Need API token or key
  - **Recommended**: Get API token from https://indico.cern.ch/user/tokens/
  - **Legacy**: Get API key/secret from https://indico.cern.ch/user/api/

**Authentication methods (in order of preference):**
1. Environment variable: `export INDICO_API_TOKEN="indp_XXXXXXXX"`
2. Command-line argument: `--api-token indp_XXXXXXXX`
3. Secret file: Create `~/.indico-secret-key` with API_KEY on line 1, SECRET_KEY on line 2

## Core Commands

### Get Upcoming Meetings

**Get this week's ATLAS meetings:**
```bash
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py upcoming --category 12592 --days 7
```

**Get today's meetings only:**
```bash
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py upcoming --today
```

**Output:** Markdown-formatted list grouped by date with meeting titles, times, locations, and URLs.

### Query Meeting Category

**Get all meetings in a date range:**
```bash
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py category 12592 \
    --from 2024-01-01 --to 2024-01-31
```

**Include contribution/talk information:**
```bash
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py category 12592 \
    --from 2024-01-01 --to 2024-01-31 --slides
```

**Output:** Markdown-formatted meeting list with optional contribution details.

### Download Event Materials

**Download all materials from an event:**
```bash
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py download 1234567 \
    --output ./my_downloads
```

**Download only slides (skip abstracts):**
```bash
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py download 1234567 \
    --output ./downloads --slides-only
```

**Output:** 
- Downloads PDF slides and materials to specified directory
- Creates `event_summary.md` with all talk abstracts and speaker info
- Returns markdown summary of downloaded files

## Common Workflows

### Workflow 1: Daily Meeting Summary

Get today's ATLAS meetings for a quick daily briefing:

```bash
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py upcoming --today
```

### Workflow 2: Weekly Meeting Plan

See what's coming up this week:

```bash
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py upcoming --days 7
```

### Workflow 3: Conference Material Download

Download all slides from ATLAS Week or similar event:

```bash
# Step 1: Find the event ID from the Indico URL
# Example: https://indico.cern.ch/event/1234567/ -> event ID is 1234567

# Step 2: Download everything
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py download 1234567 \
    --output ./atlas_week_2024
```

This creates:
- `./atlas_week_2024/<event_name>/event_summary.md` - Complete event summary with abstracts
- `./atlas_week_2024/<event_name>/contrib_XXX_*.pdf` - Individual talk slides

### Workflow 4: Research Past Meetings

Find meetings on a specific topic in a date range:

```bash
# Get all meetings from a category with contribution details
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py category 12592 \
    --from 2023-01-01 --to 2023-12-31 --slides
```

Then grep the output for specific topics or speakers.

## Common Indico Category IDs

| Category | ID |
|----------|-----|
| ATLAS | 12592 |
| ATLAS Physics | 490 |
| ATLAS Upgrade | 3286 |
| Combined Performance | (varies) |

**To find a category ID:**
1. Navigate to the category on Indico
2. Look at the URL: `https://indico.cern.ch/category/XXXXX`
3. Use `XXXXX` as the category ID

**To find an event ID:**
1. Navigate to the event on Indico  
2. Look at the URL: `https://indico.cern.ch/event/YYYYYYY`
3. Use `YYYYYYY` as the event ID

## Output Format

All commands return **markdown-formatted** text that is:
- ✅ Easy to read for humans
- ✅ Easy to parse for AI agents
- ✅ Can be saved directly as `.md` files
- ✅ Includes clickable URLs to Indico

The `download` command additionally saves:
- 📄 `event_summary.md` - Complete event information
- 📎 PDF slides and other attachments
- 🔗 Relative links in markdown for easy navigation

## Troubleshooting

**Error: "HTTPError 401"**
- Cause: Authentication required but not provided
- Solution: Add `--api-token` or set `INDICO_API_TOKEN` environment variable

**Error: "HTTPError 403"**
- Cause: You don't have access to this meeting/category
- Solution: Check if meeting is restricted, contact organizers

**Error: "Event XXXXX not found"**
- Cause: Invalid event ID or event doesn't exist
- Solution: Double-check the event ID from the Indico URL

**Downloads fail with timeout:**
- Cause: Large files or slow connection
- Solution: Script has 60s timeout for downloads, try again or download manually

**No meetings found:**
- Cause: No meetings in the specified date range
- Solution: Try expanding date range or checking category ID

## Advanced Usage

### Use with Different Indico Instances

If using a non-CERN Indico instance:

```bash
export INDICO_BASE_URL="https://your-indico-instance.org"
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py upcoming
```

### Automation via cron

Set up daily meeting summaries:

```bash
# Add to crontab
0 8 * * * python3 ~/.deepagents/agent/skills/indico-meetings/indico_meetings.py upcoming --today > ~/today_meetings.md
```

### Pipe to Other Tools

```bash
# Count meetings this month
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py category 12592 \
    --from 2024-12-01 --to 2024-12-31 | grep "^##" | wc -l

# Find meetings with "Higgs" in the title
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py category 490 \
    --from 2024-01-01 | grep -i "higgs"
```

## Notes

- **Public vs. Protected:** Some meetings require authentication even to view
- **API Limits:** Be respectful of Indico servers, don't hammer with requests
- **Date Formats:** Always use YYYY-MM-DD format for dates
- **Category vs. Event:** Categories contain multiple events; events are individual meetings
- **Materials:** Not all contributions have downloadable materials
- **Markdown Output:** All text output is markdown-formatted for easy reading and parsing

## Integration with Other Skills

This skill works well with:
- **ChATLAS MCP**: Search meeting content in ChATLAS documentation
- **File operations**: Save outputs to files for later reference
- **Text processing**: Parse markdown output for specific information

## Resources

- **Indico at CERN:** https://indico.cern.ch/
- **Indicomb Project:** https://gitlab.cern.ch/indicomb/indicomb
- **API Documentation:** https://indico.docs.cern.ch/http-api/
- **API Tokens:** https://indico.cern.ch/user/tokens/
- **Legacy API Keys:** https://indico.cern.ch/user/api/

## Examples for AI Agents

When an agent needs to:

**"What meetings are happening today?"**
```bash
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py upcoming --today
```

**"Show me ATLAS Physics meetings from last month"**
```bash
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py category 490 \
    --from 2024-11-01 --to 2024-11-30
```

**"Download all slides from event 1234567"**
```bash
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py download 1234567 \
    --output ./event_materials
```

The markdown output can be:
1. Displayed directly to the user
2. Saved to a file for reference
3. Parsed for specific information
4. Used as input for further queries
