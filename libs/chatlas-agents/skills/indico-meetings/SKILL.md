---
name: indico-meetings
description: Fetch ATLAS meeting agendas, slides, and abstracts from CERN Indico
---

# Indico Meetings Skill

Query CERN Indico for ATLAS meeting information, download slides, and extract abstracts. Based on [indicomb](https://gitlab.cern.ch/indicomb/indicomb).

## When to Use

- Get upcoming ATLAS meetings (today/this week)
- Find meeting agendas for date ranges
- Download slides from conferences/workshops
- Extract talk abstracts and speaker information

## Prerequisites

**No special setup** - uses Python standard library only.

**Authentication (optional):**
- Public meetings: No auth needed
- Protected meetings: Get API token from https://indico.cern.ch/user/tokens/
- Set via: `export INDICO_API_TOKEN="indp_XXXXXXXX"` or `--api-token` flag
- Alternative: Create `~/.indico-secret-key` with API_KEY and SECRET_KEY on separate lines

## Commands

### Get Upcoming Meetings

```bash
# This week's meetings
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py upcoming --days 7

# Today only
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py upcoming --today
```

Returns markdown with meeting titles, times, locations, URLs grouped by date.

### Query Category

```bash
# Meetings in date range
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py category 12592 \
    --from 2024-01-01 --to 2024-01-31

# Include talk/contribution info
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py category 12592 \
    --from 2024-01-01 --to 2024-01-31 --slides
```

### Download Event Materials

```bash
# Download all slides and materials
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py download <event_id> \
    --output ./downloads

# Slides only (skip abstracts)
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py download <event_id> \
    --output ./downloads --slides-only
```

Creates:
- `event_summary.md` - Event info with abstracts
- `contrib_XXX_*.pdf` - Talk slides

## Common Category IDs

| Category | ID |
|----------|-----|
| ATLAS | 12592 |
| ATLAS Physics | 490 |
| ATLAS Upgrade | 3286 |

**Find IDs:** Check Indico URL
- Category: `https://indico.cern.ch/category/XXXXX` → use XXXXX
- Event: `https://indico.cern.ch/event/YYYYYYY` → use YYYYYYY

## Common Workflows

**Daily briefing:**
```bash
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py upcoming --today
```

**Download conference materials:**
```bash
# Get event ID from URL, then:
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py download 1234567 \
    --output ./atlas_week_2024
```

**Search past meetings:**
```bash
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py category 12592 \
    --from 2023-01-01 --to 2023-12-31 --slides | grep -i "higgs"
```

## Output Format

All commands return markdown text:
- Human-readable and AI-parseable
- Can be saved as `.md` files
- Includes clickable Indico URLs

## Troubleshooting

| Error | Solution |
|-------|----------|
| HTTPError 401 | Add `--api-token` or set `INDICO_API_TOKEN` |
| HTTPError 403 | Meeting is restricted, check access |
| Event not found | Verify event ID from Indico URL |
| Download timeout | Large files, retry or download manually |
| No meetings found | Expand date range or check category ID |

## Advanced Usage

**Different Indico instance:**
```bash
export INDICO_BASE_URL="https://your-indico.org"
```

**Automate with cron:**
```bash
0 8 * * * python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py upcoming --today > ~/meetings.md
```

**Process output:**
```bash
# Count meetings
... | grep "^##" | wc -l

# Filter topics
... | grep -i "higgs"
```

## Notes

- Date format: YYYY-MM-DD
- Categories contain events; events are individual meetings
- Not all talks have downloadable materials
- Be respectful of Indico API limits

## Resources

- Indico: https://indico.cern.ch/
- API docs: https://indico.docs.cern.ch/http-api/
- API tokens: https://indico.cern.ch/user/tokens/

## Agent Examples

```bash
# "What meetings are today?"
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py upcoming --today

# "Show ATLAS Physics meetings from last month"
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py category 490 \
    --from 2024-11-01 --to 2024-11-30

# "Download slides from event 1234567"
python3 [SKILLS_DIR]/indico-meetings/indico_meetings.py download 1234567
```
