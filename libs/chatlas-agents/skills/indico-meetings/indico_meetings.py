#!/usr/bin/env python3
"""CERN Indico Meeting Operations.

Script to interact with CERN Indico for fetching meeting agendas, slides, and abstracts.
Based on the indicomb project (https://gitlab.cern.ch/indicomb/indicomb).

This script provides AI-agent friendly markdown output for ATLAS meeting information.
"""

import argparse
import hashlib
import hmac
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import urllib.request
        import urllib.parse
        import urllib.error
    except ImportError:
        return "Error: Failed to import required urllib modules"
    return None


def build_indico_request(
    path: str,
    params: dict,
    api_token: Optional[str] = None,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> str:
    """Build an authenticated Indico API request URL.
    
    Based on indicomb's build_indico_request function.
    
    Parameters
    ----------
    path : str
        API endpoint path (e.g., '/export/categ/12592.json')
    params : dict
        Query parameters
    api_token : str, optional
        Bearer token for authentication
    api_key : str, optional
        API key (legacy authentication)
    secret_key : str, optional
        Secret key for HMAC signing (legacy authentication)
        
    Returns
    -------
    str
        Full URL with authentication parameters
    """
    items = list(params.items())
    
    # Legacy authentication with API key/secret
    if api_key:
        items.append(('apikey', api_key))
    if secret_key:
        items.append(('timestamp', str(int(time.time()))))
        items = sorted(items, key=lambda x: x[0].lower())
        url = '%s?%s' % (path, urllib.parse.urlencode(items))
        signature = hmac.new(secret_key.encode('utf-8'), url.encode('utf-8'), hashlib.sha1).hexdigest()
        items.append(('signature', signature))
    
    if not items:
        return path
    return '%s?%s' % (path, urllib.parse.urlencode(items))


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename or directory name.
    
    Parameters
    ----------
    name : str
        Original name to sanitize
        
    Returns
    -------
    str
        Sanitized name safe for filesystem use
    """
    # Remove non-alphanumeric characters except spaces, hyphens, underscores
    safe_name = re.sub(r'[^\w\s-]', '_', name).strip()
    # Replace multiple spaces/hyphens with single underscore
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    return safe_name


def get_upcoming_meetings(
    category_id: str = "12592",  # Default: ATLAS category
    days_ahead: int = 7,
    base_url: str = "https://indico.cern.ch",
    api_token: Optional[str] = None,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> str:
    """Fetch upcoming ATLAS meetings for the next N days.

    Parameters
    ----------
    category_id : str
        The Indico category ID (default: 12592 for ATLAS).
    days_ahead : int
        Number of days to look ahead (default: 7).
    base_url : str
        Base URL for Indico instance (default: https://indico.cern.ch).
    api_token : str, optional
        Bearer token for authentication
    api_key : str, optional
        API key for legacy authentication
    secret_key : str, optional
        Secret key for legacy authentication

    Returns
    -------
    str
        Markdown-formatted summary of upcoming meetings.
    """
    try:
        # Calculate date range
        today = datetime.now()
        end_date = today + timedelta(days=days_ahead)

        # Build API request
        path = '/export/categ/%s.json' % category_id
        params = {
            'from': today.strftime("%Y-%m-%d"),
            'to': end_date.strftime("%Y-%m-%d"),
            'detail': 'events',
            'pretty': 'yes',
            'order': 'start',
        }

        full_url = base_url + build_indico_request(path, params, api_token, api_key, secret_key)

        # Make request
        if api_token:
            req = urllib.request.Request(
                full_url,
                headers={"Authorization": "Bearer " + api_token, "Accept": "application/json"}
            )
        else:
            req = urllib.request.Request(full_url)

        response = urllib.request.urlopen(req, timeout=30)
        data = json.loads(response.read().decode('utf-8'))

        # Format output as markdown
        output = []
        output.append(f"# Upcoming ATLAS Meetings ({today.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
        output.append("")

        if "results" not in data or not data["results"]:
            output.append("No upcoming meetings found.")
            return "\n".join(output)

        # Group by date
        events_by_date = {}
        for event in data["results"]:
            event_date = event.get("startDate", {}).get("date", "Unknown")
            if event_date not in events_by_date:
                events_by_date[event_date] = []
            events_by_date[event_date].append(event)

        # Output by date
        for date in sorted(events_by_date.keys()):
            output.append(f"## {date}")
            output.append("")

            for event in events_by_date[date]:
                title = event.get("title", "Untitled")
                url = event.get("url", "")
                time = event.get("startDate", {}).get("time", "")
                location = event.get("location", "") or event.get("roomFullname", "")

                output.append(f"### {title}")
                if time:
                    output.append(f"**Time:** {time}")
                if location:
                    output.append(f"**Location:** {location}")
                if url:
                    output.append(f"**URL:** {url}")
                output.append("")

        return "\n".join(output)

    except urllib.error.HTTPError as e:
        return f"Error fetching meetings from Indico (HTTP {e.code}): {e.reason}"
    except urllib.error.URLError as e:
        return f"Error connecting to Indico: {e.reason}"
    except Exception as e:
        return f"Unexpected error: {e}"


def get_category_meetings(
    category_id: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    base_url: str = "https://indico.cern.ch",
    api_token: Optional[str] = None,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    include_slides: bool = False,
) -> str:
    """Get meeting history/future meetings for a category.

    Parameters
    ----------
    category_id : str
        The Indico category ID.
    from_date : str, optional
        Start date in YYYY-MM-DD format.
    to_date : str, optional
        End date in YYYY-MM-DD format.
    base_url : str
        Base URL for Indico instance (default: https://indico.cern.ch).
    api_token : str, optional
        Bearer token for authentication
    api_key : str, optional
        API key for legacy authentication
    secret_key : str, optional
        Secret key for legacy authentication
    include_slides : bool
        Whether to include information about available slides.

    Returns
    -------
    str
        Markdown-formatted list of meetings.
    """
    try:
        # Build API request
        path = '/export/categ/%s.json' % category_id
        params = {
            'detail': 'contributions' if include_slides else 'events',
            'pretty': 'yes',
            'order': 'start',
        }

        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        full_url = base_url + build_indico_request(path, params, api_token, api_key, secret_key)

        # Make request
        if api_token:
            req = urllib.request.Request(
                full_url,
                headers={"Authorization": "Bearer " + api_token, "Accept": "application/json"}
            )
        else:
            req = urllib.request.Request(full_url)

        response = urllib.request.urlopen(req, timeout=30)
        data = json.loads(response.read().decode('utf-8'))

        # Format output as markdown
        output = []
        output.append(f"# Meetings in Category {category_id}")
        if from_date or to_date:
            output.append(f"**Date Range:** {from_date or 'earliest'} to {to_date or 'latest'}")
        output.append("")

        if "results" not in data or not data["results"]:
            output.append("No meetings found.")
            return "\n".join(output)

        for event in data["results"]:
            title = event.get("title", "Untitled")
            url = event.get("url", "")
            start_date = event.get("startDate", {}).get("date", "Unknown")
            location = event.get("location", "") or event.get("roomFullname", "")

            output.append(f"## {title}")
            output.append(f"**Date:** {start_date}")
            if location:
                output.append(f"**Location:** {location}")
            output.append(f"**URL:** {url}")

            if include_slides and "contributions" in event:
                output.append("")
                output.append("### Contributions")
                for contrib in event.get("contributions", []):
                    contrib_title = contrib.get("title", "Untitled contribution")
                    speakers = contrib.get("speakers", [])
                    speaker_names = ", ".join([
                        f"{s.get('first_name', '')} {s.get('last_name', '')}".strip()
                        for s in speakers
                    ])
                    
                    output.append(f"- **{contrib_title}**")
                    if speaker_names:
                        output.append(f"  - Speakers: {speaker_names}")
                    
                    # Check for attachments/slides
                    folders = contrib.get("folders", [])
                    has_materials = any(
                        fold.get("attachments") for fold in folders
                    )
                    if has_materials:
                        output.append("  - **Has attachments/slides**")

            output.append("")

        return "\n".join(output)

    except urllib.error.HTTPError as e:
        return f"Error fetching category meetings (HTTP {e.code}): {e.reason}"
    except urllib.error.URLError as e:
        return f"Error connecting to Indico: {e.reason}"
    except Exception as e:
        return f"Unexpected error: {e}"


def download_event_materials(
    event_id: str,
    output_dir: str = "./indico_downloads",
    base_url: str = "https://indico.cern.ch",
    api_token: Optional[str] = None,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    slides_only: bool = False,
) -> str:
    """Download slides and abstracts for an event.

    Parameters
    ----------
    event_id : str
        The Indico event ID.
    output_dir : str
        Directory to save downloaded materials (default: ./indico_downloads).
    base_url : str
        Base URL for Indico instance.
    api_token : str, optional
        Bearer token for authentication
    api_key : str, optional
        API key for legacy authentication
    secret_key : str, optional
        Secret key for legacy authentication
    slides_only : bool
        Download only slides, skip abstracts.

    Returns
    -------
    str
        Summary of downloaded materials in markdown format.
    """
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Build API request for event details
        path = '/export/event/%s.json' % event_id
        params = {
            'detail': 'contributions',
            'pretty': 'yes',
        }

        full_url = base_url + build_indico_request(path, params, api_token, api_key, secret_key)

        # Make request
        if api_token:
            req = urllib.request.Request(
                full_url,
                headers={"Authorization": "Bearer " + api_token, "Accept": "application/json"}
            )
        else:
            req = urllib.request.Request(full_url)

        response = urllib.request.urlopen(req, timeout=30)
        data = json.loads(response.read().decode('utf-8'))

        if "results" not in data or not data["results"]:
            return f"Error: Event {event_id} not found."

        event = data["results"][0]
        event_title = event.get("title", f"event_{event_id}")

        # Sanitize event title for directory name
        safe_title = sanitize_filename(event_title)
        event_dir = output_path / safe_title
        event_dir.mkdir(parents=True, exist_ok=True)

        downloaded_files = []
        summary = []

        summary.append(f"# Downloaded Materials for: {event_title}")
        summary.append(f"**Event ID:** {event_id}")
        summary.append(f"**Output Directory:** {event_dir}")
        summary.append("")

        # Create event summary markdown
        event_summary = []
        event_summary.append(f"# {event_title}")
        event_summary.append("")
        event_summary.append(f"**URL:** {event.get('url', '')}")
        event_summary.append(f"**Date:** {event.get('startDate', {}).get('date', 'Unknown')}")
        event_summary.append(f"**Location:** {event.get('location', 'Unknown')}")
        event_summary.append("")

        # Process contributions
        if "contributions" in event:
            event_summary.append("## Contributions")
            event_summary.append("")

            for idx, contrib in enumerate(event["contributions"], 1):
                contrib_title = contrib.get("title", "Untitled")
                speakers = contrib.get("speakers", [])
                speaker_names = ", ".join([
                    f"{s.get('first_name', '')} {s.get('last_name', '')}".strip()
                    for s in speakers
                ])

                event_summary.append(f"### {idx}. {contrib_title}")
                if speaker_names:
                    event_summary.append(f"**Speakers:** {speaker_names}")

                # Get abstract/description
                if not slides_only and "description" in contrib:
                    description = contrib.get("description", "")
                    if description:
                        event_summary.append("")
                        event_summary.append("**Abstract:**")
                        event_summary.append(description)

                # Download slides/materials from folders
                folders = contrib.get("folders", [])
                if folders:
                    event_summary.append("")
                    event_summary.append("**Materials:**")

                    for folder in folders:
                        attachments = folder.get("attachments", [])
                        for attachment in attachments:
                            file_title = attachment.get("title", "file")
                            download_url = attachment.get("download_url", "")
                            file_size = attachment.get("size", 0)

                            if download_url:
                                try:
                                    # Download file
                                    if api_token:
                                        file_req = urllib.request.Request(
                                            download_url,
                                            headers={"Authorization": "Bearer " + api_token}
                                        )
                                    else:
                                        file_req = urllib.request.Request(download_url)
                                    
                                    file_response = urllib.request.urlopen(file_req, timeout=60)
                                    file_data = file_response.read()

                                    # Determine file extension from URL or content-type
                                    ext = Path(download_url).suffix or ".bin"
                                    safe_filename = sanitize_filename(file_title)
                                    safe_filename = f"contrib_{idx:03d}_{safe_filename}{ext}"
                                    file_path = event_dir / safe_filename

                                    with open(file_path, "wb") as f:
                                        f.write(file_data)

                                    downloaded_files.append(str(file_path))
                                    size_str = f" ({file_size} bytes)" if file_size else ""
                                    event_summary.append(f"- [{file_title}]({safe_filename}){size_str}")
                                    summary.append(f"✓ Downloaded: {safe_filename}")

                                except Exception as e:
                                    event_summary.append(f"- {file_title} (download failed: {e})")
                                    summary.append(f"✗ Failed: {file_title} - {e}")

                event_summary.append("")

        # Save event summary
        summary_file = event_dir / "event_summary.md"
        with open(summary_file, "w", encoding='utf-8') as f:
            f.write("\n".join(event_summary))

        downloaded_files.append(str(summary_file))
        summary.append("")
        summary.append(f"**Summary saved to:** {summary_file}")
        summary.append(f"**Total files downloaded:** {len(downloaded_files)}")

        return "\n".join(summary)

    except urllib.error.HTTPError as e:
        return f"Error downloading event materials (HTTP {e.code}): {e.reason}"
    except urllib.error.URLError as e:
        return f"Error connecting to Indico: {e.reason}"
    except Exception as e:
        return f"Unexpected error: {e}"


def main():
    """Main entry point for the script."""
    # Check dependencies first
    dep_error = check_dependencies()
    if dep_error:
        print(dep_error, file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="CERN Indico Meeting Operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get upcoming ATLAS meetings for the next week
  %(prog)s upcoming --category 12592 --days 7

  # Get today's ATLAS meetings
  %(prog)s upcoming --today

  # Get all ATLAS meetings in January 2024
  %(prog)s category 12592 --from 2024-01-01 --to 2024-01-31

  # Download materials from a specific event
  %(prog)s download 1234567 --output ./my_downloads

Environment Variables:
  INDICO_API_TOKEN  Bearer token for authentication (recommended, from https://indico.cern.ch/user/tokens/)
  INDICO_API_KEY    API key for legacy authentication
  INDICO_SECRET_KEY Secret key for legacy authentication
  INDICO_BASE_URL   Base URL for Indico instance (default: https://indico.cern.ch)

Authentication:
  You can authenticate using either:
  1. API Token (recommended): Get from https://indico.cern.ch/user/tokens/
  2. API Key + Secret: Get from https://indico.cern.ch/user/api/ (legacy)
  3. ~/.indico-secret-key file with API_KEY and SECRET_KEY on separate lines
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Upcoming meetings command
    upcoming_parser = subparsers.add_parser("upcoming", help="Get upcoming meetings")
    upcoming_parser.add_argument(
        "--category",
        type=str,
        default="12592",
        help="Indico category ID (default: 12592 for ATLAS)",
    )
    upcoming_parser.add_argument(
        "--days", type=int, default=7, help="Number of days to look ahead (default: 7)"
    )
    upcoming_parser.add_argument(
        "--today", action="store_true", help="Get only today's meetings (equivalent to --days 1)"
    )

    # Category meetings command
    category_parser = subparsers.add_parser("category", help="Get meetings in a category")
    category_parser.add_argument("category_id", type=str, help="Indico category ID")
    category_parser.add_argument(
        "--from", dest="from_date", type=str, help="Start date (YYYY-MM-DD)"
    )
    category_parser.add_argument("--to", dest="to_date", type=str, help="End date (YYYY-MM-DD)")
    category_parser.add_argument(
        "--slides", action="store_true", help="Include information about slides"
    )

    # Download materials command
    download_parser = subparsers.add_parser("download", help="Download event materials")
    download_parser.add_argument("event_id", type=str, help="Indico event ID")
    download_parser.add_argument(
        "--output",
        type=str,
        default="./indico_downloads",
        help="Output directory (default: ./indico_downloads)",
    )
    download_parser.add_argument(
        "--slides-only", action="store_true", help="Download only slides, skip abstracts"
    )

    # Global options
    parser.add_argument("--base-url", type=str, help="Base URL for Indico instance")
    parser.add_argument("--api-token", type=str, help="Bearer token for authentication")
    parser.add_argument("--api-key", type=str, help="API key for legacy authentication")
    parser.add_argument("--secret-key", type=str, help="Secret key for legacy authentication")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Get configuration from environment or arguments
    base_url = args.base_url or os.environ.get("INDICO_BASE_URL", "https://indico.cern.ch")
    api_token = args.api_token or os.environ.get("INDICO_API_TOKEN", "")
    api_key = args.api_key or os.environ.get("INDICO_API_KEY", "")
    secret_key = args.secret_key or os.environ.get("INDICO_SECRET_KEY", "")

    # Try to load from secret file if no authentication provided
    if not api_token and not (api_key and secret_key):
        try:
            secret_file = os.path.expanduser("~/.indico-secret-key")
            if os.path.exists(secret_file):
                with open(secret_file) as f:
                    lines = f.read().strip().split('\n')
                    if len(lines) >= 2:
                        api_key = lines[0].strip()
                        secret_key = lines[1].strip()
        except Exception:
            pass

    # Execute command
    if args.command == "upcoming":
        days = 1 if args.today else args.days
        result = get_upcoming_meetings(
            category_id=args.category,
            days_ahead=days,
            base_url=base_url,
            api_token=api_token,
            api_key=api_key,
            secret_key=secret_key,
        )
        print(result)

    elif args.command == "category":
        result = get_category_meetings(
            category_id=args.category_id,
            from_date=args.from_date,
            to_date=args.to_date,
            base_url=base_url,
            api_token=api_token,
            api_key=api_key,
            secret_key=secret_key,
            include_slides=args.slides,
        )
        print(result)

    elif args.command == "download":
        result = download_event_materials(
            event_id=args.event_id,
            output_dir=args.output,
            base_url=base_url,
            api_token=api_token,
            api_key=api_key,
            secret_key=secret_key,
            slides_only=args.slides_only,
        )
        print(result)


if __name__ == "__main__":
    main()
