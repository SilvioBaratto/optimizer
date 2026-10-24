"""Release timezones for the economic-event calendar's ``Region`` codes.

Yahoo reports every economic event as a UTC instant of the local release time
(Japan CPI at 08:30 JST arrives as 23:30Z the previous day), while the calendar
keys events on a calendar date. The date that matters to a reader is the one in
the region where the event is released, so the repository converts the instant
into the region's timezone before taking its date.
"""

from __future__ import annotations

from functools import cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# One zone per region code (ISO 3166 alpha-2, plus Yahoo's "EU"). Multi-zone
# countries use the zone of their statistics office or central bank.
REGION_TIMEZONES: dict[str, str] = {
    "AE": "Asia/Dubai",
    "AO": "Africa/Luanda",
    "AR": "America/Argentina/Buenos_Aires",
    "AT": "Europe/Vienna",
    "AU": "Australia/Sydney",
    "BE": "Europe/Brussels",
    "BG": "Europe/Sofia",
    "BH": "Asia/Bahrain",
    "BR": "America/Sao_Paulo",
    "BW": "Africa/Gaborone",
    "CA": "America/Toronto",
    "CH": "Europe/Zurich",
    "CL": "America/Santiago",
    "CN": "Asia/Shanghai",
    "CO": "America/Bogota",
    "CY": "Asia/Nicosia",
    "CZ": "Europe/Prague",
    "DE": "Europe/Berlin",
    "DK": "Europe/Copenhagen",
    "EE": "Europe/Tallinn",
    "EG": "Africa/Cairo",
    "ES": "Europe/Madrid",
    "EU": "Europe/Brussels",
    "FI": "Europe/Helsinki",
    "FR": "Europe/Paris",
    "GB": "Europe/London",
    "GH": "Africa/Accra",
    "GR": "Europe/Athens",
    "HK": "Asia/Hong_Kong",
    "HR": "Europe/Zagreb",
    "HU": "Europe/Budapest",
    "ID": "Asia/Jakarta",
    "IE": "Europe/Dublin",
    "IL": "Asia/Jerusalem",
    "IN": "Asia/Kolkata",
    "IS": "Atlantic/Reykjavik",
    "IT": "Europe/Rome",
    "JP": "Asia/Tokyo",
    "KE": "Africa/Nairobi",
    "KR": "Asia/Seoul",
    "KW": "Asia/Kuwait",
    "LT": "Europe/Vilnius",
    "LU": "Europe/Luxembourg",
    "LV": "Europe/Riga",
    "MT": "Europe/Malta",
    "MU": "Indian/Mauritius",
    "MX": "America/Mexico_City",
    "MY": "Asia/Kuala_Lumpur",
    "MZ": "Africa/Maputo",
    "NA": "Africa/Windhoek",
    "NG": "Africa/Lagos",
    "NL": "Europe/Amsterdam",
    "NO": "Europe/Oslo",
    "NZ": "Pacific/Auckland",
    "OM": "Asia/Muscat",
    "PE": "America/Lima",
    "PH": "Asia/Manila",
    "PK": "Asia/Karachi",
    "PL": "Europe/Warsaw",
    "PT": "Europe/Lisbon",
    "QA": "Asia/Qatar",
    "RO": "Europe/Bucharest",
    "RU": "Europe/Moscow",
    "SA": "Asia/Riyadh",
    "SE": "Europe/Stockholm",
    "SG": "Asia/Singapore",
    "SI": "Europe/Ljubljana",
    "SK": "Europe/Bratislava",
    "TH": "Asia/Bangkok",
    "TR": "Europe/Istanbul",
    "TW": "Asia/Taipei",
    "TZ": "Africa/Dar_es_Salaam",
    "UA": "Europe/Kiev",
    "UG": "Africa/Kampala",
    "US": "America/New_York",
    "VN": "Asia/Ho_Chi_Minh",
    "ZA": "Africa/Johannesburg",
    "ZM": "Africa/Lusaka",
}


@cache
def region_timezone(region: str | None) -> ZoneInfo | None:
    """Return the release timezone for a calendar region code.

    Args:
        region: Region code as Yahoo reports it (case and surrounding
            whitespace are ignored); ``None`` or the ``"?"`` placeholder for a
            missing region are accepted.

    Returns:
        The region's timezone, or ``None`` when the region is unknown or the
        platform's tz database lacks the zone, so callers can fall back to the
        UTC date instead of failing the whole batch.
    """
    name = REGION_TIMEZONES.get((region or "").strip().upper())
    if name is None:
        return None
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return None
