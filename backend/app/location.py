"""Validated camera locations and common time/spatial filtering."""
import datetime
import math
import os


def valid_coordinates(lat, lng):
    try:
        return (lat is not None and lng is not None and math.isfinite(float(lat))
                and math.isfinite(float(lng)) and -90 <= float(lat) <= 90 and -180 <= float(lng) <= 180)
    except (TypeError, ValueError):
        return False


def resolve_location(values, existing=None):
    supplied = "lat" in values or "lng" in values
    if supplied:
        lat, lng = values.get("lat"), values.get("lng")
        if lat in (None, "") and lng in (None, ""):
            return 0.0, 0.0, False
        if not valid_coordinates(lat, lng):
            raise ValueError("Latitude and longitude must be a finite pair within -90..90 and -180..180")
        return float(lat), float(lng), True
    if existing is not None:
        return existing.lat, existing.lng, bool(existing.location_known)
    lat, lng = os.getenv("ANPR_DEFAULT_LAT"), os.getenv("ANPR_DEFAULT_LNG")
    if lat is not None or lng is not None:
        if not valid_coordinates(lat, lng):
            raise ValueError("Invalid ANPR_DEFAULT_LAT / ANPR_DEFAULT_LNG configuration")
        return float(lat), float(lng), True
    return 0.0, 0.0, False


def camera_location(camera):
    if camera is None or not getattr(camera, "location_known", True):
        return None, None
    return (camera.lat, camera.lng) if valid_coordinates(camera.lat, camera.lng) else (None, None)


def utc_naive(value):
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return value


def time_window(start=None, end=None, minutes=None):
    start, end = utc_naive(start), utc_naive(end)
    if start is None and minutes is not None:
        start = (end or datetime.datetime.utcnow()) - datetime.timedelta(minutes=minutes)
    if start is not None and end is not None and start > end:
        raise ValueError("Start must be before end")
    return start, end


def iso_utc(value):
    return utc_naive(value).isoformat() + "Z"
