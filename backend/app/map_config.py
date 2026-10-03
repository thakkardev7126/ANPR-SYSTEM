"""Configurable map tile settings for the dashboard.

The dashboard should not be tied directly to the public OpenStreetMap tile
server. This small config helper gives the SIH demo a sensible default while
letting deployments provide their own tile endpoint and attribution.
"""
from __future__ import annotations

import os


DEFAULT_TILE_PROVIDERS = {
    "esri_world_street": {
        "tile_url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        "attribution": (
            'Tiles &copy; <a href="https://www.esri.com/">Esri</a> and contributors; '
            'data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        ),
        "subdomains": "",
        "max_zoom": 19,
    },
    "carto_dark": {
        "tile_url": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        "attribution": (
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
            'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        ),
        "subdomains": "abcd",
        "max_zoom": 19,
    },
    "carto_light": {
        "tile_url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        "attribution": (
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
            'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        ),
        "subdomains": "abcd",
        "max_zoom": 19,
    },
}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def map_tile_config():
    provider_name = os.environ.get("ANPR_MAP_TILE_PROVIDER", "esri_world_street").strip() or "esri_world_street"
    provider = DEFAULT_TILE_PROVIDERS.get(provider_name, DEFAULT_TILE_PROVIDERS["esri_world_street"]).copy()
    custom_url = os.environ.get("ANPR_MAP_TILE_URL", "").strip()
    custom_attribution = os.environ.get("ANPR_MAP_TILE_ATTRIBUTION", "").strip()
    custom_subdomains = os.environ.get("ANPR_MAP_TILE_SUBDOMAINS", "").strip()
    if custom_url:
        provider_name = "custom"
        provider["tile_url"] = custom_url
    if custom_attribution:
        provider["attribution"] = custom_attribution
    if custom_subdomains:
        provider["subdomains"] = custom_subdomains
    provider["max_zoom"] = _int_env("ANPR_MAP_TILE_MAX_ZOOM", provider.get("max_zoom", 19))
    provider["provider"] = provider_name
    provider["fallback_message"] = (
        "Map tiles are temporarily unavailable. Camera markers and trajectory data are still shown."
    )
    return provider
