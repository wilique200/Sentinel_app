# ============================================================================
# StormSentinel Backend — Geocoding
# Global location search (Open-Meteo geocoding API) with disambiguation,
# plus country → region mapping for the model's region one-hot features.
# ============================================================================

import requests

# Full US state name -> abbreviation (kept for display purposes only —
# region, not state, drives the model's geographic features in v2)
US_STATE_NAMES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
    "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

# Country code -> one of the 8 regions the model was trained on. Countries
# not listed here are genuinely outside the training footprint — the
# feature builder leaves region dummies at zero for them (the model's
# learned "no specific region matched" behavior) rather than force-fitting
# them into a geographically/climatically wrong bucket. East Asia (China,
# Japan, Korea) is a notable gap — the training set never included it.
COUNTRY_TO_REGION = {
    # North America
    "US": "North America", "CA": "North America", "MX": "North America",
    # Europe
    "GB": "Europe", "DE": "Europe", "FR": "Europe", "IT": "Europe", "ES": "Europe",
    "PT": "Europe", "GR": "Europe", "NL": "Europe", "BE": "Europe", "CH": "Europe",
    "AT": "Europe", "SE": "Europe", "NO": "Europe", "DK": "Europe", "FI": "Europe",
    "IE": "Europe", "PL": "Europe", "CZ": "Europe", "HU": "Europe", "RO": "Europe",
    "BG": "Europe", "HR": "Europe", "SK": "Europe", "SI": "Europe", "LT": "Europe",
    "LV": "Europe", "EE": "Europe", "IS": "Europe", "LU": "Europe", "MT": "Europe", "CY": "Europe",
    # Middle East
    "KW": "Middle East", "AE": "Middle East", "SA": "Middle East", "IQ": "Middle East",
    "IR": "Middle East", "IL": "Middle East", "JO": "Middle East", "LB": "Middle East",
    "SY": "Middle East", "YE": "Middle East", "OM": "Middle East", "QA": "Middle East", "BH": "Middle East",
    # South Asia
    "IN": "South Asia", "BD": "South Asia", "PK": "South Asia", "LK": "South Asia",
    "NP": "South Asia", "BT": "South Asia", "MV": "South Asia", "AF": "South Asia",
    # Southeast Asia
    "PH": "Southeast Asia", "ID": "Southeast Asia", "TH": "Southeast Asia", "VN": "Southeast Asia",
    "MY": "Southeast Asia", "SG": "Southeast Asia", "MM": "Southeast Asia", "KH": "Southeast Asia",
    "LA": "Southeast Asia", "BN": "Southeast Asia", "TL": "Southeast Asia",
    # Oceania
    "AU": "Oceania", "NZ": "Oceania", "FJ": "Oceania", "PG": "Oceania",
    # Africa
    "KE": "Africa", "ZA": "Africa", "NG": "Africa", "EG": "Africa", "ET": "Africa",
    "GH": "Africa", "TZ": "Africa", "UG": "Africa", "DZ": "Africa", "MA": "Africa",
    "TN": "Africa", "LY": "Africa", "SD": "Africa", "CI": "Africa", "CM": "Africa",
    "ZW": "Africa", "ZM": "Africa", "MZ": "Africa", "AO": "Africa", "SN": "Africa", "ML": "Africa",
    # South America
    "CL": "South America", "BR": "South America", "AR": "South America", "PE": "South America",
    "CO": "South America", "VE": "South America", "EC": "South America", "BO": "South America",
    "PY": "South America", "UY": "South America", "GY": "South America", "SR": "South America",
}


def get_region_for_country(country_code: str) -> str | None:
    """Returns the trained region name, or None if genuinely outside the training footprint."""
    return COUNTRY_TO_REGION.get(country_code)


def geocode_location(query: str) -> list[dict]:
    """
    Converts free-text location into a list of matching candidates ANYWHERE
    in the world, using Open-Meteo's free geocoding API.

    Ambiguous names (e.g. "Paris" -> Paris FR, Paris TX, Paris KY...) return
    multiple entries sorted by population — largest first, purely to make
    the list easier to scan, never auto-picked. The caller (the /geocode
    endpoint) is responsible for disambiguation UX; this function never
    guesses which one the user meant.
    """
    if not query or not query.strip():
        return []

    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": query.strip(), "count": 20, "language": "en", "format": "json"}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        results = r.json().get("results", [])
    except Exception:
        return []

    if not results:
        return []

    candidates = []
    for res in results:
        country_code = res.get("country_code", "")
        country_name = res.get("country", country_code)
        admin1 = res.get("admin1", "")
        admin2 = res.get("admin2", "")
        population = res.get("population")
        is_us = country_code == "US"

        state_label = US_STATE_NAMES.get(admin1, admin1 or "??") if is_us else (admin1 or country_name)
        region = get_region_for_country(country_code)

        label = f"{res.get('name', query)}, {state_label}"
        if is_us and admin2:
            label += f" ({admin2})"
        elif not is_us:
            label += f", {country_name}"
        if population:
            label += f" — pop. {population:,}"

        display_name = f"{res.get('name', query)}, {state_label}"
        if not is_us:
            display_name += f", {country_name}"

        candidates.append({
            "city": res.get("name", query),
            "state": state_label,
            "country_code": country_code,
            "country_name": country_name,
            "region": region,  # None if outside the training footprint
            "lat": res.get("latitude"),
            "lon": res.get("longitude"),
            "is_us": is_us,
            "display_name": display_name,
            "disambiguation_label": label,
            "population": population or 0,
        })

    candidates.sort(key=lambda c: c["population"], reverse=True)
    return candidates
