#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
import re
from typing import Optional, Dict, List
from datetime import datetime
import time


# Country code mapping to Trading Economics URL slugs
COUNTRY_MAPPING = {
    "USA": "united-states",
    "Germany": "germany",
    "France": "france",
    "UK": "united-kingdom",
    "Japan": "japan",
}

# Country code to full name mapping (for industrial production page)
COUNTRY_NAME_MAPPING = {
    "USA": "United States",
    "Germany": "Germany",
    "France": "France",
    "UK": "United Kingdom",
    "Japan": "Japan",
}

# Indicator name variations (Trading Economics uses different naming conventions)
INDICATOR_PATTERNS = {
    "gdp_growth_rate": [
        "GDP Growth Rate",
    ],
    "gdp_growth_yoy": [
        "GDP Annual Growth Rate",
        "GDP YoY",
    ],
    "unemployment_rate": [
        "Unemployment Rate",
    ],
    "inflation_rate": [
        "Inflation Rate",
    ],
    "inflation_rate_mom": [
        "Inflation Rate MoM",
    ],
    "industrial_production": [
        "Industrial Production",
    ],
    "manufacturing_pmi": [
        "Manufacturing PMI",
    ],
    "services_pmi": [
        "Services PMI",
    ],
    "interest_rate": [
        "Interest Rate",
    ],
    "government_debt_gdp": [
        "Government Debt to GDP",
    ],
    "budget_balance_gdp": [
        "Government Budget",
    ],
    "consumer_confidence": [
        "Consumer Confidence",
    ],
    "business_confidence": [
        "Business Confidence",
    ],
    "core_inflation": [
        "Core Inflation Rate",
    ],
    "retail_sales_mom": [
        "Retail Sales MoM",
    ],
    "retail_sales_yoy": [
        "Retail Sales YoY",
    ],
    "current_account": [
        "Current Account",
    ],
    "current_account_gdp": [
        "Current Account to GDP",
    ],
    "trade_balance": [
        "Balance of Trade",
    ],
    "composite_pmi": [
        "Composite PMI",
    ],
}


class TradingEconomicsIndicatorsScraper:
    BASE_URL = "https://tradingeconomics.com"

    def __init__(self, timeout: int = 20, rate_limit_delay: float = 1.0):
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()

        # Browser-like headers to avoid blocking
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )

    def get_country_indicators(self, country: str, include_bonds: bool = True) -> Dict:
        if country not in COUNTRY_MAPPING:
            return {
                "country": country,
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": f"Country not supported. Available: {list(COUNTRY_MAPPING.keys())}",
            }

        country_slug = COUNTRY_MAPPING[country]
        url = f"{self.BASE_URL}/{country_slug}/indicators"

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            indicators = self._parse_indicators_table(soup)

            result = {
                "country": country,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "source_url": url,
                "indicators": indicators,
                "num_indicators": len(indicators),
            }

            # Fetch bond yields
            if include_bonds:
                bond_yields = self.get_bond_yields(country)
                if bond_yields.get("status") == "success":
                    result["bond_yields"] = bond_yields.get("yields", {})
                    result["num_bond_yields"] = len(bond_yields.get("yields", {}))
                else:
                    result["bond_yields"] = {}
                    result["bond_yields_error"] = bond_yields.get("error")

            return result

        except requests.RequestException as e:
            return {
                "country": country,
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": f"HTTP request failed: {str(e)}",
            }
        except Exception as e:
            return {
                "country": country,
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": f"Parsing failed: {str(e)}",
            }

    def _parse_indicators_table(self, soup: BeautifulSoup) -> Dict:
        indicators = {}

        # Find all tables with class "table table-hover"
        tables = soup.find_all("table", class_="table-hover")

        for table in tables:
            rows = table.find_all("tr")

            for row in rows:
                cells = row.find_all("td")

                # Need at least 7 columns: Indicator | Last | Previous | Highest | Lowest | Unit | Reference
                if len(cells) >= 3:
                    try:
                        # Column 0: Indicator name (inside <a> tag)
                        indicator_cell = cells[0]
                        indicator_link = indicator_cell.find("a")
                        if not indicator_link:
                            continue

                        indicator_name = indicator_link.get_text(strip=True)

                        # Column 1: Last (current value)
                        last_text = cells[1].get_text(strip=True)

                        # Column 2: Previous value
                        previous_text = cells[2].get_text(strip=True)

                        # Column 5: Unit (if exists)
                        unit = cells[5].get_text(strip=True) if len(cells) > 5 else ""

                        # Column 6: Reference date (if exists)
                        reference = cells[6].get_text(strip=True) if len(cells) > 6 else ""

                        # Extract numeric values
                        last_value = self._extract_number(last_text)
                        previous_value = self._extract_number(previous_text)

                        if last_value is not None:
                            # Match indicator to our standard naming
                            matched_key = self._match_indicator_name(indicator_name)

                            if matched_key:
                                indicators[matched_key] = {
                                    "value": last_value,
                                    "previous": previous_value,
                                    "unit": unit,
                                    "reference": reference,
                                    "raw_name": indicator_name,
                                }

                    except (ValueError, IndexError, AttributeError):
                        # Skip rows that don't match expected format
                        continue

        return indicators

    def _match_indicator_name(self, name: str) -> Optional[str]:
        name_lower = name.lower()

        for standard_key, patterns in INDICATOR_PATTERNS.items():
            for pattern in patterns:
                # Exact match (case-insensitive)
                if pattern.lower() == name_lower:
                    return standard_key

        return None

    def _extract_number(self, text: str) -> Optional[float]:
        if not text or text.strip().upper() in ["N/A", "NA", "-", ""]:
            return None

        # Remove common non-numeric characters but keep decimal, negative, and digits
        cleaned = re.sub(r"[%,$€£¥\s]", "", text)

        # Extract number (handles negative, decimal)
        match = re.search(r"(-?\d+\.?\d*)", cleaned)

        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None

        return None

    def get_bond_yields(self, country: str) -> Dict:
        if country not in COUNTRY_MAPPING:
            return {
                "status": "error",
                "country": country,
                "error": f"Country not supported. Available: {list(COUNTRY_MAPPING.keys())}",
            }

        country_slug = COUNTRY_MAPPING[country]
        url = f"{self.BASE_URL}/{country_slug}/government-bond-yield"

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            yields = self._parse_bond_yields_table(soup, country)

            return {
                "status": "success",
                "country": country,
                "source_url": url,
                "yields": yields,
                "timestamp": datetime.now().isoformat(),
            }

        except requests.RequestException as e:
            return {
                "status": "error",
                "country": country,
                "error": f"HTTP request failed: {str(e)}",
            }
        except Exception as e:
            return {"status": "error", "country": country, "error": f"Parsing failed: {str(e)}"}

    def _parse_bond_yields_table(self, soup: BeautifulSoup, _country: str) -> Dict:
        yields = {}

        # Find the bonds table (table-heatmap class)
        tables = soup.find_all("table", class_="table-heatmap")

        # Fallback: find any table with sortable theme
        if not tables:
            tables = soup.find_all("table", class_="sortable-theme-minimal")

        for table in tables:
            rows = table.find_all("tr")

            for row in rows:
                cells = row.find_all("td")

                # Need at least 7 columns
                if len(cells) >= 4:
                    try:
                        # Column 0: Bond name (inside <a> tag)
                        bond_cell = cells[0]
                        bond_link = bond_cell.find("a")
                        if not bond_link:
                            continue

                        bond_name = bond_link.get_text(strip=True)

                        # Extract maturity (e.g., "10Y" from "France 10Y")
                        maturity_match = re.search(r"(\d+Y|1M|3M|6M|52W)", bond_name)
                        if not maturity_match:
                            continue

                        maturity = maturity_match.group(1)

                        # Normalize maturity names
                        if maturity == "52W":
                            maturity = "1Y"

                        # Only keep key maturities: 2Y, 5Y, 10Y, 30Y
                        if maturity not in ["2Y", "5Y", "10Y", "30Y"]:
                            continue

                        # Column 1: Yield value
                        yield_text = cells[1].get_text(strip=True)
                        yield_value = self._extract_number(yield_text)

                        if yield_value is None:
                            continue

                        # Column 3: Day change
                        day_change_text = cells[3].get_text(strip=True) if len(cells) > 3 else ""
                        day_change = self._extract_number(day_change_text)

                        # Column 4: Month change
                        month_change_text = cells[4].get_text(strip=True) if len(cells) > 4 else ""
                        month_change = self._extract_number(month_change_text)

                        # Column 5: Year change
                        year_change_text = cells[5].get_text(strip=True) if len(cells) > 5 else ""
                        year_change = self._extract_number(year_change_text)

                        # Column 6: Date
                        date_text = cells[6].get_text(strip=True) if len(cells) > 6 else ""

                        yields[maturity] = {
                            "yield": yield_value,
                            "day_change": day_change,
                            "month_change": month_change,
                            "year_change": year_change,
                            "date": date_text,
                            "raw_name": bond_name,
                        }

                    except (ValueError, IndexError, AttributeError):
                        continue

        return yields

    def get_industrial_production_all(self) -> Dict:
        url = f"{self.BASE_URL}/country-list/industrial-production"

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            production_data = self._parse_industrial_production_table(soup)

            return {
                "status": "success",
                "source_url": url,
                "data": production_data,
                "timestamp": datetime.now().isoformat(),
            }

        except requests.RequestException as e:
            return {"status": "error", "error": f"HTTP request failed: {str(e)}"}
        except Exception as e:
            return {"status": "error", "error": f"Parsing failed: {str(e)}"}

    def get_capacity_utilization_all(self) -> Dict:
        url = f"{self.BASE_URL}/country-list/capacity-utilization?continent=g20"

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            capacity_data = self._parse_capacity_utilization_table(soup)

            return {
                "status": "success",
                "source_url": url,
                "data": capacity_data,
                "timestamp": datetime.now().isoformat(),
            }

        except requests.RequestException as e:
            return {"status": "error", "error": f"HTTP request failed: {str(e)}"}
        except Exception as e:
            return {"status": "error", "error": f"Parsing failed: {str(e)}"}

    def _parse_industrial_production_table(self, soup: BeautifulSoup) -> Dict:
        production = {}

        # Find the industrial production table (table-heatmap class)
        tables = soup.find_all("table", class_="table-heatmap")

        for table in tables:
            rows = table.find_all("tr")

            for row in rows:
                cells = row.find_all("td")

                # Need at least 5 columns
                if len(cells) >= 3:
                    try:
                        # Column 0: Country name (inside <a> tag)
                        country_cell = cells[0]
                        country_link = country_cell.find("a")
                        if not country_link:
                            continue

                        country_name = country_link.get_text(strip=True)

                        # Column 1: Last value
                        last_text = cells[1].get_text(strip=True)
                        last_value = self._extract_number(last_text)

                        if last_value is None:
                            continue

                        # Column 2: Previous value
                        previous_text = cells[2].get_text(strip=True)
                        previous_value = self._extract_number(previous_text)

                        # Column 3: Reference date
                        reference = cells[3].get_text(strip=True) if len(cells) > 3 else ""

                        # Column 4: Unit
                        unit = cells[4].get_text(strip=True) if len(cells) > 4 else ""

                        production[country_name] = {
                            "value": last_value,
                            "previous": previous_value,
                            "reference": reference,
                            "unit": unit,
                        }

                    except (ValueError, IndexError, AttributeError):
                        continue

        return production

    def _parse_capacity_utilization_table(self, soup: BeautifulSoup) -> Dict:
        capacity = {}

        # Find the capacity utilization table (table-heatmap class)
        tables = soup.find_all("table", class_="table-heatmap")

        for table in tables:
            rows = table.find_all("tr")

            for row in rows:
                cells = row.find_all("td")

                # Need at least 3 columns
                if len(cells) >= 3:
                    try:
                        # Column 0: Country name (inside <a> tag)
                        country_cell = cells[0]
                        country_link = country_cell.find("a")
                        if not country_link:
                            continue

                        country_name = country_link.get_text(strip=True)

                        # Column 1: Last value
                        last_text = cells[1].get_text(strip=True)
                        last_value = self._extract_number(last_text)

                        if last_value is None:
                            continue

                        # Column 2: Previous value
                        previous_text = cells[2].get_text(strip=True)
                        previous_value = self._extract_number(previous_text)

                        # Column 3: Reference date
                        reference = cells[3].get_text(strip=True) if len(cells) > 3 else ""

                        # Column 4: Unit
                        unit = cells[4].get_text(strip=True) if len(cells) > 4 else ""

                        capacity[country_name] = {
                            "value": last_value,
                            "previous": previous_value,
                            "reference": reference,
                            "unit": unit,
                        }

                    except (ValueError, IndexError, AttributeError):
                        continue

        return capacity

    def get_multiple_countries(
        self,
        countries: List[str],
        include_bonds: bool = True,
        include_industrial_production: bool = True,
        include_capacity_utilization: bool = True,
    ) -> Dict[str, Dict]:
        results = {}

        # Fetch industrial production data once for all countries (more efficient)
        industrial_production_data = {}
        if include_industrial_production:
            prod_result = self.get_industrial_production_all()
            if prod_result.get("status") == "success":
                industrial_production_data = prod_result.get("data", {})

        # Fetch capacity utilization data once for all countries (more efficient)
        capacity_utilization_data = {}
        if include_capacity_utilization:
            capacity_result = self.get_capacity_utilization_all()
            if capacity_result.get("status") == "success":
                capacity_utilization_data = capacity_result.get("data", {})

        for i, country in enumerate(countries):
            results[country] = self.get_country_indicators(country, include_bonds=include_bonds)

            # Add industrial production data if available
            if include_industrial_production and industrial_production_data:
                country_full_name = COUNTRY_NAME_MAPPING.get(country)
                if country_full_name and country_full_name in industrial_production_data:
                    results[country]["industrial_production"] = industrial_production_data[
                        country_full_name
                    ]

            # Add capacity utilization data if available
            if include_capacity_utilization and capacity_utilization_data:
                country_full_name = COUNTRY_NAME_MAPPING.get(country)
                if country_full_name and country_full_name in capacity_utilization_data:
                    results[country]["capacity_utilization"] = capacity_utilization_data[
                        country_full_name
                    ]

            # Rate limiting: delay between requests (except for last)
            if i < len(countries) - 1:
                time.sleep(self.rate_limit_delay)

        return results

    def get_all_portfolio_countries(self) -> Dict[str, Dict]:
        return self.get_multiple_countries(list(COUNTRY_MAPPING.keys()))

    def get_indicator_summary(self, country_data: Dict) -> str:
        if country_data.get("status") != "success":
            return f"{country_data['country']}: Error - {country_data.get('error', 'Unknown')}"

        country = country_data["country"]
        indicators = country_data.get("indicators", {})

        lines = [f"\n{country} Economic Indicators:"]
        lines.append("=" * 80)

        # Key indicators to highlight
        key_indicators = [
            ("gdp_growth_rate", "GDP Growth Rate (QoQ)"),
            ("gdp_growth_yoy", "GDP Annual Growth Rate"),
            ("unemployment_rate", "Unemployment Rate"),
            ("inflation_rate", "Inflation Rate"),
            ("manufacturing_pmi", "Manufacturing PMI"),
            ("services_pmi", "Services PMI"),
            ("interest_rate", "Interest Rate"),
            ("government_debt_gdp", "Government Debt/GDP"),
            ("budget_balance_gdp", "Budget Balance/GDP"),
        ]

        for key, label in key_indicators:
            if key in indicators:
                data = indicators[key]
                value = data["value"]
                previous = data.get("previous")
                unit = data.get("unit", "")
                reference = data.get("reference", "")

                change_indicator = ""
                if previous is not None:
                    if value > previous:
                        change_indicator = "↑"
                    elif value < previous:
                        change_indicator = "↓"
                    else:
                        change_indicator = "→"

                prev_str = f"{previous:.2f}" if previous is not None else "N/A"
                lines.append(
                    f"  {label:30s}: {value:7.2f} {unit:15s} {change_indicator} "
                    f"(prev: {prev_str}) [{reference}]"
                )

        lines.append(f"\nTotal indicators fetched: {len(indicators)}")

        # Show bond yields if available
        bond_yields = country_data.get("bond_yields", {})
        if bond_yields:
            lines.append(f"\nGovernment Bond Yields:")
            for maturity in ["2Y", "5Y", "10Y", "30Y"]:
                if maturity in bond_yields:
                    data = bond_yields[maturity]
                    yield_value = data["yield"]
                    day_change = data.get("day_change", 0)
                    date = data.get("date", "")

                    change_indicator = (
                        "↑"
                        if day_change and day_change > 0
                        else ("↓" if day_change and day_change < 0 else "→")
                    )
                    lines.append(
                        f"  {maturity:5s}: {yield_value:5.2f}% {change_indicator} "
                        f"(day: {day_change:+.3f}%) [{date}]"
                    )

        # Show industrial production if available
        industrial_production = country_data.get("industrial_production")
        if industrial_production:
            value = industrial_production.get("value")
            previous = industrial_production.get("previous")
            unit = industrial_production.get("unit", "")
            reference = industrial_production.get("reference", "")

            if value is not None:
                change_indicator = ""
                if previous is not None:
                    if value > previous:
                        change_indicator = "↑"
                    elif value < previous:
                        change_indicator = "↓"
                    else:
                        change_indicator = "→"

                prev_str = f"{previous:.2f}" if previous is not None else "N/A"
                lines.append(f"\nIndustrial Production:")
                lines.append(
                    f"  {value:7.2f} {unit:15s} {change_indicator} "
                    f"(prev: {prev_str}) [{reference}]"
                )

        # Show capacity utilization if available
        capacity_utilization = country_data.get("capacity_utilization")
        if capacity_utilization:
            value = capacity_utilization.get("value")
            previous = capacity_utilization.get("previous")
            unit = capacity_utilization.get("unit", "")
            reference = capacity_utilization.get("reference", "")

            if value is not None:
                change_indicator = ""
                if previous is not None:
                    if value > previous:
                        change_indicator = "↑"
                    elif value < previous:
                        change_indicator = "↓"
                    else:
                        change_indicator = "→"

                prev_str = f"{previous:.2f}" if previous is not None else "N/A"
                lines.append(f"\nCapacity Utilization:")
                lines.append(
                    f"  {value:7.2f} {unit:15s} {change_indicator} "
                    f"(prev: {prev_str}) [{reference}]"
                )

        lines.append("=" * 80)

        return "\n".join(lines)


if __name__ == "__main__":
    # Test scraper with portfolio countries
    scraper = TradingEconomicsIndicatorsScraper()
    scraper.get_country_indicators("UK")
    scraper.get_all_portfolio_countries()
