#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from fredapi import Fred

from optimizer.src.yfinance import YFinanceClient


class FREDDataFetcher:
    """Fetch VIX and HY Spread from Yahoo Finance and FRED."""

    # FRED series IDs (simplified)
    SERIES_IDS = {
        "hy_spread": "BAMLH0A0HYM2",  # ICE BofA US High Yield OAS
        "ig_spread": "BAMLC0A0CM",  # ICE BofA Investment Grade OAS (optional)
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED data fetcher.
        """
        self.api_key = api_key or os.getenv("FRED_API_KEY")

        if not self.api_key:
            raise ValueError(
                "FRED API key required. Get one at: https://fred.stlouisfed.org/docs/api/api_key.html\n"
                "Set via: export FRED_API_KEY='your_key' or pass to constructor"
            )

        self.fred = Fred(api_key=self.api_key)

    def get_latest_value(self, series_id: str) -> Optional[float]:
        """Get most recent value for a FRED series."""
        try:
            series = self.fred.get_series(series_id)
            if series is None or len(series) == 0:
                return None
            return float(series.dropna().iloc[-1])
        except Exception:
            return None

    def get_credit_spreads(self) -> Dict[str, Optional[float]]:
        """
        Get credit spreads (High Yield and Investment Grade).
        """
        hy = self.get_latest_value(self.SERIES_IDS["hy_spread"])
        ig = self.get_latest_value(self.SERIES_IDS["ig_spread"])

        # FRED returns spreads already in percentage points
        return {
            "hy_spread": hy * 100 if hy else None,  # Convert to bps
            "ig_spread": ig * 100 if ig else None,
        }

    def get_vix(self, lookback_days: int = 5) -> Optional[float]:
        """
        Get latest VIX (volatility index) from Yahoo Finance.
        """
        try:
            # Get singleton YFinanceClient instance
            client = YFinanceClient.get_instance()

            # VIX with 5d period may only return 2-4 rows due to weekends/holidays
            # Set min_rows=1 since we only need the latest value
            vix_data = client.fetch_history("^VIX", period=f"{lookback_days}d", min_rows=1)

            if vix_data is None or vix_data.empty:
                return None

            # Use .item() to extract scalar value from Series
            current_vix = vix_data["Close"].iloc[-1].item() if len(vix_data) > 0 else None
            return current_vix

        except Exception:
            return None

    def get_all_indicators(self) -> Dict:
        """
        Fetch VIX and HY Spread.
        """

        # Get indicators
        credit_spreads = self.get_credit_spreads()
        vix = self.get_vix()

        # Calculate derived signals
        credit_signal = self._calculate_credit_signal(credit_spreads["hy_spread"])
        vix_signal = self.classify_vix_regime(vix)

        data = {
            # Core indicators
            "hy_spread": credit_spreads["hy_spread"],
            "ig_spread": credit_spreads["ig_spread"],
            "vix": vix,
            # Derived signals
            "credit_signal": credit_signal,
            "vix_signal": vix_signal,
            # Metadata
            "timestamp": datetime.now().isoformat(),
            "data_quality": self._assess_data_quality([credit_spreads["hy_spread"], vix]),
        }

        return data

    def classify_vix_regime(self, vix_level: Optional[float]) -> str:
        """
        Classify market volatility regime based on VIX level.
        Based on portfolio_filter.py methodology.
        """
        if vix_level is None:
            return "normal"

        if vix_level < 15:
            return "low_volatility"
        elif vix_level < 20:
            return "normal"
        elif vix_level < 30:
            return "elevated"
        else:
            return "high_volatility"

    def _calculate_credit_signal(self, hy_spread: Optional[float]) -> str:
        """
        Convert HY spread to credit signal.
        Based on guide.md page 87 thresholds.
        """
        if hy_spread is None:
            return "unknown"
        elif hy_spread < 350:
            return "risk_on"
        elif hy_spread < 500:
            return "neutral"
        elif hy_spread < 700:
            return "risk_off"
        else:
            return "severe_stress"

    def _assess_data_quality(self, values: list) -> str:
        """Check how many indicators are available."""
        available = sum(1 for v in values if v is not None)
        total = len(values)

        if available == total:
            return "excellent"
        elif available >= total * 0.7:
            return "good"
        elif available >= total * 0.5:
            return "fair"
        else:
            return "poor"


if __name__ == "__main__":
    # Load environment variables
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).parent.parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass

    # Test FRED fetcher
    fetcher = FREDDataFetcher()
    data = fetcher.get_all_indicators()
