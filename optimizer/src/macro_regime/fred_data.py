#!/usr/bin/env python3
"""
FRED Market Indicators Fetcher
================================
Fetches critical market indicators from FRED and Yahoo Finance:
- VIX Volatility Index (Yahoo Finance)
- High Yield Credit Spreads (FRED - BAML OAS)

Note: Manufacturing PMI and Treasury Yields are now fetched from Trading Economics
for better data coverage and reliability.

Requires:
- pip install fredapi  # For FRED data
- pip install yfinance  # For VIX data

Get free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
"""

import os
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from fredapi import Fred

from src.yfinance import YFinanceClient

class FREDDataFetcher:
    """Fetch VIX and HY Spread from Yahoo Finance and FRED."""

    # FRED series IDs (simplified)
    SERIES_IDS = {
        'hy_spread': 'BAMLH0A0HYM2',           # ICE BofA US High Yield OAS
        'ig_spread': 'BAMLC0A0CM',             # ICE BofA Investment Grade OAS (optional)
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED data fetcher.

        Parameters
        ----------
        api_key : str, optional
            FRED API key. If not provided, looks for FRED_API_KEY env variable.
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY')

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
        except Exception as e:
            print(f"Error fetching {series_id}: {e}")
            return None


    def get_credit_spreads(self) -> Dict[str, Optional[float]]:
        """
        Get credit spreads (High Yield and Investment Grade).

        Returns
        -------
        dict
            {'hy_spread': float, 'ig_spread': float} in basis points
        """
        hy = self.get_latest_value(self.SERIES_IDS['hy_spread'])
        ig = self.get_latest_value(self.SERIES_IDS['ig_spread'])

        # FRED returns spreads already in percentage points
        return {
            'hy_spread': hy * 100 if hy else None,  # Convert to bps
            'ig_spread': ig * 100 if ig else None
        }

    def get_vix(self, lookback_days: int = 5) -> Optional[float]:
        """
        Get latest VIX (volatility index) from Yahoo Finance.

        VIX (CBOE Volatility Index) measures implied volatility of S&P 500 options
        over next 30 days. Known as the "fear gauge" - higher VIX = higher expected volatility.

        Parameters
        ----------
        lookback_days : int
            Number of recent days to fetch (uses latest close)

        Returns
        -------
        float or None
            Current VIX level, or None if fetch fails
        """
        try:
            # Get singleton YFinanceClient instance
            client = YFinanceClient.get_instance()

            # VIX with 5d period may only return 2-4 rows due to weekends/holidays
            # Set min_rows=1 since we only need the latest value
            vix_data = client.fetch_history("^VIX", period=f"{lookback_days}d", min_rows=1)

            if vix_data is None or vix_data.empty:
                print("Warning: Could not fetch VIX data from Yahoo Finance")
                return None

            # Use .item() to extract scalar value from Series
            current_vix = vix_data['Close'].iloc[-1].item() if len(vix_data) > 0 else None
            return current_vix

        except Exception as e:
            print(f"Warning: VIX fetch failed from Yahoo Finance: {e}")
            return None

    def get_all_indicators(self) -> Dict:
        """
        Fetch VIX and HY Spread.

        Returns
        -------
        dict
            VIX and credit spread data with regime signals
        """
        print("Fetching market indicators (VIX + HY Spread)...")

        # Get indicators
        credit_spreads = self.get_credit_spreads()
        vix = self.get_vix()

        # Calculate derived signals
        credit_signal = self._calculate_credit_signal(credit_spreads['hy_spread'])
        vix_signal = self.classify_vix_regime(vix)

        data = {
            # Core indicators
            'hy_spread': credit_spreads['hy_spread'],
            'ig_spread': credit_spreads['ig_spread'],
            'vix': vix,

            # Derived signals
            'credit_signal': credit_signal,
            'vix_signal': vix_signal,

            # Metadata
            'timestamp': datetime.now().isoformat(),
            'data_quality': self._assess_data_quality([credit_spreads['hy_spread'], vix])
        }

        return data

    def classify_vix_regime(self, vix_level: Optional[float]) -> str:
        """
        Classify market volatility regime based on VIX level.
        Based on portfolio_filter.py methodology.

        Parameters
        ----------
        vix_level : float or None
            Current VIX level

        Returns
        -------
        str
            Volatility regime: 'low_volatility', 'normal', 'elevated', 'high_volatility'
        """
        if vix_level is None:
            return 'normal'

        if vix_level < 15:
            return 'low_volatility'
        elif vix_level < 20:
            return 'normal'
        elif vix_level < 30:
            return 'elevated'
        else:
            return 'high_volatility'

    def _calculate_credit_signal(self, hy_spread: Optional[float]) -> str:
        """
        Convert HY spread to credit signal.
        Based on guide.md page 87 thresholds.

        Parameters
        ----------
        hy_spread : float or None
            High-yield credit spread in basis points

        Returns
        -------
        str
            Credit signal: 'risk_on', 'neutral', 'risk_off', 'severe_stress'
        """
        if hy_spread is None:
            return 'unknown'
        elif hy_spread < 350:
            return 'risk_on'
        elif hy_spread < 500:
            return 'neutral'
        elif hy_spread < 700:
            return 'risk_off'
        else:
            return 'severe_stress'

    def _assess_data_quality(self, values: list) -> str:
        """Check how many indicators are available."""
        available = sum(1 for v in values if v is not None)
        total = len(values)

        if available == total:
            return 'excellent'
        elif available >= total * 0.7:
            return 'good'
        elif available >= total * 0.5:
            return 'fair'
        else:
            return 'poor'


if __name__ == "__main__":
    # Load environment variables
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent.parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass

    # Test FRED fetcher
    fetcher = FREDDataFetcher()
    data = fetcher.get_all_indicators()

    print("\n" + "="*60)
    print("MARKET INDICATORS (VIX + CREDIT SPREADS)")
    print("="*60)

    print(f"\nCore Indicators:")
    print(f"  HY Credit Spread: {data['hy_spread']:.0f} bps" if data.get('hy_spread') else "  HY Credit Spread: N/A")
    print(f"  IG Credit Spread: {data['ig_spread']:.0f} bps" if data.get('ig_spread') else "  IG Credit Spread: N/A")
    print(f"  VIX: {data['vix']:.2f}" if data.get('vix') else "  VIX: N/A")

    print(f"\nRegime Signals:")
    print(f"  Credit: {data['credit_signal']}")
    print(f"  VIX: {data['vix_signal']}")

    print(f"\nData Quality: {data['data_quality']}")
    print("\nNote: Manufacturing PMI and Treasury Yields are now fetched from Trading Economics")
    print("="*60)
