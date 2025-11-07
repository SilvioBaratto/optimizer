#!/usr/bin/env python3
"""
Regime Transition Tracker
==========================
Tracks business cycle regime over time and detects transitions.

Now uses database for historical regime tracking instead of CSV files.
Maintains backward compatibility with CSV-based tracking.
"""

import pandas as pd
import sys
from datetime import datetime
from typing import Dict, Optional, Union
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.database import DatabaseManager
from app.models.macro_regime import CountryRegimeAssessment
from sqlalchemy import select, desc


class RegimeTracker:
    """
    Track business cycle regimes over time using database.

    This class now queries the database for historical regime data
    instead of maintaining a local CSV file.
    """

    def __init__(self, country: str = 'USA', use_database: bool = True):
        """
        Initialize regime tracker.

        Parameters
        ----------
        country : str
            Country code to track (default: 'USA')
        use_database : bool
            If True, uses database for tracking (default)
            If False, falls back to CSV file (legacy mode)
        """
        self.country = country
        self.use_database = use_database

        if use_database:
            self.db_manager = DatabaseManager()
            try:
                self.db_manager.initialize()
            except Exception as e:
                print(f"Warning: Database initialization failed: {e}")
                print("Falling back to CSV-based tracking")
                self.use_database = False

        if not self.use_database:
            # Legacy CSV mode
            script_dir = Path(__file__).parent
            output_dir = script_dir.parent / 'outputs' / 'regime_tracking'
            output_dir.mkdir(parents=True, exist_ok=True)
            self.history_file = output_dir / f'regime_history_{country}.csv'
            self.history_df = self._load_history_csv()

    def _load_history_csv(self) -> pd.DataFrame:
        """Load historical regime data from CSV (legacy mode)."""
        if self.history_file.exists():
            return pd.read_csv(self.history_file, parse_dates=['timestamp'])
        else:
            return pd.DataFrame(columns=[
                'timestamp', 'regime', 'confidence',
                'recession_risk_6m', 'recession_risk_12m',
                'ism_pmi', 'yield_curve', 'hy_spread',
                'gdp_forecast', 'inflation_forecast'
            ])

    def _load_history_from_db(self) -> pd.DataFrame:
        """Load historical regime data from database."""
        with self.db_manager.get_session() as session:
            stmt = (
                select(CountryRegimeAssessment)
                .where(CountryRegimeAssessment.country == self.country)
                .order_by(CountryRegimeAssessment.assessment_timestamp)
            )
            results = session.execute(stmt).scalars().all()

            if not results:
                return pd.DataFrame(columns=[
                    'timestamp', 'regime', 'confidence',
                    'recession_risk_6m', 'recession_risk_12m'
                ])

            records = []
            for assessment in results:
                records.append({
                    'timestamp': assessment.assessment_timestamp,
                    'regime': assessment.regime,
                    'confidence': assessment.confidence,
                    'recession_risk_6m': assessment.recession_risk_6m,
                    'recession_risk_12m': assessment.recession_risk_12m
                })

            return pd.DataFrame(records)

    def add_assessment(self, assessment, ilsole_data: Dict, fred_data: Dict):
        """
        Add new regime assessment to history (legacy CSV mode only).

        NOTE: When using database mode, assessments are saved directly
        by database_saver.py, so this method is not needed.

        Parameters
        ----------
        assessment : CycleAssessment
            Current cycle assessment
        ilsole_data : dict
            Il Sole data for context
        fred_data : dict
            FRED data for context
        """
        if self.use_database:
            # In database mode, data is saved by database_saver
            # This method is just for compatibility
            return

        # Legacy CSV mode
        new_row = {
            'timestamp': pd.Timestamp(assessment.timestamp),
            'regime': assessment.regime,
            'confidence': assessment.confidence,
            'recession_risk_6m': assessment.recession_risk_6m,
            'recession_risk_12m': assessment.recession_risk_12m,
            'ism_pmi': fred_data.get('ism_pmi'),
            'yield_curve': fred_data.get('yield_curve_2s10s'),
            'hy_spread': fred_data.get('hy_spread'),
            'gdp_forecast': ilsole_data.get('forecast', {}).get('gdp_growth_6m'),
            'inflation_forecast': ilsole_data.get('forecast', {}).get('inflation_6m')
        }

        self.history_df = pd.concat([
            self.history_df,
            pd.DataFrame([new_row])
        ], ignore_index=True)

        self._save_history_csv()

    def _save_history_csv(self):
        """Save history to CSV (legacy mode)."""
        self.history_df.to_csv(self.history_file, index=False)

    def detect_transition(self) -> Optional[Dict]:
        """
        Detect if regime has changed from last assessment.

        Returns
        -------
        dict or None
            Transition details if regime changed, None otherwise
        """
        if self.use_database:
            history_df = self._load_history_from_db()
        else:
            history_df = self.history_df

        if len(history_df) < 2:
            return None

        current = history_df.iloc[-1]
        previous = history_df.iloc[-2]

        if current['regime'] != previous['regime']:
            days_since_last = (current['timestamp'] - previous['timestamp']).days

            return {
                'transition_detected': True,
                'from_regime': previous['regime'],
                'to_regime': current['regime'],
                'transition_date': current['timestamp'].strftime('%Y-%m-%d'),
                'days_since_last_transition': days_since_last,
                'confidence': current['confidence'],
                'alert_level': 'HIGH' if current['confidence'] > 0.8 else 'MEDIUM'
            }

        return None

    def get_regime_duration(self) -> Optional[int]:
        """Get number of days in current regime."""
        if self.use_database:
            history_df = self._load_history_from_db()
        else:
            history_df = self.history_df

        if len(history_df) < 2:
            return None

        current_regime = history_df.iloc[-1]['regime']

        # Find when current regime started
        for i in range(len(history_df) - 2, -1, -1):
            if history_df.iloc[i]['regime'] != current_regime:
                start_date = history_df.iloc[i + 1]['timestamp']
                current_date = history_df.iloc[-1]['timestamp']
                return (current_date - start_date).days

        # Regime has been constant throughout history
        if len(history_df) > 0:
            start_date = history_df.iloc[0]['timestamp']
            current_date = history_df.iloc[-1]['timestamp']
            return (current_date - start_date).days

        return None

    def get_summary(self) -> Dict:
        """Get summary of regime history."""
        if self.use_database:
            history_df = self._load_history_from_db()
        else:
            history_df = self.history_df

        if len(history_df) == 0:
            return {'status': 'no_data'}

        current = history_df.iloc[-1]
        transition = self.detect_transition()
        duration = self.get_regime_duration()

        # Calculate regime statistics
        regime_counts = history_df['regime'].value_counts().to_dict()
        avg_confidence = history_df['confidence'].mean()

        return {
            'current_regime': current['regime'],
            'current_confidence': current['confidence'],
            'regime_duration_days': duration,
            'total_observations': len(history_df),
            'first_observation': history_df.iloc[0]['timestamp'].strftime('%Y-%m-%d'),
            'latest_observation': current['timestamp'].strftime('%Y-%m-%d'),
            'transition_detected': transition is not None,
            'transition_details': transition,
            'regime_distribution': regime_counts,
            'average_confidence': avg_confidence,
            'recession_risk': {
                '6month': current['recession_risk_6m'],
                '12month': current['recession_risk_12m']
            }
        }

    def get_recent_history(self, n: int = 10) -> pd.DataFrame:
        """Get last N regime assessments."""
        if self.use_database:
            history_df = self._load_history_from_db()
        else:
            history_df = self.history_df
        return history_df.tail(n)


if __name__ == "__main__":
    # Test tracker
    print("\n" + "="*80)
    print("REGIME TRACKER - TEST MODE")
    print("="*80)
    print("\nTo test the regime tracker, use run_regime_analysis.py:")
    print("  python run_regime_analysis.py --country USA")
    print("\nThe tracker stores regime history in:")
    print("  ../outputs/regime_tracking/regime_history.csv")
    print("="*80)
