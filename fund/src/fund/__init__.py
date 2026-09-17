"""portopt-fund: the deepagents investment-fund bridge.

LLM chooses (structured inputs + enums), the optimizer computes (weights). This
package is the ONLY workspace member that imports both ``optimizer`` and
``portopt_db`` — it is the bridge. ``deepagents``/``langgraph`` live here and
nowhere else; the ingestion daemon and the DB layer stay clean, guarded by
import-scan hygiene tests on both sides.
"""

__version__ = "0.1.0"
