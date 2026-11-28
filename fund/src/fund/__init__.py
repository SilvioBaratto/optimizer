"""portopt-fund: the deepagents bridge between LLM agents and the optimizer.

The agents prepare structured inputs and enums, the optimizer computes the candidate
portfolios (weights), and the user chooses one; orders are paper tickets only. This
package is the ONLY workspace member that imports both ``optimizer`` and
``portopt_db`` — it is the bridge. ``deepagents``/``langgraph`` live here and
nowhere else; the ingestion daemon and the DB layer stay clean, guarded by
import-scan hygiene tests on both sides.
"""

__version__ = "0.2.0"
