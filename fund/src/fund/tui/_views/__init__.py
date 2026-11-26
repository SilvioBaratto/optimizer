"""The shell's views, one module per view.

Import them through :mod:`fund.tui.views`, which re-exports every name. At module
scope these import only Textual, the stdlib and the agent-stack-free
``fund.observe`` read model, so ``import fund.tui.app`` stays agent-stack-light.
"""
