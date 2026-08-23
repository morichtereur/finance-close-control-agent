"""Shared infrastructure used by both process modules.

Configuration, error types, the provider factory, the append-only trace and the
audit log are deliberately module-agnostic: the close module and the I2P module
run on the same instrumentation, so a record from either reads the same way.
"""
