"""One module per pipeline stage. Built with /build-stage from specs/<stage>.md.

Rule (enforced by tests/test_architecture.py): stages import provider Protocols from
`hwva.providers.base`, never concrete providers — they get instances via `Context.providers`."""
