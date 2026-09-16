"""Moved to selfconcept.common.chat (it is not SOO-specific). This re-export
keeps the judge jobs queued before the move importable; delete once none of
them are pending."""

from selfconcept.common.chat import *  # noqa: F401,F403
