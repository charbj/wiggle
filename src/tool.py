# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

""" Import all dependencies """
from chimerax.core.tools import ToolInstance
from chimerax.core import logger
from Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np
from functools import partial, lru_cache
import os.path
import time
import subprocess

class wiggle(ToolInstance):

    SESSION_ENDURING = False    # Does this instance persist when session closes
    SESSION_SAVE = True         # We do save/restore in sessions
    help = "help:user/tools/tutorial.html"
                                # Let ChimeraX know about our help page
    def __init__(self, session, tool_name):
        # 'session'   - chimerax.core.session.Session instance
        # 'tool_name' - string

        # Initialize base class.
        super().__init__(session, tool_name)

        self.display_name = "WIGGLE - 0.2.2 dev"

        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        # self.tool_window.fill_context_menu = self.fill_context_menu

        from .wiggle import Wiggle
        w = Wiggle(session)
        layout = w.chimera_initialise()

        # Set the layout as the contents of our window
        self.tool_window.ui_area.setLayout(layout)
        self.tool_window.ui_area.contextMenuEvent = lambda event : None

        # Show the window on the user-preferred side of the ChimeraX
        # main window
        self.tool_window.manage(None, fixed_size=True)

    # def fill_context_menu(self, menu, x, y):
    #     # Add any tool-specific items to the given context menu (a QMenu instance).
    #     # The menu will then be automatically filled out with generic tool-related actions
    #     # (e.g. Hide Tool, Help, Dockable Tool, etc.)
    #     #
    #     # The x,y args are the x() and y() values of QContextMenuEvent, in the rare case
    #     # where the items put in the menu depends on where in the tool interface the menu
    #     # was raised.
    #     if 0 < x < 795 and 78 < y < 748:
    #         pass
    #     else:
    #         from Qt.QtWidgets import QAction
    #         clear_action = QAction("Clear", menu)
    #         clear_action.triggered.connect(lambda *args: self.line_edit.clear())
    #         menu.addAction(clear_action)

    def take_snapshot(self, session, flags):
        return {
            'version': 1,
            'current text': self.line_edit.text()
        }

    @classmethod
    def restore_snapshot(class_obj, session, data):
        # Instead of using a fixed string when calling the constructor below, we could
        # have saved the tool name during take_snapshot() (from self.tool_name, inherited
        # from ToolInstance) and used that saved tool name.  There are pros and cons to
        # both approaches.
        inst = class_obj(session, "wiggle")
        inst.line_edit.setText(data['current text'])
        return inst
