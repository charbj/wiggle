import numpy as np
import time
import pyqtgraph as pg
pg.setConfigOptions(useOpenGL=True)

import mrcfile
import torch
import argparse, textwrap
import os, shutil


from Qt import QtCore, QtGui, QtWidgets
from functools import partial
from math import floor

np.seterr(divide='ignore', invalid='ignore')
lastClicked = []

class FileIO:
    def __init__(self, ui):
        self.ui = ui
        self.inputType = {
            0: 'cryodrgn',
            1: 'cryosparc_3dva',
            2: 'cryosparc_3dflx'
        }

    def browse_file(self, comment, button, type):
        if button == self.ui.pushButton_9:
            fileName, _ = QtWidgets.QFileDialog.getOpenFileNames(None, comment, "", type,
                                                                options=QtWidgets.QFileDialog.DontUseNativeDialog)
            fileName.sort()
        else:
            fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, comment, "", type,
                                                                    options=QtWidgets.QFileDialog.DontUseNativeDialog)

        if fileName:
            #cryoSPARC 3D flex
            if button == self.ui.pushButton_3:
                self.ui.lineEdit.setText(fileName)
            elif button == self.ui.pushButton_4:
                self.ui.lineEdit_2.setText(fileName)

            #cryoSPARC 3DVA
            elif button == self.ui.pushButton_5:
                self.ui.lineEdit_13.setText(fileName)
            elif button == self.ui.pushButton_9:
                self.ui.lineEdit_14.setText(','.join(fileName))
            elif button == self.ui.pushButton_13:
                self.ui.lineEdit_15.setText(fileName)
                self.ui.comboBox_3.setEnabled(True)

            #cryoSPARC single file format
            elif button == self.ui.pushButton_16:
                self.ui.lineEdit_16.setText(fileName)
                self.ui.comboBox_3.setEnabled(True)

            #cryoDRGN inputs
            elif button == self.ui.pushButton_6: #To do: make this a requirement for volumiser()
                self.ui.lineEdit_6.setText(fileName)
                # self.w.config = fileName
            elif button == self.ui.pushButton_7: #To do: make a requirement for volumiser()
                self.ui.lineEdit_5.setText(fileName)
                # self.w.weights = fileName
            elif button == self.ui.pushButton_8: #z_space minimal requirement for dimensionality reduction.
                self.ui.lineEdit_4.setText(fileName)
                self.ui.comboBox_3.setEnabled(True)

            # cryoDRGN single file format
            elif button == self.ui.pushButton_21:
                self.ui.lineEdit_18.setText(fileName)
                self.ui.comboBox_3.setEnabled(True)

            self.input_type()

    def input_type(self):
        type = self.inputType[self.ui.comboBox.currentIndex()]
        single_input = self.ui.checkBox_14.isChecked()
        self.ui.doubleSpinBox_10.setEnabled(False)
        self.ui.label_33.setEnabled(False)
        self.ui.radioButton_3.setEnabled(True)
        self.ui.radioButton_4.setEnabled(True)
        self.ui.pushButton_15.setEnabled(True)

        if single_input:
            if type == 'cryodrgn':
                self.ui.stackedWidget.setCurrentIndex(4)
                self.ui.stackedWidget_4.setCurrentIndex(1)
                self.ui.data_path = [self.ui.lineEdit_18.text()]
                self.ui.radioButton_3.setEnabled(False)
                self.ui.radioButton_4.setEnabled(False)
                self.ui.pushButton_15.setEnabled(False)
            elif type == 'cryosparc_3dva':
                self.ui.stackedWidget.setCurrentIndex(3)
                self.ui.data_path = [self.ui.lineEdit_16.text()]
                self.ui.stackedWidget_4.setCurrentIndex(0)
            elif type == 'cryosparc_3dflx':
                self.ui.stackedWidget.setCurrentIndex(5)
                self.ui.stackedWidget_4.setCurrentIndex(0)
            self.ui.w.graphicsView.setGeometry(QtCore.QRect(0, 26, 796, 721))  #0, 80, 796, 666
        else:
            if type == 'cryodrgn':
                self.ui.stackedWidget.setCurrentIndex(0)
                self.ui.stackedWidget_4.setCurrentIndex(1)
                self.ui.data_path = [self.ui.lineEdit_6.text(), self.ui.lineEdit_5.text(), self.ui.lineEdit_4.text()]
                self.ui.doubleSpinBox_10.setEnabled(True)
                self.ui.label_33.setEnabled(True)
                self.ui.radioButton_3.setEnabled(False)
                self.ui.radioButton_4.setEnabled(False)
                self.ui.pushButton_15.setEnabled(False)
            elif type == 'cryosparc_3dva':
                self.ui.stackedWidget.setCurrentIndex(1)
                self.ui.stackedWidget_4.setCurrentIndex(0)
                self.ui.data_path = [self.ui.lineEdit_15.text(), self.ui.lineEdit_13.text(), self.ui.lineEdit_14.text()]
            elif type == 'cryosparc_3dflx':
                pass
            self.ui.w.graphicsView.setGeometry(QtCore.QRect(0, 80, 796, 666))

    def generate_csg(self, name, meta, meta_size, passthrough, passthrough_size):
        from datetime import datetime
        sparc_dict = {
            'created': None,
            'group':
                {
                    'description': 'Subset of particles that were processed, including alignments',
                    'name': 'Wiggle subset',
                    'type': 'particle'
                 },
            'results':
                {
                    'alignments3D':
                        {
                        'metafile': ">empty",
                        'num_items': None,
                        'type': 'particle.alignments3D'
                        },
                    'blob': {
                        'metafile': ">empty",
                        'num_items': None,
                        'type': 'particle.blob'
                    },
                    'ctf': {
                        'metafile': ">empty",
                        'num_items': None,
                        'type': 'particle.ctf'
                    },
                },
            'version': 'v3.2.0'
        }
        sparc_dict['created'] = datetime.now()
        sparc_dict['group']['name'] = name
        sparc_dict['results']['alignments3D']['metafile'] = ''.join(('>',str(meta)))
        sparc_dict['results']['ctf']['metafile'] = ''.join(('>',str(meta)))
        sparc_dict['results']['blob']['metafile'] = ''.join(('>',str(passthrough)))
        sparc_dict['results']['alignments3D']['num_items'] = meta_size
        sparc_dict['results']['ctf']['num_items'] = meta_size
        sparc_dict['results']['blob']['num_items'] = passthrough_size
        return sparc_dict

    # NEEDS ADDRESSING # CBJ
    def export_roi(self):
        import yaml
        if self.ui.radioButton_3.isChecked():

            #Try the _maskAt method with a QPointF or QRectF?? Might need rejig

            if self.ui.checkBox_7.isChecked():
                select = [pt.data() for subPlot in self.ui.w.scatterPlots
                          if subPlot.isVisible()
                          for pt in subPlot.points()
                          if not self.ui.w.ROIs.mapToItem(subPlot, self.ui.w.ROIs.shape()).contains(pt.pos())]
            else:
                select = [pt.data() for subPlot in self.ui.w.scatterPlots
                          if subPlot.isVisible()
                          for pt in subPlot.points()
                          if self.ui.w.ROIs.mapToItem(subPlot, self.ui.w.ROIs.shape()).contains(pt.pos())]

            output_dir = QtWidgets.QFileDialog.getExistingDirectory(None, 'Create an output directory where results will be saved', "")
            if output_dir == "":
                print("User cancelled. Exiting...")
                return
            passthrough, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Select cryoSPARC passthrough file', "")
            if passthrough == "":
                print("User cancelled. Exiting...")
                return
            proj_dir = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select cryoSPARC project directory',"")
            if proj_dir == "":
                print("User cancelled. Exiting...")
                return

            passthrough_meta = np.load(passthrough)
            size_passthrough = passthrough_meta.shape[0]
            particles = ui.w.data[3][select]
            output = 'wiggle_lasso_particles'

            with open(''.join((os.path.join(output_dir, output), '.cs')), 'wb') as f:
                np.save(f, particles, allow_pickle=False)

            path, fileName = os.path.split(output)
            path2, fileName2 = os.path.split(passthrough)
            csg_output = self.generate_csg(output,
                              ''.join((fileName,'.cs')),
                              particles.shape[0],
                              fileName2,
                              size_passthrough)

            with open(''.join((os.path.join(output_dir, output), '.csg')), 'w') as f:
                yaml.dump(csg_output, f, default_flow_style=False)


            set_of_files = set(passthrough_meta['blob/path'].astype(str))
            for f in set_of_files:
                f_out = f.split('/')[-1]
                pth_out = os.path.join(*f.split('/')[:-1])
                if not os.path.exists(os.path.join(output_dir, pth_out)):
                    os.makedirs(os.path.join(output_dir, pth_out), exist_ok=True)
                os.symlink(os.path.join(proj_dir,f), os.path.join(output_dir,pth_out,f_out))

            shutil.copy(passthrough, os.path.join(output_dir,fileName2))

            print("A total of " + str(len(select)) + " particles were exported as cryoSPARC format. "
                                                     "\n -------------------------")

    # NEEDS ADDRESSING # CBJ
    def export_clusters(self):
        import yaml
        if self.ui.radioButton_4.isChecked():
            try:
                self.ui.w.labels
            except:
                print("You must run clustering before you can export observation clusters "
                                        "\n -------------------------")
            else:
                labels = np.array(self.ui.w.labels)
                output_dir = QtWidgets.QFileDialog.getExistingDirectory(None,'Output directory to save output',"")
                if output_dir == "":
                    print("User cancelled. Exiting...")
                    return
                passthrough, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Select cryoSPARC passthrough file', "")
                if passthrough == "":
                    print("User cancelled. Exiting...")
                    return
                proj_dir = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select cryoSPARC project directory', "")
                if proj_dir == "":
                    print("User cancelled. Exiting...")
                    return
                passthrough_meta = np.load(passthrough)
                size_passthrough = passthrough_meta.shape[0]
                path2, fileName2 = os.path.split(passthrough)

                for label in set(self.ui.w.labels):
                    _indices = np.where(labels==label)[0]

                    particles = ui.w.data[3][_indices]
                    output = ''.join(('wiggle_cluster_', str(label)))

                    with open(''.join((os.path.join(output_dir, output), '.cs')), 'wb') as f:
                        np.save(f, particles, allow_pickle=False)

                    path, fileName = os.path.split(output)

                    csg_output = self.generate_csg(output,
                                                   ''.join((fileName, '.cs')),
                                                   particles.shape[0],
                                                   fileName2,
                                                   size_passthrough)

                    with open(''.join((os.path.join(output_dir, output), '.csg')), 'w') as f:
                        yaml.dump(csg_output, f, default_flow_style=False)

                    print("\t Done 'cluster_" + str(label) + "'. Contains " + str(_indices.shape[0]) + " particles.")

                set_of_files = set(passthrough_meta['blob/path'].astype(str))
                for f in set_of_files:
                    f_out = f.split('/')[-1]
                    pth_out = os.path.join(*f.split('/')[:-1])
                    if not os.path.exists(os.path.join(output_dir, pth_out)):
                        os.makedirs(os.path.join(output_dir, pth_out), exist_ok=True)
                    os.symlink(os.path.join(proj_dir, f), os.path.join(output_dir, pth_out, f_out))

                shutil.copy(passthrough, os.path.join(output_dir, fileName2))

                print("-------------------------")

class Ui_MainWindow(object):
    def __init__(self, session):
        self.state = 'idle'
        self.inputType = {
            0 : 'cryodrgn',
            1 : 'cryosparc_3dva',
            2 : 'cryosparc_3dflx'
        }
        # self.previous_embedding_type = None
        # self.embedType = {
        #     0: 'UMAP',
        #     1: 'PCA',
        #     2: 'tSNE',
        #     3: 'PHATE',
        #     4: 'CVAE'
        # }
        self.resolution = {
            0: 96,
            1: 128,
            2: 156,
            3: 196,
            4: 256
        }
        self.data_path = ''
        # self.subset_state = None
        # self.previous_data = None
        self.wiggle = session

    def status(self, kill: bool):
        if self.state == 'busy':
            self.state = ''
        elif self.state == '':
            self.state = 'busy'
        elif self.state == 'idle':
            self.state = 'busy'

        if not kill:
            self.statusTimer.start(700)
            self.statusBar.setStyleSheet("background-color : pink")
        else:
            self.state = 'idle'
            self.statusBar.setStyleSheet("background-color: rgb(239, 239, 239);")
            self.statusTimer.stop()

        self.statusBar.showMessage("Status: " + str(self.state))

    def reportProgress(self, ETA, t0, init: bool, ping: bool, kill: bool):
        if init:
            self.ETA = ETA
            self.t0 = t0
            progText = 'Estimated time for completion: ' + str(self.ETA/60)[0:5] + ' minutes.'
            print(progText)

        if ping:
            try:
                ETA = self.ETA
                t0 = self.t0
                if (time.time() - t0 < ETA):
                    progress = 100 * ((time.time() - t0) / ETA)
                    self.progressBar_7.setValue(int(progress))
                    self.timer.start(50)
                    QtCore.QCoreApplication.processEvents()
            except:
                print('no good - problem in reportProgress function')

        if kill:
            self.progressBar_7.setValue(int(100))
            self.timer.stop()

    def reset(self):
        '''
        ###########################################################################################################
        Disable GUI until user re-runs dimensionality reduction OR changes their minds
        ###########################################################################################################
        '''
        if self.comboBox_2.isEnabled():
            self.comboBox_2.setDisabled(True)
            self.comboBox_2.setCurrentIndex(0)
            self.stackedWidget_3.setCurrentIndex(0)

        if self.checkBox_5.isEnabled():
            self.checkBox_5.setDisabled(True)
            self.checkBox_5.setChecked(False)

        # if self.previous_data is not None:
        #     if len(self.previous_data) == int(self.lineEdit_3.text()):
        #         self.subset_state = False
        #     else:
        #         self.subset_state = True

    def run_embedding(self):
        from .deps.embedding import Embedder
        self.progressBar_7.reset()
        self.pushButton_19.setEnabled(False)
        # Step 2: Create a QThread object
        self.thread = QtCore.QThread()
        # Step 3: Create a worker object

        # self.current_embedding_type = self.embedType[self.comboBox_3.currentIndex()]
        #
        # if self.current_embedding_type is not self.previous_embedding_type:
        self.e = Embedder(
            ui,
            self.comboBox_3.currentIndex(),
            self.data_path,
            self.lineEdit_3.text(),
            (self.checkBox_12.isChecked() and self.lineEdit_3.text().isnumeric())
        )

        # Step 4: Move worker to the thread
        self.e.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.e.run_embedding)
        self.e.exit.connect(self.thread.quit)
        self.e.exit.connect(self.e.deleteLater)
        self.e.finished.connect(self.thread.quit)
        self.e.finished.connect(self.e.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.e.progress.connect(self.reportProgress)
        self.e.status.connect(self.status)
        # self.e.msg.connect(self.textBrowser.append)
        # Step 6: Start the thread
        self.thread.start()

        # Final resets
        def update_interactive_window():
            self.w.embedding = self.e.embedding
            self.state = False
            self.w.data = self.e.data
            self.w.apix = self.e.apix
            self.w.currentZind = None
            self.w.lastClicked = None
            if self.doubleSpinBox_10.isEnabled():
                print("got called and apix now True")
                self.w.user_apix = True

        self.e.finished.connect(update_interactive_window)

        self.e.finished.connect(
            lambda: self.comboBox_2.setEnabled(True)
        )

        self.e.finished.connect(
            lambda : self.progressBar_7.setValue(100))

        self.e.finished.connect(
            lambda: self.pushButton_19.setEnabled(True)
        )

        self.e.exit.connect(
            lambda: self.pushButton_19.setEnabled(True)
        )

        self.e.finished.connect(
            lambda: self.w.modify_state('space', True)
        )

        self.e.finished.connect(self.w.initialise_volume_engine)
        self.e.finished.connect(self.w.plotArea.autoRange)
        # self.previous_embedding_type = self.current_embedding_type

    def run_clustering(self):
        from .deps.clustering import Clusterer

        self.progressBar_7.reset()
        self.pushButton_23.setEnabled(False)
        # Step 2: Create a QThread object
        self.thread2 = QtCore.QThread()
        # Step 3: Create a worker object
        self.c = Clusterer(self.e.data[2], self.spinBox_2.value(), self.comboBox_4.currentIndex())
        # Step 4: Move worker to the thread
        self.c.moveToThread(self.thread2)
        # Step 5: Connect signals and slots
        self.thread2.started.connect(self.c.run_clustering)
        self.c.finished.connect(self.thread2.quit)
        self.c.finished.connect(self.c.deleteLater)
        self.thread2.finished.connect(self.thread2.deleteLater)
        self.c.progress.connect(self.reportProgress)
        # self.c.msg.connect(self.textBrowser.append)
        self.c.status.connect(self.status)
        # Step 6: Start the thread
        self.thread2.start()

        # Final resets
        def update_interactive_window():
            self.w.labels = self.c.labels
            self.checkBox_5.setEnabled(True)
            self.checkBox_5.setChecked(True)
            self.w.lastClicked = None
            if self.checkBox_6.isChecked():
                self.w.volumiser_by_cluster()

        self.c.finished.connect(
            lambda : update_interactive_window()
        )

        self.c.finished.connect(
            lambda : self.w.modify_state('space', self.checkBox_5.isChecked())
        )

        self.c.finished.connect(
            lambda: self.progressBar_7.setValue(100)
        )

        self.c.finished.connect(
            lambda: self.pushButton_23.setEnabled(True)
        )

    def run_MEP(self):
        from .deps.path_inference_dijkstra import Path
        self.progressBar_7.reset()
        self.pushButton_10.setEnabled(False)
        # Step 2: Create a QThread object
        self.thread3 = QtCore.QThread()
        # Step 3: Create a worker object
        self.p = Path(
            self.w.embedding,
            float(self.lineEdit_8.text()),
            int(self.comboBox_7.currentText()),
            int(self.comboBox_8.currentText()),
            self.resolution[self.comboBox_9.currentIndex()],
            int(self.lineEdit_7.text()),
            int(self.lineEdit_17.text()),
            float(self.lineEdit_10.text()),
            float(self.lineEdit_11.text()),
        )
        # Step 4: Move worker to the thread
        self.p.moveToThread(self.thread3)
        # Step 5: Connect signals and slots
        self.p.finished.connect(self.thread3.quit)
        self.p.finished.connect(self.p.deleteLater)
        self.thread3.finished.connect(self.thread3.deleteLater)
        self.p.progress.connect(self.reportProgress)
        # self.p.msg.connect(self.textBrowser.append)
        self.p.status.connect(self.status)

        self.thread3.started.connect(self.p.costFuncApproach)
        # Step 6: Start the thread
        self.thread3.start()

        # Final resets
        def update_interactive_window():
            self.w.MEP_TrajLists = self.p.trajectories

        self.p.finished.connect(
            lambda : update_interactive_window()
        )

        self.p.finished.connect(
            lambda: self.w.MEP_path_display()
        )

        self.p.finished.connect(
            lambda : self.w.modify_state('MEP', self.checkBox_13.isChecked())
        )

        self.p.finished.connect(
            lambda: self.progressBar_7.setValue(100)
        )

        self.p.finished.connect(
            lambda: self.pushButton_10.setEnabled(True)
        )

    def setupUi(self, MainWindow):
        '''
        ###########################################################################################################
        Build the main window and define all attributes for Qt
        :param MainWindow:
        :return:
        ###########################################################################################################
        '''
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1008, 772)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(1008, 772))
        MainWindow.setMaximumSize(QtCore.QSize(1008, 772))

        MainWindow.setWindowOpacity(1.0)
        MainWindow.setWhatsThis("")
        MainWindow.setStyleSheet("")
        MainWindow.setDocumentMode(False)

        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)


        self.timer = QtCore.QTimer()
        self.onlyInt = QtGui.QIntValidator()
        self.onlyFloat = QtGui.QDoubleValidator()
        self.statusTimer = QtCore.QTimer()
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        # self.textBrowser.setGeometry(QtCore.QRect(0, 750, 796, 96))
        # self.textBrowser.setObjectName("textBrowser")
        # self.textBrowser_3 = QtWidgets.QTextBrowser(self.centralwidget)
        # self.textBrowser_3.setEnabled(False)
        # self.textBrowser_3.setGeometry(QtCore.QRect(800, 780, 206, 66))
        # self.textBrowser_3.setAcceptDrops(True)
        # self.textBrowser_3.setStyleSheet("background-color: rgb(255, 255, 255);")
        # self.textBrowser_3.setObjectName("textBrowser_3")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setGeometry(QtCore.QRect(-20, -5, 1031, 861))
        # self.graphicsView_2.setPalette(palette)
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(0, 0, 801, 91))
        self.stackedWidget.setObjectName("stackedWidget")
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.pushButton_8 = QtWidgets.QPushButton(self.page_2)
        self.pushButton_8.setGeometry(QtCore.QRect(0, 30, 196, 21))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_6 = QtWidgets.QPushButton(self.page_2)
        self.pushButton_6.setGeometry(QtCore.QRect(0, 5, 196, 21))
        self.pushButton_6.setObjectName("pushButton_6")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.page_2)
        self.lineEdit_5.setGeometry(QtCore.QRect(200, 55, 596, 21))
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.pushButton_7 = QtWidgets.QPushButton(self.page_2)
        self.pushButton_7.setGeometry(QtCore.QRect(0, 55, 196, 21))
        self.pushButton_7.setObjectName("pushButton_7")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.page_2)
        self.lineEdit_6.setGeometry(QtCore.QRect(200, 5, 596, 21))
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.page_2)
        self.lineEdit_4.setGeometry(QtCore.QRect(200, 30, 596, 21))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.stackedWidget.addWidget(self.page_2)
        self.page_4 = QtWidgets.QWidget()
        self.page_4.setObjectName("page_4")
        self.lineEdit_13 = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_13.setGeometry(QtCore.QRect(200, 5, 596, 21))
        self.lineEdit_13.setObjectName("lineEdit_13")
        self.pushButton_5 = QtWidgets.QPushButton(self.page_4)
        self.pushButton_5.setGeometry(QtCore.QRect(0, 5, 196, 21))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_9 = QtWidgets.QPushButton(self.page_4)
        self.pushButton_9.setGeometry(QtCore.QRect(0, 30, 196, 21))
        self.pushButton_9.setObjectName("pushButton_9")
        self.lineEdit_14 = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_14.setGeometry(QtCore.QRect(200, 30, 596, 21))
        self.lineEdit_14.setObjectName("lineEdit_14")
        self.pushButton_13 = QtWidgets.QPushButton(self.page_4)
        self.pushButton_13.setGeometry(QtCore.QRect(0, 55, 196, 21))
        self.pushButton_13.setObjectName("pushButton_13")
        self.lineEdit_15 = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_15.setGeometry(QtCore.QRect(200, 55, 596, 21))
        self.lineEdit_15.setObjectName("lineEdit_15")
        self.stackedWidget.addWidget(self.page_4)

        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.pushButton_3 = QtWidgets.QPushButton(self.page)
        self.pushButton_3.setEnabled(False)
        self.pushButton_3.setGeometry(QtCore.QRect(0, 5, 196, 21))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.page)
        self.pushButton_4.setEnabled(False)
        self.pushButton_4.setGeometry(QtCore.QRect(0, 30, 196, 21))
        self.pushButton_4.setObjectName("pushButton_4")
        self.lineEdit = QtWidgets.QLineEdit(self.page)
        self.lineEdit.setEnabled(False)
        self.lineEdit.setGeometry(QtCore.QRect(200, 5, 596, 21))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.page)
        self.lineEdit_2.setEnabled(False)
        self.lineEdit_2.setGeometry(QtCore.QRect(200, 30, 596, 21))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.label_41 = QtWidgets.QLabel(self.page)
        self.label_41.setGeometry(QtCore.QRect(5, 55, 276, 21))
        self.label_41.setFont(font)
        self.label_41.setObjectName("label_41")
        self.stackedWidget.addWidget(self.page)
        self.page_5 = QtWidgets.QWidget()
        self.page_5.setObjectName("page_5")
        self.pushButton_16 = QtWidgets.QPushButton(self.page_5)
        self.pushButton_16.setGeometry(QtCore.QRect(0, 5, 196, 21))
        self.pushButton_16.setObjectName("pushButton_16")
        self.lineEdit_16 = QtWidgets.QLineEdit(self.page_5)
        self.lineEdit_16.setGeometry(QtCore.QRect(200, 5, 596, 21))
        self.lineEdit_16.setObjectName("lineEdit_16")
        self.stackedWidget.addWidget(self.page_5)
        self.page_6 = QtWidgets.QWidget()
        self.page_6.setObjectName("page_6")
        self.pushButton_21 = QtWidgets.QPushButton(self.page_6)
        self.pushButton_21.setGeometry(QtCore.QRect(0, 5, 196, 21))
        self.pushButton_21.setObjectName("pushButton_21")
        self.lineEdit_18 = QtWidgets.QLineEdit(self.page_6)
        self.lineEdit_18.setGeometry(QtCore.QRect(200, 5, 596, 21))
        self.lineEdit_18.setObjectName("lineEdit_18")
        self.stackedWidget.addWidget(self.page_6)

        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(805, 120, 191, 23))
        self.comboBox.setObjectName("comboBox")
        # palette = QtGui.QPalette()
        # brush = QtGui.QBrush(QtGui.QColor(176, 154, 225))
        # brush.setStyle(QtCore.Qt.SolidPattern)
        # palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        # self.comboBox.setPalette(palette)
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")

        self.comboBox_3 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_3.setEnabled(False)
        self.comboBox_3.setGeometry(QtCore.QRect(805, 182, 146, 23))
        self.comboBox_3.setObjectName("comboBox_3")
        # palette = QtGui.QPalette()
        # brush = QtGui.QBrush(QtGui.QColor(243, 129, 129))
        # brush.setStyle(QtCore.Qt.SolidPattern)
        # palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        # brush = QtGui.QBrush(QtGui.QColor(243, 129, 129))
        # brush.setStyle(QtCore.Qt.SolidPattern)
        # palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        # self.comboBox_3.setPalette(palette)
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")

        self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_2.setEnabled(False)
        self.comboBox_2.setGeometry(QtCore.QRect(805, 232, 191, 23))
        self.comboBox_2.setToolTip("")
        self.comboBox_2.setObjectName("comboBox_2")
        # palette = QtGui.QPalette()
        # brush = QtGui.QBrush(QtGui.QColor(142, 218, 248))
        # brush.setStyle(QtCore.Qt.SolidPattern)
        # palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        # self.comboBox_2.setPalette(palette)
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")

        self.stackedWidget_3 = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget_3.setEnabled(True)
        self.stackedWidget_3.setGeometry(QtCore.QRect(795, 260, 221, 381))
        self.stackedWidget_3.setObjectName("stackedWidget_3")
        self.page_12 = QtWidgets.QWidget()
        self.page_12.setObjectName("page_12")
        self.checkBox = QtWidgets.QCheckBox(self.page_12)
        self.checkBox.setGeometry(QtCore.QRect(10, 25, 146, 21))
        self.checkBox.setChecked(True)
        self.checkBox.setObjectName("checkBox")
        self.label_8 = QtWidgets.QLabel(self.page_12)
        self.label_8.setGeometry(QtCore.QRect(10, -10, 221, 31))
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.line_4 = QtWidgets.QFrame(self.page_12)
        self.line_4.setGeometry(QtCore.QRect(10, 10, 201, 20))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.pushButton_27 = QtWidgets.QPushButton(self.page_12)
        self.pushButton_27.setEnabled(False)
        self.pushButton_27.setGeometry(QtCore.QRect(150, 25, 56, 21))
        self.pushButton_27.setObjectName("pushButton_27")
        self.checkBox_3 = QtWidgets.QCheckBox(self.page_12)
        self.checkBox_3.setGeometry(QtCore.QRect(10, 50, 181, 21))
        self.checkBox_3.setChecked(True)
        self.checkBox_3.setObjectName("checkBox_3")
        self.label_25 = QtWidgets.QLabel(self.page_12)
        self.label_25.setEnabled(True)
        self.label_25.setGeometry(QtCore.QRect(10, 80, 126, 21))
        self.label_25.setObjectName("label_25")
        self.doubleSpinBox_9 = QtWidgets.QDoubleSpinBox(self.page_12)
        self.doubleSpinBox_9.setEnabled(True)
        self.doubleSpinBox_9.setGeometry(QtCore.QRect(120, 79, 51, 22))
        self.doubleSpinBox_9.setAcceptDrops(False)
        self.doubleSpinBox_9.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_9.setMinimum(-1.0)
        self.doubleSpinBox_9.setMaximum(50.0)
        self.doubleSpinBox_9.setSingleStep(0.05)
        self.doubleSpinBox_9.setProperty("value", -1.0)
        self.doubleSpinBox_9.setObjectName("doubleSpinBox_9")
        self.label_26 = QtWidgets.QLabel(self.page_12)
        self.label_26.setEnabled(True)
        self.label_26.setGeometry(QtCore.QRect(10, 105, 136, 22))
        self.label_26.setObjectName("label_26")
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.page_12)
        self.doubleSpinBox.setGeometry(QtCore.QRect(160, 105, 51, 22))
        self.doubleSpinBox.setFrame(True)
        self.doubleSpinBox.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox.setMinimum(0.01)
        self.doubleSpinBox.setMaximum(10)
        self.doubleSpinBox.setSingleStep(0.01)
        self.doubleSpinBox.setProperty("value", 0.1)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.pushButton_25 = QtWidgets.QPushButton(self.page_12)
        self.pushButton_25.setEnabled(True)
        self.pushButton_25.setGeometry(QtCore.QRect(171, 80, 40, 20))
        self.pushButton_25.setObjectName("pushButton_25")
        self.spinBox_2B = QtWidgets.QSpinBox(self.page_12)
        self.spinBox_2B.setGeometry(QtCore.QRect(160, 128, 51, 22))
        self.spinBox_2B.setFrame(True)
        self.spinBox_2B.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinBox_2B.setMinimum(-1)
        self.spinBox_2B.setMaximum(1024)
        self.spinBox_2B.setProperty("value", -1)
        self.spinBox_2B.setObjectName("spinBox_2")

        self.label_33 = QtWidgets.QLabel(self.page_12)
        self.label_33.setEnabled(True)
        self.label_33.setGeometry(QtCore.QRect(60, 171, 96, 22))
        self.label_33.setObjectName("label_33")
        self.doubleSpinBox_10 = QtWidgets.QDoubleSpinBox(self.page_12)
        self.doubleSpinBox_10.setEnabled(True)
        self.doubleSpinBox_10.setGeometry(QtCore.QRect(160, 173, 51, 22))
        self.doubleSpinBox_10.setAcceptDrops(False)
        self.doubleSpinBox_10.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_10.setDecimals(3)
        self.doubleSpinBox_10.setMinimum(-1.0)
        self.doubleSpinBox_10.setMaximum(50.0)
        self.doubleSpinBox_10.setSingleStep(0.001)
        self.doubleSpinBox_10.setProperty("value", -1.0)
        self.doubleSpinBox_10.setObjectName("doubleSpinBox_10")

        # self.label_27 = QtWidgets.QLabel(self.page_12)
        # self.label_27.setEnabled(True)
        # self.label_27.setGeometry(QtCore.QRect(10, 152, 136, 22))
        # self.label_27.setObjectName("label_27")
        self.checkBox_4 = QtWidgets.QCheckBox(self.page_12)
        self.checkBox_4.setGeometry(QtCore.QRect(10, 195, 121, 21))
        self.checkBox_4.setChecked(False)
        self.checkBox_4.setObjectName("checkBox_4")

        self.stackedWidget_4 = QtWidgets.QStackedWidget(self.page_12)
        self.stackedWidget_4.setGeometry(QtCore.QRect(10, 125, 146, 46))
        self.stackedWidget_4.setObjectName("stackedWidget_4")
        self.page_11 = QtWidgets.QWidget()
        self.page_11.setObjectName("page_11")
        self.label_28 = QtWidgets.QLabel(self.page_11)
        self.label_28.setEnabled(True)
        self.label_28.setGeometry(QtCore.QRect(30, 24, 116, 22))
        self.label_28.setObjectName("label_28")
        self.label_27 = QtWidgets.QLabel(self.page_11)
        self.label_27.setEnabled(True)
        self.label_27.setGeometry(QtCore.QRect(0, 2, 136, 22))
        self.label_27.setObjectName("label_27")
        self.stackedWidget_4.addWidget(self.page_11)
        self.page_14 = QtWidgets.QWidget()
        self.page_14.setObjectName("page_14")
        self.label_31 = QtWidgets.QLabel(self.page_14)
        self.label_31.setEnabled(True)
        self.label_31.setGeometry(QtCore.QRect(45, 24, 101, 22))
        self.label_31.setObjectName("label_31")
        self.label_32 = QtWidgets.QLabel(self.page_14)
        self.label_32.setEnabled(True)
        self.label_32.setGeometry(QtCore.QRect(0, 2, 136, 22))
        self.label_32.setObjectName("label_32")
        self.stackedWidget_4.addWidget(self.page_14)


        self.spinBox_9 = QtWidgets.QSpinBox(self.page_12)
        self.spinBox_9.setGeometry(QtCore.QRect(160, 150, 51, 22))
        self.spinBox_9.setFrame(True)
        self.spinBox_9.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinBox_9.setMinimum(-1)
        self.spinBox_9.setMaximum(1024)
        self.spinBox_9.setProperty("value", -1)
        self.spinBox_9.setObjectName("spinBox_9")
        # self.label_28 = QtWidgets.QLabel(self.page_12)
        # self.label_28.setEnabled(True)
        # self.label_28.setGeometry(QtCore.QRect(10, 174, 136, 22))
        # self.label_28.setObjectName("label_28")
        self.stackedWidget_3.addWidget(self.page_12)


        # self.page_12 = QtWidgets.QWidget()
        # self.page_12.setObjectName("page_12")
        # self.checkBox = QtWidgets.QCheckBox(self.page_12)
        # self.checkBox.setGeometry(QtCore.QRect(10, 25, 146, 21))
        # self.checkBox.setChecked(True)
        # self.checkBox.setObjectName("checkBox")
        # self.label_8 = QtWidgets.QLabel(self.page_12)
        # self.label_8.setGeometry(QtCore.QRect(10, -10, 221, 31))
        # self.label_8.setFont(font)
        # self.label_8.setObjectName("label_8")
        # self.line_4 = QtWidgets.QFrame(self.page_12)
        # self.line_4.setGeometry(QtCore.QRect(10, 10, 201, 20))
        # self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        # self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        # self.line_4.setObjectName("line_4")
        # self.pushButton_27 = QtWidgets.QPushButton(self.page_12)
        # self.pushButton_27.setEnabled(False)
        # self.pushButton_27.setGeometry(QtCore.QRect(110, 45, 86, 26))
        # self.pushButton_27.setObjectName("pushButton_27")
        # self.checkBox_3 = QtWidgets.QCheckBox(self.page_12)
        # self.checkBox_3.setGeometry(QtCore.QRect(10, 75, 181, 21))
        # self.checkBox_3.setChecked(False)
        # self.checkBox_3.setObjectName("checkBox_3")
        # self.label_25 = QtWidgets.QLabel(self.page_12)
        # self.label_25.setEnabled(True)
        # self.label_25.setGeometry(QtCore.QRect(10, 100, 126, 21))
        # self.label_25.setObjectName("label_25")
        # self.doubleSpinBox_9 = QtWidgets.QDoubleSpinBox(self.page_12)
        # self.doubleSpinBox_9.setEnabled(True)
        # self.doubleSpinBox_9.setGeometry(QtCore.QRect(120, 104, 51, 22))
        # self.doubleSpinBox_9.setAcceptDrops(False)
        # self.doubleSpinBox_9.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        # self.doubleSpinBox_9.setMinimum(-1.0)
        # self.doubleSpinBox_9.setMaximum(50.0)
        # self.doubleSpinBox_9.setSingleStep(0.05)
        # self.doubleSpinBox_9.setProperty("value", -1.0)
        # self.doubleSpinBox_9.setObjectName("doubleSpinBox_9")
        # self.label_26 = QtWidgets.QLabel(self.page_12)
        # self.label_26.setEnabled(True)
        # self.label_26.setGeometry(QtCore.QRect(10, 128, 136, 26))
        # self.label_26.setObjectName("label_26")
        # self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.page_12)
        # self.doubleSpinBox.setGeometry(QtCore.QRect(160, 130, 51, 22))
        # self.doubleSpinBox.setFrame(True)
        # self.doubleSpinBox.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        # self.doubleSpinBox.setMinimum(0.1)
        # self.doubleSpinBox.setMaximum(500.0)
        # self.doubleSpinBox.setSingleStep(0.1)
        # self.doubleSpinBox.setObjectName("doubleSpinBox")
        # self.pushButton_25 = QtWidgets.QPushButton(self.page_12)
        # self.pushButton_25.setEnabled(False)
        # self.pushButton_25.setGeometry(QtCore.QRect(171, 105, 40, 20))
        # self.pushButton_25.setObjectName("pushButton_25")
        # self.spinBox_2B = QtWidgets.QSpinBox(self.page_12)
        # self.spinBox_2B.setGeometry(QtCore.QRect(160, 152, 51, 22))
        # self.spinBox_2B.setFrame(True)
        # self.spinBox_2B.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        # self.spinBox_2B.setMinimum(-1)
        # self.spinBox_2B.setMaximum(1024)
        # self.spinBox_2B.setProperty("value", -1)
        # self.spinBox_2B.setObjectName("spinBox_2B")
        # self.label_27 = QtWidgets.QLabel(self.page_12)
        # self.label_27.setEnabled(True)
        # self.label_27.setGeometry(QtCore.QRect(10, 155, 136, 21))
        # self.label_27.setObjectName("label_27")
        # self.checkBox_4 = QtWidgets.QCheckBox(self.page_12)
        # self.checkBox_4.setGeometry(QtCore.QRect(10, 180, 121, 21))
        # self.checkBox_4.setChecked(False)
        # self.checkBox_4.setObjectName("checkBox_4")
        # self.stackedWidget_3.addWidget(self.page_12)

        self.page_16 = QtWidgets.QWidget()
        self.page_16.setObjectName("page_16")
        self.label_7 = QtWidgets.QLabel(self.page_16)
        self.label_7.setGeometry(QtCore.QRect(10, -10, 171, 31))
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_11 = QtWidgets.QLabel(self.page_16)
        self.label_11.setGeometry(QtCore.QRect(15, 60, 121, 31))
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.spinBox_2 = QtWidgets.QSpinBox(self.page_16)
        self.spinBox_2.setGeometry(QtCore.QRect(140, 65, 61, 21))
        self.spinBox_2.setProperty("value", 10)
        self.spinBox_2.setObjectName("spinBox_2")
        self.pushButton_23 = QtWidgets.QPushButton(self.page_16)
        self.pushButton_23.setGeometry(QtCore.QRect(135, 235, 75, 23))
        self.pushButton_23.setObjectName("pushButton_23")
        # self.pushButton_24 = QtWidgets.QPushButton(self.page_16)
        # self.pushButton_24.setGeometry(QtCore.QRect(135, 235, 75, 23))
        # self.pushButton_24.setObjectName("pushButton_24")
        self.line_3 = QtWidgets.QFrame(self.page_16)
        self.line_3.setGeometry(QtCore.QRect(10, 10, 201, 20))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.line_5 = QtWidgets.QFrame(self.page_16)
        self.line_5.setGeometry(QtCore.QRect(5, 220, 201, 20))
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.checkBox_5 = QtWidgets.QCheckBox(self.page_16)
        self.checkBox_5.setGeometry(QtCore.QRect(5, 95, 176, 21))
        self.checkBox_5.setObjectName("checkBox_5")
        self.checkBox_6 = QtWidgets.QCheckBox(self.page_16)
        self.checkBox_6.setGeometry(QtCore.QRect(5, 115, 231, 31))
        self.checkBox_6.setObjectName("checkBox_6")
        self.comboBox_4 = QtWidgets.QComboBox(self.page_16)
        self.comboBox_4.setEnabled(True)
        self.comboBox_4.setGeometry(QtCore.QRect(5, 25, 201, 26))
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_5 = QtWidgets.QComboBox(self.page_16)
        self.comboBox_5.setEnabled(True)
        self.comboBox_5.setGeometry(QtCore.QRect(5, 170, 201, 26))
        self.comboBox_5.setObjectName("comboBox_5")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.checkBox_2 = QtWidgets.QCheckBox(self.page_16)
        self.checkBox_2.setEnabled(False)
        self.checkBox_2.setGeometry(QtCore.QRect(5, 145, 221, 21))
        self.checkBox_2.setWhatsThis("")
        self.checkBox_2.setAccessibleDescription("")
        self.checkBox_2.setObjectName("checkBox_2")
        self.stackedWidget_3.addWidget(self.page_16)

        self.page_13 = QtWidgets.QWidget()
        self.page_13.setObjectName("page_13")
        self.pushButton = QtWidgets.QPushButton(self.page_13)
        self.pushButton.setEnabled(False)
        self.pushButton.setGeometry(QtCore.QRect(150, 200, 61, 23))
        self.pushButton.setObjectName("pushButton")
        self.label_5 = QtWidgets.QLabel(self.page_13)
        self.label_5.setGeometry(QtCore.QRect(10, -10, 171, 31))
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.pushButton_2 = QtWidgets.QPushButton(self.page_13)
        self.pushButton_2.setEnabled(False)
        self.pushButton_2.setGeometry(QtCore.QRect(80, 200, 61, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_6 = QtWidgets.QLabel(self.page_13)
        self.label_6.setGeometry(QtCore.QRect(20, 175, 131, 21))
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_10 = QtWidgets.QLabel(self.page_13)
        self.label_10.setGeometry(QtCore.QRect(95, 120, 41, 31))
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.line = QtWidgets.QFrame(self.page_13)
        self.line.setGeometry(QtCore.QRect(10, 265, 196, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.page_13)
        self.line_2.setGeometry(QtCore.QRect(5, 10, 201, 20))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.checkBox_21 = QtWidgets.QCheckBox(self.page_13)
        self.checkBox_21.setGeometry(QtCore.QRect(5, 150, 166, 22))
        self.checkBox_21.setChecked(False)
        self.checkBox_21.setObjectName("checkBox_21")
        self.checkBox_8 = QtWidgets.QCheckBox(self.page_13)
        self.checkBox_8.setGeometry(QtCore.QRect(5, 280, 156, 22))
        self.checkBox_8.setChecked(True)
        self.checkBox_8.setObjectName("checkBox_8")
        self.checkBox_10 = QtWidgets.QCheckBox(self.page_13)
        self.checkBox_10.setGeometry(QtCore.QRect(5, 300, 156, 22))
        self.checkBox_10.setObjectName("checkBox_10")
        self.line_7 = QtWidgets.QFrame(self.page_13)
        self.line_7.setGeometry(QtCore.QRect(5, 125, 81, 20))
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.line_8 = QtWidgets.QFrame(self.page_13)
        self.line_8.setGeometry(QtCore.QRect(120, 125, 81, 20))
        self.line_8.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.lineEdit_12 = QtWidgets.QLineEdit(self.page_13)
        self.lineEdit_12.setEnabled(True)
        self.lineEdit_12.setGeometry(QtCore.QRect(160, 175, 46, 21))
        self.lineEdit_12.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_12.setReadOnly(True)
        self.lineEdit_12.setObjectName("lineEdit_12")
        self.label_20 = QtWidgets.QLabel(self.page_13)
        self.label_20.setGeometry(QtCore.QRect(20, 25, 121, 21))
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.spinBox_3 = QtWidgets.QSpinBox(self.page_13)
        self.spinBox_3.setGeometry(QtCore.QRect(160, 25, 46, 21))
        self.spinBox_3.setAlignment(QtCore.Qt.AlignCenter)
        self.spinBox_3.setProperty("value", 15)
        self.spinBox_3.setObjectName("spinBox_3")
        self.pushButton_17 = QtWidgets.QPushButton(self.page_13)
        self.pushButton_17.setEnabled(False)
        self.pushButton_17.setGeometry(QtCore.QRect(5, 200, 61, 23))
        self.pushButton_17.setObjectName("pushButton_17")
        self.pushButton_18 = QtWidgets.QPushButton(self.page_13)
        self.pushButton_18.setEnabled(False)
        self.pushButton_18.setGeometry(QtCore.QRect(145, 280, 61, 23))
        self.pushButton_18.setObjectName("pushButton_18")
        self.label_21 = QtWidgets.QLabel(self.page_13)
        self.label_21.setGeometry(QtCore.QRect(10, 225, 116, 21))
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.comboBox_10 = QtWidgets.QComboBox(self.page_13)
        self.comboBox_10.setEnabled(False)
        self.comboBox_10.setGeometry(QtCore.QRect(5, 245, 166, 26))
        self.comboBox_10.setObjectName("comboBox_10")
        self.pushButton_20 = QtWidgets.QPushButton(self.page_13)
        self.pushButton_20.setEnabled(False)
        self.pushButton_20.setGeometry(QtCore.QRect(175, 245, 36, 26))
        self.pushButton_20.setObjectName("pushButton_20")
        self.checkBox_19 = QtWidgets.QCheckBox(self.page_13)
        self.checkBox_19.setEnabled(True)
        self.checkBox_19.setGeometry(QtCore.QRect(5, 320, 101, 22))
        self.checkBox_19.setCheckable(True)
        self.checkBox_19.setChecked(True)
        self.checkBox_19.setObjectName("checkBox_19")
        self.checkBox_20 = QtWidgets.QCheckBox(self.page_13)
        self.checkBox_20.setEnabled(True)
        self.checkBox_20.setGeometry(QtCore.QRect(5, 340, 101, 22))
        self.checkBox_20.setCheckable(True)
        self.checkBox_20.setObjectName("checkBox_20")
        self.pushButton_22 = QtWidgets.QPushButton(self.page_13)
        self.pushButton_22.setGeometry(QtCore.QRect(175, 65, 36, 26))
        self.pushButton_22.setObjectName("pushButton_22")
        self.comboBox_12 = QtWidgets.QComboBox(self.page_13)
        self.comboBox_12.setGeometry(QtCore.QRect(5, 65, 166, 26))
        self.comboBox_12.setObjectName("comboBox_12")
        self.label_29 = QtWidgets.QLabel(self.page_13)
        self.label_29.setGeometry(QtCore.QRect(10, 45, 146, 21))
        self.label_29.setFont(font)
        self.label_29.setObjectName("label_29")
        self.pushButton_28 = QtWidgets.QPushButton(self.page_13)
        self.pushButton_28.setGeometry(QtCore.QRect(5, 95, 61, 23))
        self.pushButton_28.setObjectName("pushButton_28")
        self.stackedWidget_3.addWidget(self.page_13)

        # self.page_13 = QtWidgets.QWidget()
        # self.page_13.setObjectName("page_13")
        # self.pushButton = QtWidgets.QPushButton(self.page_13)
        # self.pushButton.setGeometry(QtCore.QRect(55, 235, 75, 23))
        # self.pushButton.setObjectName("pushButton")
        # self.label_5 = QtWidgets.QLabel(self.page_13)
        # self.label_5.setGeometry(QtCore.QRect(10, -10, 221, 31))
        # self.label_5.setFont(font)
        # self.label_5.setObjectName("label_5")
        # self.pushButton_2 = QtWidgets.QPushButton(self.page_13)
        # self.pushButton_2.setGeometry(QtCore.QRect(135, 235, 75, 23))
        # self.pushButton_2.setObjectName("pushButton_2")
        # self.label_6 = QtWidgets.QLabel(self.page_13)
        # self.label_6.setGeometry(QtCore.QRect(20, 150, 131, 21))
        # self.label_6.setFont(font)
        # self.label_6.setObjectName("label_6")
        # self.label_10 = QtWidgets.QLabel(self.page_13)
        # self.label_10.setGeometry(QtCore.QRect(95, 95, 41, 31))
        # self.label_10.setFont(font)
        # self.label_10.setObjectName("label_10")
        # self.line = QtWidgets.QFrame(self.page_13)
        # self.line.setGeometry(QtCore.QRect(10, 220, 196, 20))
        # self.line.setFrameShape(QtWidgets.QFrame.HLine)
        # self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        # self.line.setObjectName("line")
        # self.line_2 = QtWidgets.QFrame(self.page_13)
        # self.line_2.setGeometry(QtCore.QRect(10, 10, 201, 20))
        # self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        # self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        # self.line_2.setObjectName("line_2")
        #
        # self.spinBox_4 = QtWidgets.QSpinBox(self.page_13)
        # self.spinBox_4.setGeometry(QtCore.QRect(165, 25, 46, 21))
        # self.spinBox_4.setAlignment(QtCore.Qt.AlignCenter)
        # self.spinBox_4.setObjectName("spinBox_4")
        # self.radioButton = QtWidgets.QRadioButton(self.page_13)
        # self.radioButton.setGeometry(QtCore.QRect(5, 25, 166, 21))
        # self.radioButton.setLayoutDirection(QtCore.Qt.LeftToRight)
        # self.radioButton.setChecked(True)
        # self.radioButton.setAutoExclusive(True)
        # self.radioButton.setObjectName("radioButton")
        #
        # self.radioButton_2 = QtWidgets.QRadioButton(self.page_13)
        # self.radioButton_2.setGeometry(QtCore.QRect(5, 125, 166, 22))
        # self.radioButton_2.setObjectName("radioButton_2")
        # self.checkBox_8 = QtWidgets.QCheckBox(self.page_13)
        # self.checkBox_8.setGeometry(QtCore.QRect(5, 180, 156, 22))
        # self.checkBox_8.setObjectName("checkBox_8")
        # self.checkBox_10 = QtWidgets.QCheckBox(self.page_13)
        # self.checkBox_10.setGeometry(QtCore.QRect(5, 200, 156, 22))
        # self.checkBox_10.setObjectName("checkBox_10")
        # self.line_7 = QtWidgets.QFrame(self.page_13)
        # self.line_7.setGeometry(QtCore.QRect(5, 100, 81, 20))
        # self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        # self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        # self.line_7.setObjectName("line_7")
        # self.line_8 = QtWidgets.QFrame(self.page_13)
        # self.line_8.setGeometry(QtCore.QRect(120, 100, 81, 20))
        # self.line_8.setFrameShape(QtWidgets.QFrame.HLine)
        # self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        # self.line_8.setObjectName("line_8")
        # self.lineEdit_12 = QtWidgets.QLineEdit(self.page_13)
        # self.lineEdit_12.setEnabled(True)
        # self.lineEdit_12.setGeometry(QtCore.QRect(145, 150, 61, 21))
        # self.lineEdit_12.setAlignment(QtCore.Qt.AlignCenter)
        # self.lineEdit_12.setReadOnly(True)
        # self.lineEdit_12.setObjectName("lineEdit_12")
        #
        # self.label_20 = QtWidgets.QLabel(self.page_13)
        # self.label_20.setGeometry(QtCore.QRect(40, 50, 121, 21))
        # self.label_20.setFont(font)
        # self.label_20.setObjectName("label_20")
        # self.spinBox_3 = QtWidgets.QSpinBox(self.page_13)
        # self.spinBox_3.setGeometry(QtCore.QRect(165, 50, 46, 21))
        # self.spinBox_3.setAlignment(QtCore.Qt.AlignCenter)
        # self.spinBox_3.setProperty("value", 8)
        # self.spinBox_3.setObjectName("spinBox_3")
        #
        # self.pushButton_17 = QtWidgets.QPushButton(self.page_13)
        # self.pushButton_17.setGeometry(QtCore.QRect(55, 265, 75, 23))
        # self.pushButton_17.setObjectName("pushButton_17")
        # self.pushButton_18 = QtWidgets.QPushButton(self.page_13)
        # self.pushButton_18.setGeometry(QtCore.QRect(135, 265, 75, 23))
        # self.pushButton_18.setObjectName("pushButton_18")
        # self.label_21 = QtWidgets.QLabel(self.page_13)
        # self.label_21.setGeometry(QtCore.QRect(10, 290, 181, 21))
        # self.label_21.setFont(font)
        # self.label_21.setObjectName("label_21")
        # self.comboBox_10 = QtWidgets.QComboBox(self.page_13)
        # self.comboBox_10.setEnabled(False)
        # self.comboBox_10.setGeometry(QtCore.QRect(5, 310, 166, 26))
        # self.comboBox_10.setObjectName("comboBox_10")
        # self.pushButton_20 = QtWidgets.QPushButton(self.page_13)
        # self.pushButton_20.setEnabled(False)
        # self.pushButton_20.setGeometry(QtCore.QRect(175, 310, 36, 26))
        # self.pushButton_20.setObjectName("pushButton_20")
        # self.checkBox_19 = QtWidgets.QCheckBox(self.page_13)
        # self.checkBox_19.setEnabled(False)
        # self.checkBox_19.setGeometry(QtCore.QRect(3, 335, 101, 22))
        # self.checkBox_19.setChecked(True)
        # self.checkBox_19.setObjectName("checkBox_19")
        # self.checkBox_20 = QtWidgets.QCheckBox(self.page_13)
        # self.checkBox_20.setEnabled(False)
        # self.checkBox_20.setGeometry(QtCore.QRect(105, 335, 101, 22))
        # self.checkBox_20.setChecked(False)
        # self.checkBox_20.setObjectName("checkBox_20")
        # self.pushButton_22 = QtWidgets.QPushButton(self.page_13)
        # self.pushButton_22.setEnabled(True)
        # self.pushButton_22.setGeometry(QtCore.QRect(175, 75, 36, 26))
        # self.pushButton_22.setObjectName("pushButton_22")
        # self.stackedWidget_3.addWidget(self.page_13)


        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.label_9 = QtWidgets.QLabel(self.page_3)
        self.label_9.setGeometry(QtCore.QRect(10, -10, 181, 31))
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.line_6 = QtWidgets.QFrame(self.page_3)
        self.line_6.setGeometry(QtCore.QRect(5, 10, 201, 20))
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.line_12 = QtWidgets.QFrame(self.page_3)
        self.line_12.setGeometry(QtCore.QRect(10, 235, 201, 20))
        self.line_12.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_12.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_12.setObjectName("line_12")
        self.pushButton_10 = QtWidgets.QPushButton(self.page_3)
        self.pushButton_10.setGeometry(QtCore.QRect(55, 250, 75, 23))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_11 = QtWidgets.QPushButton(self.page_3)
        self.pushButton_11.setGeometry(QtCore.QRect(135, 250, 75, 23))
        self.pushButton_11.setObjectName("pushButton_11")
        self.comboBox_6 = QtWidgets.QComboBox(self.page_3)
        self.comboBox_6.setEnabled(False)
        self.comboBox_6.setGeometry(QtCore.QRect(5, 300, 166, 26))
        self.comboBox_6.setObjectName("comboBox_6")
        self.line_13 = QtWidgets.QFrame(self.page_3)
        self.line_13.setGeometry(QtCore.QRect(10, 270, 201, 20))
        self.line_13.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_13.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_13.setObjectName("line_13")
        self.comboBox_7 = QtWidgets.QComboBox(self.page_3)
        self.comboBox_7.setEnabled(True)
        self.comboBox_7.setGeometry(QtCore.QRect(155, 25, 51, 21))
        self.comboBox_7.setToolTip("")
        self.comboBox_7.setObjectName("comboBox_7")
        self.comboBox_7.addItem("")
        self.comboBox_7.addItem("")
        self.comboBox_7.addItem("")
        self.comboBox_7.addItem("")
        self.comboBox_8 = QtWidgets.QComboBox(self.page_3)
        self.comboBox_8.setEnabled(True)
        self.comboBox_8.setGeometry(QtCore.QRect(155, 50, 51, 21))
        self.comboBox_8.setToolTip("")
        self.comboBox_8.setObjectName("comboBox_8")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_9 = QtWidgets.QComboBox(self.page_3)
        self.comboBox_9.setEnabled(True)
        self.comboBox_9.setGeometry(QtCore.QRect(90, 75, 116, 21))
        self.comboBox_9.setToolTip("")
        self.comboBox_9.setObjectName("comboBox_9")
        self.comboBox_9.addItem("")
        self.comboBox_9.addItem("")
        self.comboBox_9.addItem("")
        self.comboBox_9.addItem("")
        self.comboBox_9.addItem("")
        self.label_3 = QtWidgets.QLabel(self.page_3)
        self.label_3.setGeometry(QtCore.QRect(10, 25, 126, 21))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.page_3)
        self.label_4.setGeometry(QtCore.QRect(10, 50, 126, 21))
        self.label_4.setObjectName("label_4")
        self.label_13 = QtWidgets.QLabel(self.page_3)
        self.label_13.setGeometry(QtCore.QRect(10, 75, 71, 21))
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.page_3)
        self.label_14.setGeometry(QtCore.QRect(10, 100, 136, 21))
        self.label_14.setObjectName("label_14")
        self.lineEdit_7 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_7.setEnabled(True)
        self.lineEdit_7.setGeometry(QtCore.QRect(145, 100, 61, 21))
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.lineEdit_7.setValidator(self.onlyInt)
        self.lineEdit_8 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_8.setEnabled(True)
        self.lineEdit_8.setGeometry(QtCore.QRect(145, 120, 61, 21))
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.lineEdit_8.setValidator(self.onlyFloat)
        self.label_15 = QtWidgets.QLabel(self.page_3)
        self.label_15.setGeometry(QtCore.QRect(10, 140, 86, 21))
        self.label_15.setObjectName("label_15")
        self.lineEdit_9 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_9.setEnabled(True)
        self.lineEdit_9.setGeometry(QtCore.QRect(145, 140, 61, 21))
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.label_16 = QtWidgets.QLabel(self.page_3)
        self.label_16.setGeometry(QtCore.QRect(10, 120, 86, 21))
        self.label_16.setObjectName("label_16")
        self.lineEdit_10 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_10.setEnabled(True)
        self.lineEdit_10.setGeometry(QtCore.QRect(145, 160, 61, 21))
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.lineEdit_10.setValidator(self.onlyFloat)
        self.label_17 = QtWidgets.QLabel(self.page_3)
        self.label_17.setGeometry(QtCore.QRect(10, 160, 111, 21))
        self.label_17.setObjectName("label_17")
        self.lineEdit_11 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_11.setEnabled(True)
        self.lineEdit_11.setGeometry(QtCore.QRect(145, 180, 61, 21))
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.lineEdit_11.setValidator(self.onlyFloat)
        self.label_18 = QtWidgets.QLabel(self.page_3)
        self.label_18.setGeometry(QtCore.QRect(10, 180, 86, 21))
        self.label_18.setObjectName("label_18")
        self.label_19 = QtWidgets.QLabel(self.page_3)
        self.label_19.setGeometry(QtCore.QRect(10, 280, 181, 21))
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.checkBox_13 = QtWidgets.QCheckBox(self.page_3)
        self.checkBox_13.setEnabled(False)
        self.checkBox_13.setGeometry(QtCore.QRect(10, 325, 156, 22))
        self.checkBox_13.setObjectName("checkBox_13")
        self.checkBox_13.setChecked(True)
        self.label_40 = QtWidgets.QLabel(self.page_3)
        self.label_40.setGeometry(QtCore.QRect(10, 200, 116, 21))
        self.label_40.setObjectName("label_40")
        self.lineEdit_17 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_17.setEnabled(True)
        self.lineEdit_17.setGeometry(QtCore.QRect(145, 200, 61, 21))
        self.lineEdit_17.setObjectName("lineEdit_17")
        self.lineEdit_17.setValidator(self.onlyInt)
        self.checkBox_11 = QtWidgets.QCheckBox(self.page_3)
        self.checkBox_11.setEnabled(False)
        self.checkBox_11.setGeometry(QtCore.QRect(10, 345, 156, 22))
        self.checkBox_11.setObjectName("checkBox_11")
        self.pushButton_12 = QtWidgets.QPushButton(self.page_3)
        self.pushButton_12.setEnabled(False)
        self.pushButton_12.setGeometry(QtCore.QRect(175, 300, 36, 26))
        self.pushButton_12.setObjectName("pushButton_12")
        self.checkBox_18 = QtWidgets.QCheckBox(self.page_3)
        self.checkBox_18.setGeometry(QtCore.QRect(10, 220, 196, 22))
        self.checkBox_18.setObjectName("checkBox_18")
        self.stackedWidget_3.addWidget(self.page_3)
        self.page_15 = QtWidgets.QWidget()
        self.page_15.setObjectName("page_15")
        self.label_12 = QtWidgets.QLabel(self.page_15)
        self.label_12.setGeometry(QtCore.QRect(10, -10, 156, 31))
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.line_11 = QtWidgets.QFrame(self.page_15)
        self.line_11.setGeometry(QtCore.QRect(5, 10, 201, 20))
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")
        self.radioButton_3 = QtWidgets.QRadioButton(self.page_15)
        self.radioButton_3.setGeometry(QtCore.QRect(10, 30, 101, 22))
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_4 = QtWidgets.QRadioButton(self.page_15)
        self.radioButton_4.setGeometry(QtCore.QRect(10, 75, 136, 22))
        self.radioButton_4.setObjectName("radioButton_4")
        self.checkBox_7 = QtWidgets.QCheckBox(self.page_15)
        self.checkBox_7.setEnabled(False)
        self.checkBox_7.setGeometry(QtCore.QRect(55, 50, 121, 22))
        self.checkBox_7.setObjectName("checkBox_7")
        # self.pushButton_14 = QtWidgets.QPushButton(self.page_15)
        # self.pushButton_14.setGeometry(QtCore.QRect(135, 235, 75, 23))
        # self.pushButton_14.setObjectName("pushButton_14")
        self.pushButton_15 = QtWidgets.QPushButton(self.page_15)
        self.pushButton_15.setGeometry(QtCore.QRect(135, 235, 75, 23))
        self.pushButton_15.setObjectName("pushButton_15")
        self.line_14 = QtWidgets.QFrame(self.page_15)
        self.line_14.setGeometry(QtCore.QRect(10, 220, 201, 20))
        self.line_14.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_14.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_14.setObjectName("line_14")
        self.stackedWidget_3.addWidget(self.page_15)
        self.progressBar_7 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_7.setGeometry(QtCore.QRect(800, 750, 201, 23))
        self.progressBar_7.setProperty("value", 100)
        self.progressBar_7.setObjectName("progressBar_7")
        self.pushButton_19 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_19.setGeometry(QtCore.QRect(955, 182, 41, 23))
        self.pushButton_19.setObjectName("pushButton_19")
        # palette = QtGui.QPalette()
        # brush = QtGui.QBrush(QtGui.QColor(244, 130, 130))
        # brush.setStyle(QtCore.Qt.SolidPattern)
        # palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        # self.pushButton_19.setPalette(palette)
        self.checkBox_9 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_9.setEnabled(False)
        self.checkBox_9.setGeometry(QtCore.QRect(805, 725, 176, 21))
        self.checkBox_9.setObjectName("checkBox_9")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setEnabled(False)
        self.lineEdit_3.setGeometry(QtCore.QRect(895, 80, 106, 21))
        self.lineEdit_3.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_3.setValidator(self.onlyInt)
        self.checkBox_12 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_12.setGeometry(QtCore.QRect(805, 80, 96, 21))
        self.checkBox_12.setObjectName("checkBox_12")
        self.label_37 = QtWidgets.QLabel(self.centralwidget)
        self.label_37.setGeometry(QtCore.QRect(805, 212, 41, 21))
        self.label_37.setFont(font)
        self.label_37.setObjectName("label_37")
        self.label_38 = QtWidgets.QLabel(self.centralwidget)
        self.label_38.setGeometry(QtCore.QRect(805, 162, 151, 21))
        self.label_38.setFont(font)
        self.label_38.setObjectName("label_38")
        self.label_39 = QtWidgets.QLabel(self.centralwidget)
        self.label_39.setGeometry(QtCore.QRect(805, 100, 156, 21))
        self.label_39.setFont(font)
        self.label_39.setObjectName("label_39")


######################################################################################################
        self.stackedWidget_logo = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget_logo.setGeometry(QtCore.QRect(800, 0, 211, 81))
        self.stackedWidget_logo.setObjectName("stackedWidget_logo")

        self.page_logo = QtWidgets.QWidget()
        self.page_logo.setObjectName("page_logo")

        self.label = QtWidgets.QLabel(self.page_logo)
        self.label.setGeometry(QtCore.QRect(10, 0, 196, 76))
        self.label.setText("")
        root_dir = os.path.dirname(os.path.abspath(__file__))
        img = os.path.join(root_dir, 'resources/Wiggle.PNG')
        self.label.setPixmap(QtGui.QPixmap(img))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")

        self.stackedWidget_logo.addWidget(self.page_logo)

        self.page_logo_blk = QtWidgets.QWidget()
        self.page_logo_blk.setObjectName("page_logo_blk")

        self.textBrowser_3 = QtWidgets.QTextBrowser(self.page_logo_blk)
        self.textBrowser_3.setGeometry(QtCore.QRect(0, 0, 196, 76))
        self.textBrowser_3.setAcceptDrops(True)
        self.textBrowser_3.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.textBrowser_3.setObjectName("textBrowser_3")

        self.stackedWidget_logo.addWidget(self.page_logo_blk)

######################################################################################################

        self.checkBox_14 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_14.setEnabled(True)
        self.checkBox_14.setGeometry(QtCore.QRect(820, 146, 131, 16))
        self.checkBox_14.setObjectName("checkBox_14")
        self.spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox.setEnabled(False)
        self.spinBox.setGeometry(QtCore.QRect(935, 640, 61, 21))
        self.spinBox.setMinimum(1)
        self.spinBox.setMaximum(25)
        self.spinBox.setProperty("value", 5)
        self.spinBox.setObjectName("spinBox")
        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        self.label_22.setEnabled(False)
        self.label_22.setGeometry(QtCore.QRect(805, 645, 86, 16))
        self.label_22.setObjectName("label_22")
        self.label_23 = QtWidgets.QLabel(self.centralwidget)
        self.label_23.setGeometry(QtCore.QRect(805, 625, 181, 21))
        self.label_23.setFont(font)
        self.label_23.setObjectName("label_23")
        self.checkBox_15 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_15.setEnabled(False)
        self.checkBox_15.setGeometry(QtCore.QRect(805, 705, 116, 21))
        self.checkBox_15.setObjectName("checkBox_15")
        self.spinBox_5 = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_5.setEnabled(False)
        self.spinBox_5.setGeometry(QtCore.QRect(935, 705, 61, 21))
        self.spinBox_5.setMinimum(0)
        self.spinBox_5.setMaximum(255)
        self.spinBox_5.setSingleStep(10)
        self.spinBox_5.setProperty("value", 100)
        self.spinBox_5.setObjectName("spinBox_5")
        self.checkBox_16 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_16.setEnabled(False)
        self.checkBox_16.setGeometry(QtCore.QRect(805, 685, 176, 21))
        self.checkBox_16.setObjectName("checkBox_16")
        self.label_24 = QtWidgets.QLabel(self.centralwidget)
        self.label_24.setEnabled(False)
        self.label_24.setGeometry(QtCore.QRect(805, 665, 71, 16))
        self.label_24.setObjectName("label_24")
        self.spinBox_6 = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_6.setEnabled(False)
        self.spinBox_6.setGeometry(QtCore.QRect(885, 663, 36, 21))
        self.spinBox_6.setAcceptDrops(False)
        self.spinBox_6.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinBox_6.setMinimum(0)
        self.spinBox_6.setMaximum(255)
        self.spinBox_6.setSingleStep(1)
        self.spinBox_6.setProperty("value", 65)
        self.spinBox_6.setObjectName("spinBox_6")
        self.spinBox_7 = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_7.setEnabled(False)
        self.spinBox_7.setGeometry(QtCore.QRect(925, 663, 36, 21))
        self.spinBox_7.setAcceptDrops(False)
        self.spinBox_7.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinBox_7.setMinimum(0)
        self.spinBox_7.setMaximum(255)
        self.spinBox_7.setSingleStep(1)
        self.spinBox_7.setProperty("value", 105)
        self.spinBox_7.setObjectName("spinBox_7")
        self.spinBox_8 = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_8.setEnabled(False)
        self.spinBox_8.setGeometry(QtCore.QRect(964, 663, 36, 21))
        self.spinBox_8.setAcceptDrops(False)
        self.spinBox_8.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinBox_8.setMinimum(0)
        self.spinBox_8.setMaximum(255)
        self.spinBox_8.setSingleStep(1)
        self.spinBox_8.setProperty("value", 225)
        self.spinBox_8.setObjectName("spinBox_8")
        self.checkBox_17 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_17.setEnabled(True)
        self.checkBox_17.setGeometry(QtCore.QRect(905, 725, 96, 21))
        self.checkBox_17.setObjectName("checkBox_17")
        # self.graphicsView_2.raise_()
        # # self.textBrowser.raise_()
        # self.textBrowser_3.raise_()
        # self.stackedWidget.raise_()
        # self.comboBox.raise_()
        # self.comboBox_3.raise_()
        # self.comboBox_2.raise_()
        # self.stackedWidget_3.raise_()
        # self.progressBar_7.raise_()
        # self.pushButton_19.raise_()
        # self.checkBox_9.raise_()
        # self.lineEdit_3.raise_()
        # self.checkBox_12.raise_()
        # self.label_37.raise_()
        # self.label_38.raise_()
        # self.label_39.raise_()
        # self.label.raise_()
        # self.checkBox_14.raise_()
        # self.spinBox.raise_()
        # self.label_22.raise_()
        # self.label_23.raise_()
        # self.checkBox_15.raise_()
        # self.spinBox_5.raise_()
        # self.checkBox_16.raise_()
        # self.label_24.raise_()
        # self.spinBox_6.raise_()
        # self.spinBox_7.raise_()
        # self.spinBox_8.raise_()
        # self.checkBox_17.raise_()
        # self.stackedWidget_logo.raise_()

        ########################################################################################################################
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setStyleSheet("")
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.retranslateUi(MainWindow)
        self.signals_slots_subclasses(MainWindow)

    def signals_slots_subclasses(self, MainWindow):
        def _night_mode():
            if self.checkBox_17.isChecked():
                rgb_colors = [
                    self.spinBox_6,
                    self.spinBox_7,
                    self.spinBox_8
                ]
                for f in rgb_colors:
                    f.blockSignals(True)
                    f.setValue(255 - f.value())
                    f.blockSignals(False)

                self.textBrowser_3.setStyleSheet("background-color: rgb(0, 0, 0); color: white;")
                root_dir = os.path.dirname(os.path.abspath(__file__))
                img = os.path.join(root_dir, 'resources/Wiggle2.PNG')
                self.label.setPixmap(QtGui.QPixmap(img))
                self.w.graphicsView.setBackground(background='k')
                palette = QtGui.QPalette()
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
                brush = QtGui.QBrush(QtGui.QColor(51, 51, 51))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
                brush = QtGui.QBrush(QtGui.QColor(76, 76, 76))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
                brush = QtGui.QBrush(QtGui.QColor(63, 63, 63))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
                brush = QtGui.QBrush(QtGui.QColor(25, 25, 25))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
                brush = QtGui.QBrush(QtGui.QColor(34, 34, 34))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
                brush = QtGui.QBrush(QtGui.QColor(51, 51, 51))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
                brush = QtGui.QBrush(QtGui.QColor(189, 194, 185))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Highlight, brush)
                brush = QtGui.QBrush(QtGui.QColor(25, 25, 25))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
                brush = QtGui.QBrush(QtGui.QColor(51, 51, 51))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
                brush = QtGui.QBrush(QtGui.QColor(76, 76, 76))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
                brush = QtGui.QBrush(QtGui.QColor(63, 63, 63))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
                brush = QtGui.QBrush(QtGui.QColor(25, 25, 25))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
                brush = QtGui.QBrush(QtGui.QColor(34, 34, 34))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
                brush = QtGui.QBrush(QtGui.QColor(51, 51, 51))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
                brush = QtGui.QBrush(QtGui.QColor(189, 194, 185))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Highlight, brush)
                brush = QtGui.QBrush(QtGui.QColor(25, 25, 25))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
                brush = QtGui.QBrush(QtGui.QColor(25, 25, 25))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
                brush = QtGui.QBrush(QtGui.QColor(51, 51, 51))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
                brush = QtGui.QBrush(QtGui.QColor(76, 76, 76))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
                brush = QtGui.QBrush(QtGui.QColor(63, 63, 63))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
                brush = QtGui.QBrush(QtGui.QColor(25, 25, 25))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
                brush = QtGui.QBrush(QtGui.QColor(34, 34, 34))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
                brush = QtGui.QBrush(QtGui.QColor(25, 25, 25))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
                brush = QtGui.QBrush(QtGui.QColor(25, 25, 25))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
                brush = QtGui.QBrush(QtGui.QColor(51, 51, 51))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
                brush = QtGui.QBrush(QtGui.QColor(51, 51, 51))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
                brush = QtGui.QBrush(QtGui.QColor(174, 174, 174))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Highlight, brush)
                brush = QtGui.QBrush(QtGui.QColor(51, 51, 51))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
            else:
                rgb_colors = [
                    self.spinBox_6,
                    self.spinBox_7,
                    self.spinBox_8
                ]
                for f in rgb_colors:
                    f.blockSignals(True)
                    f.setValue(255 - f.value())
                    f.blockSignals(False)

                self.textBrowser_3.setStyleSheet("background-color: rgb(255, 255, 255); color: black;")
                root_dir = os.path.dirname(os.path.abspath(__file__))
                img = os.path.join(root_dir, 'resources/Wiggle.PNG')
                self.label.setPixmap(QtGui.QPixmap(img))
                self.w.graphicsView.setBackground(background='w')
                palette = QtGui.QPalette()
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
                brush = QtGui.QBrush(QtGui.QColor(225, 225, 225))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
                brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
                brush = QtGui.QBrush(QtGui.QColor(112, 112, 112))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
                brush = QtGui.QBrush(QtGui.QColor(150, 150, 150))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
                brush = QtGui.QBrush(QtGui.QColor(225, 225, 225))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
                brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
                brush = QtGui.QBrush(QtGui.QColor(225, 225, 225))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
                brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
                brush = QtGui.QBrush(QtGui.QColor(112, 112, 112))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
                brush = QtGui.QBrush(QtGui.QColor(150, 150, 150))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
                brush = QtGui.QBrush(QtGui.QColor(225, 225, 225))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
                brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
                brush = QtGui.QBrush(QtGui.QColor(112, 112, 112))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
                brush = QtGui.QBrush(QtGui.QColor(225, 225, 225))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
                brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
                brush = QtGui.QBrush(QtGui.QColor(112, 112, 112))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
                brush = QtGui.QBrush(QtGui.QColor(150, 150, 150))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
                brush = QtGui.QBrush(QtGui.QColor(112, 112, 112))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
                brush = QtGui.QBrush(QtGui.QColor(112, 112, 112))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
                brush = QtGui.QBrush(QtGui.QColor(225, 225, 225))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
                brush = QtGui.QBrush(QtGui.QColor(225, 225, 225))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
                brush = QtGui.QBrush(QtGui.QColor(225, 225, 225))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
            self.palette = palette
            MainWindow.setPalette(self.palette)
            MainWindow.setCentralWidget(self.centralwidget)
            self.w._refresh_plot_brushes()

        def _colour_toggle():
            self.comboBox_5.blockSignals(not self.checkBox_5.isChecked())
            self.w.lastClicked = None

        self.w = InteractiveWindow(self.centralwidget, self.wiggle)
        self.FileIO = FileIO(ui)

        self.checkBox_17.toggled['bool'].connect(_night_mode)
        self.checkBox_17.stateChanged.connect(lambda : self.w.plot_clicked(None, None, None))
        self.checkBox_5.toggled['bool'].connect(_colour_toggle)

        self.checkBox_5.setEnabled(False)
        self.checkBox_8.setEnabled(True)
        self.checkBox_20.setEnabled(True)
        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.pushButton_17.setEnabled(False)
        self.pushButton_18.setEnabled(False)
        self.pushButton_28.setEnabled(True)
        self.comboBox_5.blockSignals(True)

        self.stackedWidget.setCurrentIndex(0)
        self.stackedWidget_3.setCurrentIndex(5)
        self.stackedWidget_4.setCurrentIndex(1)
        self.comboBox_9.setCurrentIndex(2)
        self.comboBox_7.setCurrentIndex(2)
        self.comboBox_8.setCurrentIndex(3)
        self.checkBox_15.toggled['bool'].connect(self.spinBox_5.setEnabled)

        self.checkBox_14.stateChanged.connect(self.FileIO.input_type)
        self.comboBox.currentIndexChanged['int'].connect(self.FileIO.input_type)
        # self.comboBox.currentIndexChanged['int'].connect(lambda: self.checkBox_14.setChecked(False))

        self.comboBox_2.activated['int'].connect(self.stackedWidget_3.setCurrentIndex)


        self.checkBox.toggled['bool'].connect(self.pushButton_27.setDisabled)
        self.radioButton_4.toggled['bool'].connect(self.checkBox_7.setDisabled)
        self.radioButton_4.clicked['bool'].connect(self.pushButton_9.setEnabled)
        self.radioButton_4.clicked['bool'].connect(self.pushButton_5.setEnabled)

        self.radioButton_3.clicked['bool'].connect(self.pushButton_5.setEnabled)
        self.radioButton_3.clicked['bool'].connect(self.pushButton_9.setEnabled)
        self.radioButton_3.toggled['bool'].connect(self.checkBox_7.setEnabled)


        self.checkBox_12.toggled['bool'].connect(self.lineEdit_3.setEnabled)
        self.checkBox_12.clicked['bool'].connect(self.lineEdit_3.setEnabled)

        self.checkBox_12.stateChanged.connect(self.reset)
        self.lineEdit_3.textChanged.connect(self.reset)
        self.pushButton_19.clicked.connect(self.reset)

        # CryoSPARC multiple inputs
        self.pushButton_5.clicked.connect(
            partial(self.FileIO.browse_file, "Select consensus map", self.pushButton_5, "Volumes (*.mrc)"))
        self.pushButton_9.clicked.connect(
            partial(self.FileIO.browse_file, "Select ALL component maps", self.pushButton_9, "Volumes (*.mrc)"))
        self.pushButton_13.clicked.connect(
            partial(self.FileIO.browse_file, "Select particle.cs file containing metadata", self.pushButton_13, "Particles (*.cs)"))

        # CryoSPARC single input
        self.pushButton_16.clicked.connect(
            partial(self.FileIO.browse_file, "Select bundled cryoSPARC file.", self.pushButton_16, "Bundle (*.npz)"))

        #CryoDRGN multiple inputs
        self.pushButton_6.clicked.connect(
            partial(self.FileIO.browse_file, "Select config file (e.g. config.pkl)", self.pushButton_6, "Pickle (*.pkl)"))
        self.pushButton_7.clicked.connect(
            partial(self.FileIO.browse_file, "Select network weights (e.g. weights.49.pkl)", self.pushButton_7, "Pickle (*.pkl)"))
        self.pushButton_8.clicked.connect(
            partial(self.FileIO.browse_file, "Select latent space (e.g. z.49.pkl)", self.pushButton_8, "Pickle (*.pkl)"))

        # CryoSPARC single input
        self.pushButton_21.clicked.connect(
            partial(self.FileIO.browse_file, "Select bundled cryoDRGN file.", self.pushButton_21, "Bundle (*.npz)"))

        self.pushButton_19.clicked.connect(self.run_embedding)
        self.pushButton_23.clicked.connect(self.run_clustering)
        self.pushButton_10.clicked.connect(self.run_MEP)

        self.checkBox_21.toggled['bool'].connect(self.pushButton.setEnabled)
        self.checkBox_21.toggled['bool'].connect(self.pushButton_2.setEnabled)
        self.checkBox_21.toggled['bool'].connect(self.pushButton_17.setEnabled)
        self.checkBox_21.toggled['bool'].connect(self.comboBox_10.setEnabled)

        self.checkBox_6.toggled['bool'].connect(self.checkBox_2.setEnabled)
        self.comboBox_6.currentIndexChanged.connect(self.w.change_current_MEP_view)

        #SLOTS TO MODIFY THE PLOT / INTERACTIVE WINDOW
        self.comboBox_5.currentIndexChanged['int'].connect(self.w._refresh_plot_brushes)

        self.checkBox_8.stateChanged.connect(self.w.plot_trajectories)

        self.checkBox_5.stateChanged.connect(
            lambda: self.w.modify_state('colour',
                                        self.checkBox_5.isChecked())
        )

        self.checkBox_9.stateChanged.connect(self.w.plot_legend)

        self.radioButton_3.toggled['bool'].connect(self.w.plot_rois)



        self.timer.timeout.connect(
            lambda: self.reportProgress('', '', False, True, False)
        )

        self.statusTimer.timeout.connect(
            lambda: self.status(False)
        )

        self.pushButton_2.clicked.connect(self.w.user_anchor_clear)
        self.pushButton.clicked.connect(self.w.user_anchor_query)
        self.pushButton_18.clicked.connect(self.w.user_anchor_save)
        self.pushButton_17.clicked.connect(self.w.user_anchor_reset)
        self.pushButton_28.clicked.connect(self.w.component_anchor_reset)

        self.pushButton_15.clicked.connect(self.FileIO.export_roi)
        self.pushButton_15.clicked.connect(self.FileIO.export_clusters)

        self.checkBox_13.stateChanged.connect(
            lambda : self.w.modify_state('MEP', self.checkBox_13.isChecked())
        )

        self.spinBox_6.valueChanged.connect(self.w._refresh_plot_brushes)
        self.spinBox_7.valueChanged.connect(self.w._refresh_plot_brushes)
        self.spinBox_8.valueChanged.connect(self.w._refresh_plot_brushes)
        self.spinBox_5.valueChanged.connect(self.w._refresh_plot_brushes)
        self.checkBox_15.stateChanged.connect(self.w._refresh_plot_brushes)
        self.checkBox_16.stateChanged.connect(self.w._refresh_plot_silhouette)
        self.spinBox.valueChanged.connect(self.w._refresh_plot_size)

        self.pushButton_22.clicked.connect(self.w.volumiser_by_component)
        self.pushButton_27.clicked.connect(self.w.volumiser)
        self.pushButton_20.clicked.connect(self.w.volumiser_by_traj)

        self.pushButton_25.clicked.connect(self.w._call_update)
        self.spinBox_2B.editingFinished.connect(self.w._call_update)
        self.spinBox_9.editingFinished.connect(self.w._call_update)
        self.doubleSpinBox_10.editingFinished.connect(self.w._call_update)

        def toggle_logo(event):
            ind = 1 - (self.stackedWidget_logo.currentIndex() % 2)
            self.stackedWidget_logo.setCurrentIndex(ind)

        self.label.mousePressEvent = toggle_logo
        self.textBrowser_3.mousePressEvent = toggle_logo

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "WIGGLE - 0.2.2 dev"))
        # self.textBrowser.setHtml(_translate("MainWindow",
        #                                     "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
        #                                     "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
        #                                     "p, li { white-space: pre-wrap; }\n"
        #                                     "</style></head><body style=\" font-family:\'Noto Sans\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
        #                                     "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\'; font-size:10pt; font-weight:600;\">Command line output will appear here</span></p></body></html>"))
        self.textBrowser_3.setHtml(_translate("MainWindow",
                                              "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                              "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                              "p, li { white-space: pre-wrap; }\n"
                                              "</style></head><body style=\" font-family:\'Noto Sans\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\'; font-size:7pt; font-weight:600;\">Charles Bayly-Jones 2021</span></p>\n"
                                              "<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'MS Shell Dlg 2\'; font-size:7pt; font-weight:600;\"><br /></p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\'; font-size:7pt; font-weight:600;\">wiggle.help@gmail.com</span></p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:7pt;\"> </span></p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\'; font-size:7pt; font-weight:600;\">Monash University, Australia</span></p></body></html>"))
        self.pushButton_8.setText(_translate("MainWindow", "Load latent (Z) space (.pkl)"))
        self.pushButton_6.setText(_translate("MainWindow", "Load cryoDRGN config (.pkl)"))
        self.pushButton_7.setText(_translate("MainWindow", "Load network weights (.pkl)"))
        self.pushButton_5.setToolTip(
            _translate("MainWindow", "Output basis map file from cryoSPARC 3D variational analysis"))
        self.pushButton_5.setText(_translate("MainWindow", "Load consensus map (.mrc)"))
        self.pushButton_9.setToolTip(
            _translate("MainWindow", "Select ALL variational component maps from cryoSPARC 3D variational analysis"))
        self.pushButton_9.setText(_translate("MainWindow", "Load component maps (.mrc)"))
        self.pushButton_13.setToolTip(_translate("MainWindow",
                                                 "Particle metadata file (cryoSPARC format, .CS). Looks like \"cryosparc_P0_J0_particles.cs\""))
        self.pushButton_13.setText(_translate("MainWindow", "Load particle metadata (.cs)"))
        self.pushButton_3.setText(_translate("MainWindow", "Load canonical map"))
        self.pushButton_4.setText(_translate("MainWindow", "Load deformation field"))
        self.label_41.setText(_translate("MainWindow", "Currently under development. Not yet available."))
        self.pushButton_16.setToolTip(_translate("MainWindow", "Compiled (single-file) binary format of 3DVA analysis"))
        self.pushButton_16.setStatusTip(_translate("MainWindow", "To generate, run \'/wiggle/scripts/compile_3DVA_output.py\' or instead swap dialog back to import individual files"))
        self.pushButton_16.setText(_translate("MainWindow", "Load cryoSPARC 3DVA (.npy)"))
        self.pushButton_21.setToolTip(_translate("MainWindow", "Compiled (single-file) binary format of cryoDRGN analysis"))
        self.pushButton_21.setStatusTip(_translate("MainWindow", "To generate, run \'/wiggle/scripts/compile_cryoDRGN_output.py\' or instead swap dialog back to import individual files"))
        self.pushButton_21.setText(_translate("MainWindow", "Load cryoDRGN (.npy)"))
        self.comboBox.setToolTip(_translate("MainWindow", "Mode"))
        self.comboBox.setItemText(0, _translate("MainWindow", "cryoDRGN"))
        self.comboBox.setItemText(1, _translate("MainWindow", "cryoSPARC 3DVA"))
        self.comboBox.setItemText(2, _translate("MainWindow", "cryoSPARC 3Dflex"))
        self.comboBox_3.setToolTip(_translate("MainWindow", "Type of dimensionality reduction"))
        self.comboBox_3.setStatusTip(_translate("MainWindow", "To enable, first import a latent space."))
        self.comboBox_3.setItemText(0, _translate("MainWindow", "UMAP (slow)"))
        self.comboBox_3.setItemText(1, _translate("MainWindow", "PCA (very fast)"))
        self.comboBox_3.setItemText(2, _translate("MainWindow", "tSNE (fast)"))
        self.comboBox_3.setItemText(3, _translate("MainWindow", "PHATE (very slow)"))
        self.comboBox_3.setItemText(4, _translate("MainWindow", "cVAE (fast)"))
        self.comboBox_2.setStatusTip(_translate("MainWindow", "To enable, first run dimensionality reduction."))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "Interactive"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "Clustering"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "Graph Traversal"))
        self.comboBox_2.setItemText(3, _translate("MainWindow", "MEP Analysis"))
        self.comboBox_2.setItemText(4, _translate("MainWindow", "Particle Export"))
        self.checkBox.setToolTip(_translate("MainWindow",
                                            "Automatically replace the displayed map in ChimeraX with the selected configurational state."))
        self.checkBox.setText(_translate("MainWindow", "Continuous update"))
        self.label_8.setText(_translate("MainWindow", "Interactive mode"))
        self.pushButton_27.setToolTip(_translate("MainWindow",
                                                 "Manually update the displayed map in ChimeraX with the configurational state currently selected."))
        self.pushButton_27.setText(_translate("MainWindow", "Manual"))
        self.checkBox_3.setToolTip(_translate("MainWindow",
                                              "Whether to maintain a single map object (overwrite previous) or create multiple map objects"))
        self.checkBox_3.setText(_translate("MainWindow", "Replace map each time?"))
        self.label_25.setText(_translate("MainWindow", "Low pass filter (A)?"))
        self.label_26.setText(_translate("MainWindow", "Isosurface threshold"))
        self.pushButton_25.setText(_translate("MainWindow", "Go!"))
        self.label_27.setText(_translate("MainWindow", "Crop to box"))
        self.checkBox_4.setToolTip(_translate("MainWindow",
                                              "Generate all volumes in the opposite hand. Helpful when chirality is incorrect."))
        self.checkBox_4.setText(_translate("MainWindow", "Z-flip all output"))
        self.label_28.setText(_translate("MainWindow", "then downsample to"))
        self.label_31.setText(_translate("MainWindow", "then crop box to"))
        self.label_32.setText(_translate("MainWindow", "Downsample to"))
        self.label_7.setText(_translate("MainWindow", "Cluster analysis"))
        self.label_11.setToolTip(
            _translate("MainWindow", "Determines the number of classes, groups or clusters generated by kmeans"))
        self.label_11.setText(
            _translate("MainWindow", "<html><head/><body><p align=\"right\">Number of clusters? :</p></body></html>"))
        self.pushButton_23.setToolTip(_translate("MainWindow", "Start analysis."))
        self.pushButton_23.setText(_translate("MainWindow", "Go!"))
        # self.pushButton_24.setToolTip(_translate("MainWindow", "Clear previous results."))
        # self.pushButton_24.setText(_translate("MainWindow", "Reset"))
        self.checkBox_5.setToolTip(_translate("MainWindow", "Colour scatterplot by k means cluster indicies"))
        self.checkBox_5.setText(_translate("MainWindow", "Colour points by clusters"))
        self.checkBox_6.setToolTip(
            _translate("MainWindow", "For each cluster center, generate a volume and display it in ChimeraX."))
        self.checkBox_6.setText(_translate("MainWindow", "Generate volumes at centers?"))
        self.comboBox_4.setToolTip(_translate("MainWindow", "To enable, first import a latent space."))
        self.comboBox_4.setItemText(0, _translate("MainWindow", "KMeans (fast)"))
        self.comboBox_4.setItemText(1, _translate("MainWindow", "Affinity Propagation (slow)"))
        self.comboBox_4.setItemText(2, _translate("MainWindow", "MeanShift (slow)"))
        self.comboBox_4.setItemText(3, _translate("MainWindow", "Spectral Clustering (slow)"))
        self.comboBox_4.setItemText(4, _translate("MainWindow", "Ward (fast)"))
        self.comboBox_4.setItemText(5, _translate("MainWindow", "Agglomerative Clustering (fast)"))
        self.comboBox_4.setItemText(6, _translate("MainWindow", "DBSCAN (fast)"))
        self.comboBox_4.setItemText(7, _translate("MainWindow", "OPTICS (slow)"))
        self.comboBox_4.setItemText(8, _translate("MainWindow", "BIRCH (fast)"))
        self.comboBox_4.setItemText(9, _translate("MainWindow", "Gaussian Mixture (fast)"))
        self.comboBox_5.setToolTip(_translate("MainWindow", "To enable, first import a latent space."))
        self.comboBox_5.setItemText(0, _translate("MainWindow", "viridis (u)"))
        self.comboBox_5.setItemText(1, _translate("MainWindow", "plasma (u)"))
        self.comboBox_5.setItemText(2, _translate("MainWindow", "inferno (u)"))
        self.comboBox_5.setItemText(3, _translate("MainWindow", "magma (u)"))
        self.comboBox_5.setItemText(4, _translate("MainWindow", "cividis (u)"))
        self.comboBox_5.setItemText(5, _translate("MainWindow", "twilight (c)"))
        self.comboBox_5.setItemText(6, _translate("MainWindow", "hsv (c)"))
        self.comboBox_5.setItemText(7, _translate("MainWindow", "seismic (d)"))
        self.comboBox_5.setItemText(8, _translate("MainWindow", "coolwarm (d)"))
        self.comboBox_5.setItemText(9, _translate("MainWindow", "Spectral (d)"))
        self.comboBox_5.setItemText(10, _translate("MainWindow", "PiYG (d)"))
        self.comboBox_5.setItemText(11, _translate("MainWindow", "PRGn (d)"))
        self.comboBox_5.setItemText(12, _translate("MainWindow", "RdGy (d)"))
        self.comboBox_5.setItemText(13, _translate("MainWindow", "bwr (d)"))
        self.checkBox_2.setToolTip(_translate("MainWindow",
                                              "Colour reconstruction within ChimeraX based on cluster label. To enable, first run kmeans."))
        self.checkBox_2.setText(_translate("MainWindow", "Colour volume by cluster"))
        self.pushButton.setToolTip(_translate("MainWindow", "Start analysis."))
        self.pushButton.setText(_translate("MainWindow", "Done"))
        self.label_5.setText(_translate("MainWindow", "Path traversal"))
        self.pushButton_2.setToolTip(_translate("MainWindow", "Clear previous results."))
        self.pushButton_2.setText(_translate("MainWindow", "Clear"))
        self.label_6.setToolTip(_translate("MainWindow",
                                           "The number of currently selected anchors is displayed here. Reset will clear if you need to start again."))
        self.label_6.setText(_translate("MainWindow", "Current anchor count:"))
        self.label_10.setText(_translate("MainWindow", "OR"))
        self.checkBox_21.setToolTip(_translate("MainWindow",
                                                 "Select multiple anchor points in the latent space and then attempt to traverse the space crossing these anchors."))
        self.checkBox_21.setText(_translate("MainWindow", "Select anchor points:"))
        self.checkBox_8.setText(_translate("MainWindow", "Plot trajectories?"))
        self.checkBox_10.setToolTip(_translate("MainWindow",
                                               "Highlight the latent coordinate corresponding to currently displayed volume."))
        self.checkBox_10.setText(_translate("MainWindow", "Link volume to plot?"))
        self.lineEdit_12.setText(_translate("MainWindow", "-1"))
        self.label_20.setToolTip(
            _translate("MainWindow", "Determines the number of classes, groups or clusters generated by kmeans"))
        self.label_20.setText(
            _translate("MainWindow", "<html><head/><body><p align=\"right\">Number of steps? :</p></body></html>"))
        self.pushButton_17.setToolTip(_translate("MainWindow", "Delete all trajectories and reset."))
        self.pushButton_17.setText(_translate("MainWindow", "Reset"))
        self.pushButton_18.setToolTip(_translate("MainWindow", "Clear previous results."))
        self.pushButton_18.setText(_translate("MainWindow", "Save"))
        self.label_21.setText(_translate("MainWindow", "Select trajectory:"))
        self.comboBox_10.setToolTip(_translate("MainWindow", "User defined paths for volume generation."))
        self.pushButton_20.setToolTip(_translate("MainWindow", "Compute volumes."))
        self.pushButton_20.setText(_translate("MainWindow", "Go!"))
        self.checkBox_19.setToolTip(_translate("MainWindow",
                                               "If selected, wiggle will generate a morph map corresponding to the trajectory. If unselected, individual volumes along the path with be rendered."))
        self.checkBox_19.setText(_translate("MainWindow", "Morph map?"))
        self.checkBox_20.setToolTip(_translate("MainWindow", "Change the direction of the path to the inverse."))
        self.checkBox_20.setText(_translate("MainWindow", "Reverse?"))
        self.pushButton_22.setToolTip(_translate("MainWindow", "Compute volumes."))
        self.pushButton_22.setText(_translate("MainWindow", "Go!"))
        self.label_9.setText(_translate("MainWindow", "Minimum energy path analysis"))
        self.pushButton_10.setToolTip(_translate("MainWindow", "Start analysis."))
        self.pushButton_10.setText(_translate("MainWindow", "Go!"))
        self.pushButton_11.setToolTip(_translate("MainWindow", "Clear previous results."))
        self.pushButton_11.setText(_translate("MainWindow", "Reset"))
        self.comboBox_6.setToolTip(_translate("MainWindow", "Ranked paths: least resistance to most resistance"))
        self.comboBox_7.setItemText(0, _translate("MainWindow", "3"))
        self.comboBox_7.setItemText(1, _translate("MainWindow", "5"))
        self.comboBox_7.setItemText(2, _translate("MainWindow", "7"))
        self.comboBox_7.setItemText(3, _translate("MainWindow", "9"))
        self.comboBox_8.setItemText(0, _translate("MainWindow", "3"))
        self.comboBox_8.setItemText(1, _translate("MainWindow", "5"))
        self.comboBox_8.setItemText(2, _translate("MainWindow", "7"))
        self.comboBox_8.setItemText(3, _translate("MainWindow", "9"))
        self.comboBox_8.setItemText(4, _translate("MainWindow", "11"))
        self.comboBox_8.setItemText(5, _translate("MainWindow", "13"))
        self.comboBox_8.setItemText(6, _translate("MainWindow", "15"))
        self.comboBox_8.setItemText(7, _translate("MainWindow", "17"))
        self.comboBox_8.setItemText(8, _translate("MainWindow", "19"))
        self.comboBox_8.setItemText(9, _translate("MainWindow", "21"))
        self.comboBox_9.setItemText(0, _translate("MainWindow", "Coarse (96)"))
        self.comboBox_9.setItemText(1, _translate("MainWindow", "Medium (128)"))
        self.comboBox_9.setItemText(2, _translate("MainWindow", "Fine (156)"))
        self.comboBox_9.setItemText(3, _translate("MainWindow", "Ultrafine (196)"))
        self.comboBox_9.setItemText(4, _translate("MainWindow", "Extreme (256)"))
        self.label_3.setText(_translate("MainWindow", "Width of weiner filter"))
        self.label_4.setText(_translate("MainWindow", "Width of median filter"))
        self.label_13.setText(_translate("MainWindow", "Resolution"))
        self.label_14.setText(_translate("MainWindow", "No. searches to spawn:"))
        self.lineEdit_7.setText(_translate("MainWindow", "1200"))
        self.lineEdit_8.setText(_translate("MainWindow", "0.05"))
        self.lineEdit_3.setText(_translate("MainWindow", "10000"))
        self.label_15.setText(_translate("MainWindow", "Sampling"))
        self.lineEdit_9.setText(_translate("MainWindow", "40"))
        self.label_16.setText(_translate("MainWindow", "Learning rate"))
        self.lineEdit_10.setText(_translate("MainWindow", "0.2"))
        self.label_17.setText(_translate("MainWindow", "Coupling constant"))
        self.lineEdit_11.setText(_translate("MainWindow", "2.5"))
        self.label_18.setText(_translate("MainWindow", "String tension"))
        self.label_19.setText(_translate("MainWindow", "Select path below"))
        self.checkBox_13.setToolTip(_translate("MainWindow", "Show trajectory as lines on the embedding."))
        self.checkBox_13.setText(_translate("MainWindow", "Plot trajectory?"))
        self.label_40.setText(_translate("MainWindow", "Minima to consider"))
        self.lineEdit_17.setText(_translate("MainWindow", "10"))
        self.checkBox_11.setToolTip(_translate("MainWindow",
                                               "Output all the volumes as a single volume series? If false, output individual volumes."))
        self.checkBox_11.setText(_translate("MainWindow", "Output volume series?"))
        self.pushButton_12.setToolTip(_translate("MainWindow", "Compute volumes."))
        self.pushButton_12.setText(_translate("MainWindow", "Go!"))
        self.checkBox_18.setToolTip(_translate("MainWindow", "Show intermedate plots for troubleshooting."))
        self.checkBox_18.setText(_translate("MainWindow", "Generate diagnostic plots?"))
        self.label_12.setText(_translate("MainWindow", "Particle curation"))
        self.radioButton_3.setText(_translate("MainWindow", "Lasso tool"))
        self.radioButton_4.setText(_translate("MainWindow", "Export by cluster"))
        self.checkBox_7.setText(_translate("MainWindow", "Invert selection"))
        # self.pushButton_14.setToolTip(_translate("MainWindow", "Clear previous results."))
        # self.pushButton_14.setText(_translate("MainWindow", "Reset"))
        self.pushButton_15.setToolTip(_translate("MainWindow", "Start analysis."))
        self.pushButton_15.setText(_translate("MainWindow", "Go!"))
        self.pushButton_19.setText(_translate("MainWindow", "Go!"))
        self.checkBox_9.setToolTip(_translate("MainWindow", "Colour scatterplot by k means cluster indicies"))
        self.checkBox_9.setText(_translate("MainWindow", "Hide legend"))
        self.lineEdit_3.setToolTip(_translate("MainWindow", "Subsample space to this many observations."))
        self.checkBox_12.setToolTip(_translate("MainWindow", "Large spaces can be slow or cause lag."))
        self.checkBox_12.setText(_translate("MainWindow", "Subsample"))
        self.label_37.setText(_translate("MainWindow", "Mode"))
        self.label_38.setText(_translate("MainWindow", "Dimensional reduction"))
        self.label_39.setText(_translate("MainWindow", "Variation analysis input"))
        self.checkBox_14.setToolTip(
            _translate("MainWindow", "Swap input dialog to accept single-file format for cryoSPARC \n"
                                     " 3DVA output. The single-file format simplifies sharing and \n"
                                     " can be compiled with \'/wiggle/scripts/compile_3DVA_output.py/\'"))
        self.checkBox_14.setText(_translate("MainWindow", "Single-file format"))
        self.label_22.setText(_translate("MainWindow", "Point size"))
        self.label_23.setText(_translate("MainWindow", "Display"))
        self.checkBox_16.setText(_translate("MainWindow", "Show silhouette"))
        self.label_24.setText(_translate("MainWindow", "Colour (rgb)"))
        self.checkBox_15.setText(_translate("MainWindow", "Transparency"))
        self.checkBox_17.setText(_translate("MainWindow", "Night mode"))
        self.label_29.setText(_translate("MainWindow", "Select latent coordinate:"))
        self.pushButton_28.setToolTip(_translate("MainWindow", "Clear all latent coordinate trajectories."))
        self.pushButton_28.setText(_translate("MainWindow", "Reset"))
        self.label_33.setText(_translate("MainWindow", "Pixel size (A/pix)"))

class InteractiveWindow(object):
    def __init__(self, centralWidget, session):
        self.wiggle = session
        self.wiggle_state = {}
        self.isColour = False
        self.isCluster = False
        self.isSubSample = False
        self.isMEPTrajPlot = False
        self.current_anchors = []
        self.isROI = False
        self.spaceChanged = True
        self.legend = True
        self.currentZind = None
        self.lastClicked = None
        self.lastPlot = None
        self._last_vol = None

        self.curvePlotsUSER = []
        self.curvePlotsUSER_pts = []
        self.TrajLists = []
        self.legendItemsTraj = []

        self.curvePlotsMEP = []
        self.MEP_TrajLists = []
        self.MEP_legendItemsTraj = []

        self.currentMEPdisplay = []
        self.currentMEPlegenddisplay = []

        self.labels = None
        self.scatterPlots = None

        self.config = ''
        self.weights = ''
        self.apix = 1
        self.user_apix = False
        self.labels = None
        self.embedding = None
        self.data = ''

        # Set up the pyqtgraph
        self.graphicsView = pg.GraphicsLayoutWidget(centralWidget)
        self.graphicsView.setGeometry(QtCore.QRect(0, 80, 796, 716))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView.setBackground(background='w')
        self.plotArea = self.graphicsView.addPlot()
        self.plotArea.addLegend()
        self.plotArea.hideAxis('bottom')
        self.plotArea.hideAxis('left')

    def new_wiggle_item(self):
        _init = {'indices': [], 'legend': '', 'curve': object, 'points': object, 'slider': None, 'type': None}
        if len(self.wiggle_state) == 0:
            self.wiggle_state[0] = _init
            return 0
        else:
            n = next(reversed(self.wiggle_state.keys()))+1
            self.wiggle_state[n] = _init
            return n

    def MEP_path_display(self):
        self.MEP_legendItemsTraj = []
        for _, item in enumerate(self.MEP_TrajLists):
            self.MEP_legendItemsTraj.append("path_" + str(_))
        self.currentMEPlegenddisplay = self.MEP_legendItemsTraj

        ui.comboBox_6.clear()
        ui.comboBox_6.addItem('all')
        [ui.comboBox_6.addItem(item) for item in self.MEP_legendItemsTraj]
        ui.comboBox_6.setEnabled(True)
        ui.checkBox_11.setEnabled(True)
        ui.checkBox_13.setEnabled(True)
        ui.pushButton_12.setEnabled(True)

    def change_current_MEP_view(self):
        if ui.comboBox_6.currentIndex() == 0:
            self.currentMEPdisplay = self.MEP_TrajLists
            self.modify_state('MEP', ui.checkBox_13.isChecked())
            self.currentMEPlegenddisplay = self.MEP_legendItemsTraj
        else:
            self.currentMEPlegenddisplay = [self.MEP_legendItemsTraj[ui.comboBox_6.currentIndex() - 1]]
            self.currentMEPdisplay = [self.MEP_TrajLists[ui.comboBox_6.currentIndex() - 1]]
            self.modify_state('MEP', ui.checkBox_13.isChecked())

    def user_anchor_query(self):
        flat_list = [item for sublist in self.current_anchors for item in sublist]

        if len(flat_list) > 0:
            item_index = self.new_wiggle_item()
            self.wiggle_state[item_index]['indices'] = flat_list
            self.wiggle_state[item_index]['legend'] = ''.join(('trajectory_', str(item_index)))
            self.wiggle_state[item_index]['type'] = 'user'
            self.current_anchors = []
            print('User selection of anchor points: ' + ' '.join([str(elem) for elem in flat_list]))

            ui.comboBox_10.clear()
            [ui.comboBox_10.addItem(item_dict['legend']) for item, item_dict in self.wiggle_state.items() if (item_dict['type'] == 'user')]
            ui.comboBox_10.setEnabled(True)
            ui.pushButton_20.setEnabled(True)
            ui.checkBox_19.setEnabled(True)
            ui.checkBox_20.setEnabled(True)

        self.plot_trajectories()
        ui.lineEdit_12.setText(str(len(self.current_anchors)))

    def user_anchor_clear(self):
        self.current_anchors = []
        ui.lineEdit_12.setText(str(len(self.current_anchors)))

    def user_anchor_reset(self):
        _remove = {}
        for item_index, item_dict in self.wiggle_state.items():
            if item_dict['type'] == 'user':
                self.plotArea.removeItem(item_dict['curve'])
                self.plotArea.removeItem(item_dict['points'])
                # self.plotArea.removeItem(item_dict['legend'])
                s = item_dict['slider']

                ### THIS NEEDS FIXING. Why does "if s is not None" return true for a NoneType object?....
                print(s)
                try:
                    s.delete()
                except:
                    print("Morph already deleted")
                _remove[item_index] = ui.comboBox_10.findText(item_dict['legend'])

        for state_index, comboBox_index in _remove.items():
            del self.wiggle_state[state_index]
            ui.comboBox_10.removeItem(comboBox_index)

        self.plot_legend()
        self.plot_trajectories()

        ui.lineEdit_12.setText(str(len(self.current_anchors)))
        if ui.comboBox_10.count() == 0:
            ui.comboBox_10.setEnabled(False)
            ui.pushButton_20.setEnabled(False)

        print(self.wiggle_state)

    def component_anchor_reset(self):
        _remove = []
        for item_index, item_dict in self.wiggle_state.items():
            if item_dict['type'] == 'latent':
                self.plotArea.removeItem(item_dict['curve'])
                self.plotArea.removeItem(item_dict['points'])
                # self.plotArea.removeItem(item_dict['legend'])
                s = item_dict['slider']

                ### THIS NEEDS FIXING. Why does "if s is not None" return true for a NoneType object?....
                print(s)
                try:
                    s.delete()
                except:
                    print("Morph already deleted")
                _remove.append(item_index)

        for state_index in _remove:
            del self.wiggle_state[state_index]

        self.plot_legend()
        self.plot_trajectories()

    def user_anchor_save(self):
        filename = QtWidgets.QFileDialog.getSaveFileName(None, "Save file", "", ".txt", options=QtWidgets.QFileDialog.DontUseNativeDialog)
        _filename = filename[0]+filename[1]
        with open(_filename, 'w') as f:
            for item_index, item_dict in self.wiggle_state.items():
                f.write(','.join(map(str, item_dict['trajectory'])))
                f.write(item_dict['legend'])
                f.write('\n')

        self.plot_trajectories()
        ui.lineEdit_12.setText(str(len(self.current_anchors)))

    def initialise_volume_engine(self):
        ui.pushButton_25.setEnabled(True)
        self.mode = ui.inputType[ui.comboBox.currentIndex()]
        if self.mode == 'cryodrgn':
            from .deps.miniDRGN import miniDRGN
            self.volume_engine = miniDRGN(ui, self.data[0], self.data[1], self.apix)
        elif self.mode == 'cryosparc_3dva':
            from .deps.miniSPARC import miniSPARC
            self.volume_engine = miniSPARC(ui, self.data[0], self.data[1], self.apix)

        ui.comboBox_12.clear()
        [ui.comboBox_12.addItem(''.join(('latent coordinate ', str(i)))) for i in range(len(self.data[2][0]))]

        self.volume_engine.done.connect(lambda: ui.pushButton_25.setEnabled(True))
        self.volume_engine.done.connect(lambda: ui.spinBox_2B.setEnabled(True))
        self.volume_engine.done.connect(lambda: ui.spinBox_9.setEnabled(True))
        def _do_volume():
            if ui.checkBox.isChecked():
                self.volumiser()
        self.volume_engine.done.connect(_do_volume)
        self._call_update()

    def _call_update(self):
        if ui.spinBox_2B != -1:
            ui.spinBox_9.setMaximum(ui.spinBox_2B.value())
        if ui.pushButton_25.isEnabled() and ui.spinBox_2B.isEnabled():
            ui.pushButton_25.setEnabled(False)
            ui.spinBox_2B.setEnabled(False)
            ui.spinBox_9.setEnabled(False)
            QtCore.QCoreApplication.processEvents()
            if ui.doubleSpinBox_10.isEnabled() and ui.doubleSpinBox_10.value() > 0:
                self.volume_engine.apix = ui.doubleSpinBox_10.value()
                self.volume_engine.apix_curr = ui.doubleSpinBox_10.value()
            elif ui.doubleSpinBox_10.isEnabled() and ui.doubleSpinBox_10.value() < 0:
                self.volume_engine.apix = 1
                self.volume_engine.apix_curr = 1
            self.volume_engine._update()

    # def points_to_original_indices(self, pts):
        # indices = []
        # for pt in pts:
        #     pt = pt.pos()
        #     x, y = pt.x(), pt.y()
        #     lx = np.argwhere(self.embedding[:, 0] == x)
        #     ly = np.argwhere(self.embedding[:, 1] == y)
        #     i = np.intersect1d(lx, ly).tolist()
        #     indices += i
        # return list(set(indices))

    def plot_clicked(self, plot, points, event):
        clickedPen = pg.mkPen((self._night_or_day(0), self._night_or_day(0), self._night_or_day(0)), width=3)
        if (points is None) and (self.lastClicked is not None):
            print(points, self.lastClicked)
            for pt in self.lastClicked:
                pt.resetPen()
                pt.setPen(clickedPen)
        elif points is not None:
            if self.lastClicked is not None:
                for pt in self.lastClicked:
                    pt.resetPen()
            for limit, pt in enumerate(points):
                pt.setPen(clickedPen)
                if limit == 5:
                    break
            self.lastClicked = points

            indices = [pt.data() for pt in points] #self.points_to_original_indices(points)

            if ui.checkBox_21.isChecked():
                if len(indices) > 1:
                    index = [indices[0]]
                else:
                    index = indices
                self.current_anchors.append(index)

            #Save the Z index for miniDRGN
            self.currentZind = indices[0]

            if ui.checkBox.isChecked():
                self.volumiser()

            ui.lineEdit_12.setText(str(len(self.current_anchors)))

    def modify_state(self, property, active: bool, val=''):  # Get rid of this, not necessary... cumbersome and redundant
        '''
        ###########################################################################################################
        Modify the plot options according to UI parameters
        ###########################################################################################################
        '''

        # Whether to trajectories should be shown
        if property == 'MEP':
            self.isMEPTrajPlot = active
            self.plot_MEPtrajectories()
            return

        #Whether to colour plot or not
        if property == 'colour':
            self.isColour = active

        # In case the embedding landscape is modified
        if property == 'space':
            self.spaceChanged = True

        if property == 'refresh':
            self.refresh_plot()
        else:
            self.plot_scatter()

    def plot_trajectories(self):
        '''
        ###########################################################################################################
        Plot (pseudo)trajectories if required
        ###########################################################################################################
        '''
        for item_index, item_dict in self.wiggle_state.items():
            self.plotArea.removeItem(item_dict['curve'])
            self.plotArea.removeItem(item_dict['points'])

        if ui.checkBox_8.isChecked():
            # Define trajectories by index
            traj = pg.mkPen((181, 34, 34), width=3, style=QtCore.Qt.DotLine)
            shadowPen = None  # or pg.mkPen((255, 255, 255), width=3)

            # Iterate over trajectories and plot these.
            for item, item_dict in self.wiggle_state.items():
                curvePlot = pg.PlotCurveItem(
                    pen=traj,
                    shadowPen=shadowPen
                )
                curvePlot_pts = pg.ScatterPlotItem(
                    # pen=pg.mkPen((0, 0, 0), width=1),
                    pxMode=True,
                    useCache=True,
                    pen=pg.mkPen(None),
                    size=ui.spinBox.value(),
                    downsample=True,
                    downsampleMethod='mean',
                    hoverable=True,
                    hoverSymbol='o',
                    hoverSize=30,
                    hoverBrush=pg.mkBrush(166, 229, 181, 150)
                )

                indices = item_dict['indices']
                curvePlot.setData(x=self.embedding[indices][:, 0], y=self.embedding[indices][:, 1])
                curvePlot_pts.setData(x=self.embedding[indices][:, 0], y=self.embedding[indices][:, 1])
                curvePlot_pts.sigHovered.connect(self.point_hover_slider)
                item_dict['curve'] = curvePlot
                item_dict['points'] = curvePlot_pts
                self.plotArea.addItem(curvePlot)
                self.plotArea.addItem(curvePlot_pts)

            if self.legend:
                self.plot_legend()

    def refresh_trajectories(self, morph_slider):
        if ui.checkBox_8.isChecked():
            slider_ind = floor(morph_slider.slider.value()/10)
            for item, item_dict in self.wiggle_state.items():
                if item_dict['slider'] == morph_slider:
                    _list = item_dict['indices']

                    pt_highlight = _list[slider_ind]
                    spots = []
                    for ind in _list:
                        x = self.embedding[ind][0]
                        y = self.embedding[ind][1]
                        if ind == pt_highlight:
                            s = {'pos': (x, y), 'size': 30, 'pen': None, 'brush': pg.mkBrush(166, 229, 181, 150), 'symbol': 'o'}
                        else:
                            s = {'pos': (x, y), 'size': ui.spinBox.value(), 'pen': None, 'symbol': 'o'}
                        spots.append(s)

                    item_dict['points'].setData(spots=spots)

    def point_hover_slider(self, plot, points):
        if ui.checkBox_10.isChecked():
            for item, item_dict in self.wiggle_state.items():
                if item_dict['points'] == plot and item_dict['slider'] is not None:
                    _list = item_dict['indices']
                    for pt in points:
                        pt = pt.pos()
                        x, y = pt.x(), pt.y()
                        lx = np.argwhere(self.embedding[:, 0] == x)
                        ly = np.argwhere(self.embedding[:, 1] == y)

                        i = np.intersect1d(lx, ly).tolist()[0]
                        if i in _list:
                            item_dict['slider'].set_slider(_list.index(i) * 10)

    def plot_MEPtrajectories(self):
        '''
        ###########################################################################################################
        Plot (pseudo)trajectories if required
        ###########################################################################################################
        '''
        for curvePlot in self.curvePlotsMEP:
            self.plotArea.removeItem(curvePlot)

        if self.isMEPTrajPlot:
            self.curvePlotsMEP = []
            # Define trajectories by index
            traj = pg.mkPen((168, 109, 0), width=3, style=QtCore.Qt.DotLine)
            shadowPen = None  # or pg.mkPen((255, 255, 255), width=3)
            # Iterate over trajectories and plot these.
            for n, index in enumerate(self.currentMEPdisplay):
                curvePlot = pg.PlotCurveItem(
                    pen=traj,
                    shadowPen=shadowPen
                )

                curvePlot.setData(x=self.embedding[index][:, 0], y=self.embedding[index][:, 1])
                self.curvePlotsMEP.append(curvePlot)
                self.plotArea.addItem(curvePlot)

            if self.legend:
                self.plot_legend()

    def plot_rois(self):
        print("_call to plot_rois")
        if self.spaceChanged:
            try:
                self.plotArea.removeItem(self.ROIs)
            except:
                pass


            axX = self.plotArea.getAxis('bottom')
            x_min, x_max = axX.range  # <------- get range of x axis
            axY = self.plotArea.getAxis('left')
            y_min, y_max = axY.range  # <------- get range of y axis
            com_x, com_y = (x_max + x_min) / 2, (y_max + y_min) / 2

            roiPen = pg.mkPen((22, 208, 115), width=3, style=QtCore.Qt.SolidLine)
            HoverPen = pg.mkPen((255, 255, 0), width=3, style=QtCore.Qt.DashLine)
            print("Made it into the function: plot_rois")
            self.ROIs = pg.PolyLineROI(
                [[0.2 * x_min + com_x, com_y],
                 [com_x, 0.2 * y_max + com_y],
                 [0.2 * x_max + com_x, com_y],
                 [com_x, 0.2 * y_min + com_y]],
                closed=True, pen=roiPen, handlePen=(153, 51, 255), hoverPen=HoverPen)

            self.spaceChanged = False

        if ui.radioButton_3.isChecked():
            self.plotArea.addItem(self.ROIs)
        else:
            self.plotArea.removeItem(self.ROIs)

    def plot_legend(self):
        s = time.time()
        if not ui.checkBox_9.isChecked():
            self.plotArea.legend.clear()
            if self.scatterPlots is not None:
                for legend, plot in zip(self.legendItems, self.scatterPlots):
                    self.plotArea.legend.addItem(plot, legend)
            if ui.checkBox_8.isChecked():
                for item, item_dict in self.wiggle_state.items():
                    self.plotArea.legend.addItem(item_dict['curve'], item_dict['legend'])
            if self.isMEPTrajPlot:
                for legend, plot in zip(self.currentMEPlegenddisplay, self.curvePlotsMEP):
                    self.plotArea.legend.addItem(plot, legend)
        else:
            self.plotArea.legend.clear()
        print(f'plot_legend time {time.time() - s}')

    def map_to_colour(self, slice):
        '''
        ###########################################################################################################
        Returns a colour object that is indexed by a value between 0 and 1
            e.g. cm[0.1] corresponds to a particular colour object
        ###########################################################################################################
        '''
        userColourOptions = (
            'viridis',
            'plasma',
            'inferno',
            'magma',
            'cividis',
            'twilight',
            'hsv',
            'seismic_r',
            'coolwarm',
            'Spectral_r',
            'PiYG_r',
            'PRGn_r',
            'RdGy_r',
            'bwr_r'
        )
        cm = pg.colormap.get(userColourOptions[ui.comboBox_5.currentIndex()], source='matplotlib')
        vals = cm.getColors()
        vals[:, -1] = self.alpha
        _cm = pg.colormap.ColorMap(pos=np.linspace(0.0, 1.0, vals.shape[0]), color=vals)
        return _cm[slice]

    def map_to_colour2(self, slice):
        '''
        ###########################################################################################################
        Returns a colour object that is indexed by a value between 0 and 1
            e.g. cm[0.1] corresponds to a particular colour object
        ###########################################################################################################
        '''
        userColourOptions = (
            'viridis',
            'plasma',
            'inferno',
            'magma',
            'cividis',
            'twilight',
            'hsv',
            'seismic_r',
            'coolwarm',
            'Spectral_r',
            'PiYG_r',
            'PRGn_r',
            'RdGy_r',
            'bwr_r'
        )
        cm = pg.colormap.get(userColourOptions[ui.comboBox_5.currentIndex()], source='matplotlib')
        vals = cm.getColors()
        vals[:, -1] = self.alpha
        _cm = pg.colormap.ColorMap(pos=np.linspace(0.0, 1.0, vals.shape[0]), color=vals)
        return _cm.map(slice)

############ NEW ################
    def _refresh_plot_silhouette(self):
        s = time.time()
        if ui.checkBox_16.isChecked():
            color = (self._night_or_day(0), self._night_or_day(0), self._night_or_day(0))
            pen_type = pg.mkPen(color, width=1)
        else:
            pen_type = None

        if self.scatterPlots is not None:
            for l, plot in enumerate(self.scatterPlots):
                plot.setPen(pen_type)
        print(f'_refresh_plot_silhouette time {time.time() - s}')

    def _refresh_plot_size(self):
        s = time.time()
        for plot in self.scatterPlots:
            plot.setSize(ui.spinBox.value())
        print(f'_refresh_plot_size time {time.time() - s}')

    def _refresh_plot_brushes(self):
        s = time.time()
        self.alpha = ui.spinBox_5.value() if ui.checkBox_15.isChecked() else 255
        pen_type = self._get_pen()
        if ui.checkBox_5.isChecked():
            unique = set(self.labels)
            unique_scaled = [item / (len(unique) - 1) for item in unique]
            self.colour_set = list(map(self.map_to_colour, unique_scaled))
            self.legendItems = ['clstr_' + str(label) for label in unique]
        else:
            self.colour_set = [pg.mkBrush(ui.spinBox_6.value(),
                                          ui.spinBox_7.value(),
                                          ui.spinBox_8.value(),
                                          self.alpha)]
            self.legendItems = ['all obs']

        if self.scatterPlots is not None:
            for l, plot in enumerate(self.scatterPlots):
                plot.setPen(pen_type)
                plot.setSize(ui.spinBox.value())
                plot.setBrush(self.colour_set[l])
        print(f'_refresh_plot_brushes time {time.time() - s}')

    def _night_or_day(self, X):
        if ui.checkBox_17.isChecked():
            return X+255
        else:
            return X

    def _get_pen(self):
        if ui.checkBox_16.isChecked():
            color = (self._night_or_day(255), self._night_or_day(255), self._night_or_day(255))
            return pg.mkPen(color, width=1)
        else:
            return None

############ OLD ################
    # def refresh_plot(self):
    #     self.alpha = ui.spinBox_5.value() if ui.checkBox_15.isChecked() else 255
    #     pen_type = self._get_pen()
    #
    #     if ui.checkBox_5.isChecked() and self.labels is not None:
    #         #Find non-redundant list of labels
    #         unique = set(self.labels)
    #         unique_scaled = [item / (len(unique) - 1) for item in unique]
    #         self.colour_set = list(map(self.map_to_colour, unique_scaled))
    #         self.colour_set_RGB = list(map(self.map_to_colour2, unique_scaled))
    #         self.legendItems = ['clstr_' + str(label) for label in unique]
    #
    #     else:
    #         self.colour_set = [pg.mkBrush(ui.spinBox_6.value(),
    #                                       ui.spinBox_7.value(),
    #                                       ui.spinBox_8.value(),
    #                                       self.alpha)]
    #         self.legendItems = ['all obs']
    #
    #     if self.scatterPlots is not None:
    #         s = time.time()
    #         for l, plot in enumerate(self.scatterPlots):
    #             plot.setPen(pen_type)
    #             plot.setSize(ui.spinBox.value())
    #             plot.setBrush(self.colour_set[l])
    #         print(f'refresh_plot time {time.time() - s}')

    def plot_scatter(self):
        '''
        ###########################################################################################################
        Plot 2D embedding of latent space as scatter plot
        Each point is an observation of the state represented by the position in the embedding
        ###########################################################################################################
        '''
        self.alpha = ui.spinBox_5.value() if ui.checkBox_15.isChecked() else 255
        pen_type = self._get_pen()

        if ui.checkBox_5.isChecked() and self.labels is not None:
            ############### FIRST BLOCK
            #Find non-redundant list of labels
            s = time.time()
            unique = set(self.labels)
            unique_scaled = [item / (len(unique) - 1) for item in unique]
            self.colour_set = list(map(self.map_to_colour, unique_scaled))
            self.legendItems = ['obs; clstr_' + str(label) for label in set(self.labels)]
            print(f'first_block time {time.time() - s}')

            ############### SECOND BLOCK
            s = time.time()
            cluster_list = []
            labels = np.array(self.labels)
            for label in set(self.labels):
                _indices = np.where(labels == label)[0]
                cluster_list.append(_indices)
            print(f'second_block time {time.time() - s}')

        else:
            cluster_list = ['']
            self.colour_set = [pg.mkBrush(ui.spinBox_6.value(),
                                          ui.spinBox_7.value(),
                                          ui.spinBox_8.value(),
                                          self.alpha)]
            self.legendItems = ['obs']

        ###########################################################################################################
        if self.scatterPlots is None:
            self.scatterPlots = []
        else:
            for subPlot in self.scatterPlots:
                self.plotArea.removeItem(subPlot)
            self.scatterPlots = []

        s = time.time()
        size = ui.spinBox.value()
        hoverPen = pg.mkPen((0, 0, 0), width=2)
        hoverBrush = pg.mkBrush(250, 128, 114)
        for cluster_n, cluster in enumerate(cluster_list):

            interactPlot = pg.ScatterPlotItem(
                pen=pen_type,
                size=size,
                # pxMode=True,
                # useCache=True,
                # symbol='o',
                # symbolPen=None,  # pg.mkPen((0, 0, 0), width=3),
                # symbolSize=10,
                # symbolBrush=colour_style,
                downsample=True,
                downsampleMethod='mean',
                hoverable=True,
                hoverSymbol='s',
                hoverSize=12,
                hoverPen=hoverPen,
                hoverBrush=hoverBrush,
                # brush=self.colour_set[cluster_n]
            )

            if self.isColour:
                interactPlot.setData(
                    pos=self.embedding[cluster], #[pt.pos() for pt in cluster],
                    data=cluster, #[pt.index() for pt in cluster], #JUMBLES THE INDICES!!! Found the bug.
                    brush=self.colour_set[cluster_n])
            else:
                interactPlot.setData(
                    pos=self.embedding,
                    data=range(0, len(self.embedding)),
                    brush=self.colour_set[0])

            interactPlot.sigClicked.connect(self.plot_clicked)
            self.scatterPlots.append(interactPlot)

        print(f'third_block time {time.time() - s} - iteration {cluster_n}')

        for l,plot in enumerate(self.scatterPlots):
            plot.setZValue(-l)
            self.plotArea.addItem(plot)

        if self.legend:
            self.plot_legend()

        # self.refresh_plot() ######## I don't think this is necessary...

        _enable = [
            ui.checkBox_9,
            ui.checkBox_15,
            ui.checkBox_16,
            ui.label_24,
            ui.label_22,
            ui.spinBox,
            ui.spinBox_6,
            ui.spinBox_7,
            ui.spinBox_8
        ]
        for f in _enable:
            f.setEnabled(True)

    def heat_map(self):
        x_min, y_min = np.amin(self.embedding, axis=0) #self.plotArea.getAxis('bottom')
        #x_min, y_min = axX.range  # <------- get range of x axis
        x_max, y_max = np.amax(self.embedding, axis=0) #self.plotArea.getAxis('left')
        #x_max, y_max = axY.range  # <------- get range of y axis


        pltRange = (x_max - x_min), (y_max - y_min)
        pixel = int(pltRange[0]) / 100
        x_range = int(pltRange[0] / pixel)
        y_range = int(pltRange[1] / pixel)

        x_r = np.arange(x_min, x_min + pltRange[0] - pixel, pixel)
        x = np.repeat(x_r, y_range)
        x = x.reshape(x_range, y_range)

        y_r = np.arange(y_min, y_min + pltRange[1] - pixel, pixel)
        y = np.tile(y_r, x_range)
        y = y.reshape(x_range, y_range)

        histo, _, _ = np.histogram2d(self.embedding[:, 0], self.embedding[:, 1], bins=[x_range, y_range])

        densityColourMesh = pg.PColorMeshItem(edgecolors=None, antialiasing=False, cmap='grey')
        densityColourMesh.setZValue(-1)
        self.plotArea.addItem(densityColourMesh)

        densityColourMesh.setData(x, y, histo[:-1, :-1])

    def volumiser(self):
        if self.currentZind is not None:
            from chimerax.map import volume_from_grid_data
            from chimerax.map_data import ArrayGridData

            v = self.volume_engine.generate(self.data[2][self.currentZind], ui.checkBox_4.isChecked())
            grid = ArrayGridData(v,
                                 name=''.join((self.mode,'_vol_i')),
                                 step=(self.volume_engine.apix_curr, self.volume_engine.apix_curr, self.volume_engine.apix_curr))

            vol = volume_from_grid_data(grid, self.wiggle, open_model=True, show_dialog=False)
            vol.set_parameters(surface_levels = [ui.doubleSpinBox.value()])

            if not (self._last_vol is None) and ui.checkBox_3.isChecked():
                if self._last_vol.deleted:
                    pass
                else:
                    self._last_vol.delete()

            self._last_vol = vol

    def volumiser_by_cluster(self):
        from chimerax.map import volume_from_grid_data
        from chimerax.map_data import ArrayGridData

        unique = set(self.labels)
        labels = np.array(self.labels)
        unique_scaled = [item / (len(unique) - 1) for item in unique]
        self.colour_set_RGB = list(map(self.map_to_colour2, unique_scaled))
        for label in unique:
            _indices = np.where(labels == label)[0]
            v = self.volume_engine.generate(self.data[2][_indices[0]], ui.checkBox_4.isChecked())
            grid = ArrayGridData(v,
                                 name=''.join((self.mode,'_cluster_',str(label))),
                                 step=(self.volume_engine.apix_curr, self.volume_engine.apix_curr, self.volume_engine.apix_curr))

            vol = volume_from_grid_data(grid, self.wiggle, open_model=True, show_dialog=False)
            if ui.checkBox_2.isChecked() and ui.checkBox_2.isEnabled():
                color = list(self.colour_set_RGB[label]/255)
                vol.set_parameters(surface_levels = [ui.doubleSpinBox.value()], surface_colors = [color])

    def volumiser_by_traj(self):
        from chimerax.map import volume_from_grid_data
        from chimerax.map_filter.morph import morph_maps
        from chimerax.map_filter.morph_gui import MorphMapSlider
        from .deps.volumiser import Volumiser

        ui.progressBar_7.reset()
        # Step 2: Create a QThread object
        self.thread4 = QtCore.QThread()
        # Step 3: Create a worker object
        self.v = Volumiser(ui, self.wiggle, self.wiggle_state, self.volume_engine, self.data, self.apix, self.mode, ui.checkBox_4.isChecked())

        # Step 4: Move worker to the thread
        self.v.moveToThread(self.thread4)
        # Step 5: Connect signals and slots
        self.thread4.started.connect(self.v.volume_by_traj)
        self.v.finished.connect(self.thread4.quit)
        self.v.finished.connect(self.v.deleteLater)
        self.thread4.finished.connect(self.thread4.deleteLater)
        self.v.progress.connect(ui.reportProgress)
        # self.v.msg.connect(ui.textBrowser.append)
        self.v.status.connect(ui.status)
        # Step 6: Start the thread
        self.thread4.start()

        # # Final resets
        def update():
            key = [k for k, v in self.wiggle_state.items() if v['legend'] == ui.comboBox_10.currentText()][0]
            grids = self.v.grids
            volumes = []
            for i in grids:
                vol = volume_from_grid_data(i, self.wiggle, open_model=True, show_dialog=False)
                vol.set_parameters(surface_levels=[ui.doubleSpinBox.value()])
                volumes.append(vol)

            if ui.checkBox_20.isChecked():
                volumes.reverse()
                self.wiggle_state[key]['indices'].reverse()

            if ui.checkBox_19.isChecked():
                frames = len(volumes) * 10 - 1
                step = 1 / frames
                volTraj = morph_maps(volumes, frames, 0, step, 1, (0.0, 1.0), False, False, None, True, False, 'all', 1, None)
                volTraj.stop_playing()

                morph_slider = MorphMapSlider(self.wiggle, volTraj)
                morph_slider.slider.valueChanged.connect(lambda : self.refresh_trajectories(morph_slider))
                self.wiggle_state[key]['slider'] = morph_slider

        self.v.finished.connect(update)

    def volumiser_by_MEP(self):
        from chimerax.map import volume_from_grid_data
        from chimerax.map_filter.morph import morph_maps
        from chimerax.map_filter.morph_gui import MorphMapSlider
        from .deps.volumiser import Volumiser

        # Step 2: Create a QThread object
        self.thread4 = QtCore.QThread()
        # Step 3: Create a worker object
        self.v = Volumiser(ui, self.wiggle, self.wiggle_state, self.volume_engine, self.data, self.apix, self.mode, ui.checkBox_4.isChecked())

        # Step 4: Move worker to the thread
        self.v.moveToThread(self.thread4)
        # Step 5: Connect signals and slots
        self.thread4.started.connect(self.v.volume_by_MEP)
        self.v.finished.connect(self.thread4.quit)
        self.v.finished.connect(self.v.deleteLater)
        self.thread4.finished.connect(self.thread4.deleteLater)
        self.v.progress.connect(ui.reportProgress)
        # self.v.msg.connect(ui.textBrowser.append)
        self.v.status.connect(ui.status)
        # Step 6: Start the thread
        self.thread4.start()

        # # Final resets
        def update():
            key = [k for k, v in self.wiggle_state.items() if v['legend'] == ui.comboBox_6.currentText()][0]
            grids = self.v.grids
            volumes = []
            for i in grids:
                vol = volume_from_grid_data(i, self.wiggle, open_model=True, show_dialog=False)
                vol.set_parameters(surface_levels=[ui.doubleSpinBox.value()])
                volumes.append(vol)

            if ui.checkBox_20.isChecked():
                volumes.reverse()
                self.wiggle_state[key]['indices'].reverse()

            if ui.checkBox_19.isChecked():
                frames = len(volumes) * 10 - 1
                step = 1 / frames
                volTraj = morph_maps(volumes, frames, 0, step, 1, (0.0, 1.0), False, False, None, True, False, 'all', 1, None)
                volTraj.stop_playing()

                morph_slider = MorphMapSlider(self.wiggle, volTraj)
                morph_slider.slider.valueChanged.connect(lambda : self.refresh_trajectories(morph_slider))
                self.wiggle_state[key]['slider'] = morph_slider

        self.v.finished.connect(update)

    def volumiser_by_component(self):
        from chimerax.map import volume_from_grid_data
        from chimerax.map_filter.morph import morph_maps
        from chimerax.map_filter.morph_gui import MorphMapSlider
        from .deps.volumiser import Volumiser

        ui.progressBar_7.reset()
        # Step 2: Create a QThread object
        self.thread4 = QtCore.QThread()
        # Step 3: Create a worker object
        self.v = Volumiser(ui, self.wiggle, self.TrajLists, self.volume_engine, self.data, self.apix, self.mode, ui.checkBox_4.isChecked())

        # Step 4: Move worker to the thread
        self.v.moveToThread(self.thread4)
        # Step 5: Connect signals and slots
        self.thread4.started.connect(self.v.volume_by_component)
        self.v.finished.connect(self.thread4.quit)
        self.v.finished.connect(self.v.deleteLater)
        self.thread4.finished.connect(self.thread4.deleteLater)
        self.v.progress.connect(ui.reportProgress)
        # self.v.msg.connect(ui.textBrowser.append)
        self.v.status.connect(ui.status)
        # Step 6: Start the thread
        self.thread4.start()

        from sklearn.neighbors import KDTree
        tree = KDTree(self.data[2])

        # # Final resets
        def update():
            grids = self.v.grids
            samples = self.v.samples
            volumes = []
            for i in grids:
                vol = volume_from_grid_data(i, self.wiggle, open_model=True, show_dialog=False)
                vol.set_parameters(surface_levels=[ui.doubleSpinBox.value()])
                volumes.append(vol)

            if ui.checkBox_20.isChecked():
                volumes.reverse()
                samples.reverse()

            if ui.checkBox_19.isChecked():
                frames = len(volumes) * 10 - 1
                step = 1 / frames
                volTraj = morph_maps(volumes, frames, 0, step, 1, (0.0, 1.0), False, False, None, True, False, 'all', 1, None)
                volTraj.stop_playing()

                item_index = self.new_wiggle_item()
                self.wiggle_state[item_index]['indices'] = [int(tree.query([s], 1, return_distance=False)) for s in samples]
                self.wiggle_state[item_index]['legend'] = ''.join(('sample_Z_', str(item_index)))
                self.wiggle_state[item_index]['type'] = 'latent'

                morph_slider = MorphMapSlider(self.wiggle, volTraj)
                morph_slider.slider.valueChanged.connect(lambda: self.refresh_trajectories(morph_slider))
                self.wiggle_state[item_index]['slider'] = morph_slider

                # ui.comboBox_10.clear()
                # [ui.comboBox_10.addItem(item_dict['legend']) for item, item_dict in self.wiggle_state.items()]

                self.plot_trajectories()

        self.v.finished.connect(update)

class Wiggle(object):
    def __init__(self, session):
        self.wiggle = session
        super().__init__()

    def chimera_initialise(self):
        layout = QtWidgets.QGridLayout()
        MainWindow = QtWidgets.QMainWindow()
        layout.addWidget(MainWindow)
        layout.setContentsMargins(0,0,0,0)

        #Initialise the Ui_MainWindow class
        global ui
        ui = Ui_MainWindow(self.wiggle)

        #Run UI setup method of Ui_MainWindow class
        ui.setupUi(MainWindow)
        return layout

class PackItUp:
    def __init__(self, args):
        self.args = args
        if self.args.mode == 'cryodrgn':
            self.read_cryodrgn()
        elif self.args.mode == '3dva':
            self.read_cryosparc()
        elif self.args.mode == '3dflex':
            print("not yet implemented, sorry...")
        else:
            print("Not recognised mode... this shouldn't happen.")

    def read_cryodrgn(self):
        config = np.load(self.args.config, allow_pickle=True)
        weights = torch.load(self.args.weights)
        z_space = np.load(self.args.z_space, allow_pickle=True)
        apix = self.args.apix
        if not os.path.isfile(self.args.output):
            np.savez_compressed(self.args.output, apix=apix, config=config, weights=weights, z_space=z_space)
        else:
            print(f'\033[93m {self.args.output} already exists. Exiting... \033[0m')

    def read_cryosparc(self):
        particles = np.load(self.args.particles)
        consensus_map = mrcfile.open(self.args.map).data
        components = [mrcfile.open(component).data for component in self.args.components]
        if not os.path.isfile(self.args.output):
            np.savez_compressed(self.args.output, particles=particles, consensus_map=consensus_map,
                                components=components)
        else:
            print(f'\033[93m {self.args.output} already exists. Exiting... \033[0m')

if __name__ == "__main__":
    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    def valid_mrc(param):
        base, ext = os.path.splitext(param)
        if ext.lower() not in ('.mrc','.MRC'):
            raise argparse.ArgumentTypeError('File must have a .mrc extension')
        return param

    def valid_pkl(param):
        base, ext = os.path.splitext(param)
        if ext.lower() not in ('.pkl',):
            raise argparse.ArgumentTypeError('File must have a .pkl extension')
        return param

    def valid_cs(param):
        base, ext = os.path.splitext(param)
        if ext.lower() not in ('.cs',):
            raise argparse.ArgumentTypeError('File must have a .cs extension')
        return param

    def valid_output(param):
        base, ext = os.path.splitext(param)
        if ext.lower() not in ('.npz',):
            return param + '.npz'
        return param

    # [print(bcolors.__dict__[key]+'TEST'+bcolors.ENDC) for key, value in bcolors.__dict__.items() if not bcolors.__dict__[key].startswith('__')]
    # create the top-level parser
    my_parser = argparse.ArgumentParser(
        prog=bcolors.OKBLUE+bcolors.BOLD+'wiggle.py'+bcolors.ENDC,
        description=textwrap.dedent(bcolors.BOLD+bcolors.HEADER+'''    ###########################################################################################################

        *** WIGGLE *** v0.1.6 - Wigget for Interactive, Graphical, & Guided Landscape Exploration

        Developed by Charles Bayly-Jones (2020-2022) - Monash University, Melbourne, Australia

        This is the command line mode to compile and bundle cryoEM conformational heterogeneity
        data types. To run wiggle interactively (i.e. for flexibility analysis and volume rendering),
        please install WIGGLE via UCSF ChimeraX toolshed and open within ChimeraX session.

    ###########################################################################################################'''+bcolors.ENDC),
        formatter_class=argparse.RawTextHelpFormatter #ArgumentDefaultsHelpFormatter
    )

    # create sub-parser
    sub_parsers = my_parser.add_subparsers(
        title="Operating modes",
        description="Select the operating mode. Supports cryoDRGN, cryoSPARC 3DVA, or cryoSPARC 3DFlex file types.",
        dest="mode",
        required=True,
    )

    # create the parser for the "agent" sub-command
    parser_agent = sub_parsers.add_parser("cryodrgn", help="cryoDRGN mode")
    parser_agent.add_argument(
        "--config",
        type=str,
        help="config file (.pkl)",
        required=True
    )
    parser_agent.add_argument(
        "--weights",
        type=str,
        help="cryoDRGN weights file containing learned parameters (.pkl)",
        required=True
    )
    parser_agent.add_argument(
        "--z_space",
        type=str,
        help="cryoDRGN latent space (z-space) file containing learned per-particle latent variables (.pkl)",
        required=True
    )
    parser_agent.add_argument(
        "--apix",
        type=float,
        help="Pixel size in Angstrom per pixel.",
        required=True
    )
    parser_agent.add_argument(
        "--output",
        type=valid_output,
        help="Name of the output file. If no extension given, string will be appended. (.npz)",
        required=True
    )


    # create the parse for the "learner" sub-command
    parser_learner = sub_parsers.add_parser("3dva", help="cryoSPARC 3DVA mode")
    parser_learner.add_argument(
        "--map",
        type=valid_mrc,
        help="Consensus or basis map (.mrc)",
        required=True
    )
    parser_learner.add_argument(
        "--components",
        type=valid_mrc,
        help="List of component maps. Accepts wildcard e.g. components*.mrc (.mrc)",
        nargs='+',
        required=True
    )
    parser_learner.add_argument(
        "--particles",
        type=valid_cs,
        help="cryoSPARC particle.cs file containing per-particle latent variables (.cs)",
        required=True
    )
    parser_learner.add_argument(
        "--output",
        type=valid_output,
        help="Name of the output file. If no extension given, string will be appended. (.npz)",
        required=True
    )

    # create the parse for the "tester" sub-command
    parser_tester = sub_parsers.add_parser("3dflex", help="cryoSPARC 3DFlex mode")
    parser_tester.add_argument(
        "-t",
        "--max_steps",
        type=int,
        help="Number of agent's steps",
        default=int(1e6),
    )
    parser_tester.add_argument(
        "--render", action="store_true", help="Render the environment"
    )
    parser_tester.add_argument(
        "-f", "--model_path", type=str, help="Path to saved model"
    )

    args = my_parser.parse_args()
    PackItUp(args)
