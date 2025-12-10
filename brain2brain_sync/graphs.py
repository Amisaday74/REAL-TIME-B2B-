from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from brainflow import BoardShim, DataFilter, DetrendOperations
from brainflow.board_shim import BrainFlowInputParams, BoardIds, LogLevels
import pandas as pd
import time
import sys

# ------------------- Graph Class with QThread ------------------- #
class Graph(QtCore.QThread):
    data_signal = QtCore.pyqtSignal(object)       # raw data
    processed_data = QtCore.pyqtSignal(object)    # processed data

    def __init__(self, eeg_channels, sampling_rate):
        super().__init__()
        self.eeg_channels = eeg_channels
        self.sampling_rate = sampling_rate
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        self.running = True   # plot enabled/disabled

        # ---------------- GUI WINDOW ----------------
        self.win = pg.GraphicsLayoutWidget(show=True, title="Real-Time EEG Data")
        self.win.closeEvent = self._handle_close_event  # intercept close()

        self._init_timeseries()
        self._init_processed()

        # ---------------- SIGNAL CONNECTIONS ----------------
        self.data_signal.connect(self.update_plot)
        self.processed_data.connect(self.update_processed)

    # ---------------- PLOTS ----------------
    def _init_timeseries(self):
        self.plots = []
        self.curves = []
        for idx, ch in enumerate(self.eeg_channels):
            p = self.win.addPlot(row=idx, col=0)
            p.showAxis('left', False)
            p.showAxis('bottom', False)
            if idx == 0:
                p.setTitle("EEG Raw Data")
            curve = p.plot()
            self.plots.append(p)
            self.curves.append(curve)

    def _init_processed(self):
        self.plots2 = []
        self.curves2 = []
        for idx, ch in enumerate(self.eeg_channels):
            p2 = self.win.addPlot(row=idx, col=1)
            p2.showAxis('left', False)
            p2.showAxis('bottom', False)
            if idx == 0:
                p2.setTitle("Processed Signal")
            curve2 = p2.plot()
            self.plots2.append(p2)
            self.curves2.append(curve2)

    # ---------------- UPDATE RAW ----------------
    @QtCore.pyqtSlot(object)
    def update_plot(self, data):
        if not self.running:
            return
        for idx, ch in enumerate(self.eeg_channels):
            self.curves[idx].setData(data[ch].tolist())
        QtWidgets.QApplication.processEvents()

    # ---------------- UPDATE PROCESSED ----------------
    @QtCore.pyqtSlot(object)
    def update_processed(self, data):
        if not self.running:
            return
        for idx, ch in enumerate(self.eeg_channels):
            self.curves2[idx].setData(data[ch].tolist())
        QtWidgets.QApplication.processEvents()

    # -----------------------------------------------------------
    # CLOSE EVENT OVERRIDE → hide instead of destroying window
    # -----------------------------------------------------------
    def _handle_close_event(self, event):
        """
        The user clicks the window X button.
        → DO NOT close the app
        → DO NOT quit Qt
        → Simply hide the window and pause plotting
        """
        print("[Graph] Window hidden — acquisition continues.")

        self.running = False   # pause plotting
        self.win.hide()        # hide GUI window
        event.ignore()         # prevent object destruction
# ------------------- Main Data Collection Loop ------------------- #
def main():
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    board_id = BoardIds.SYNTHETIC_BOARD.value  # Replace with your board ID
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    board = BoardShim(board_id, params)
    
    # Start Board Session
    board.prepare_session()
    board.start_stream(45000)
    print('---- Starting the EEG Data Streaming ----')

    # Initialize the Graph in a separate thread
    graph = Graph(eeg_channels, sampling_rate)

    while True:
        time.sleep(1)  # 4-second window
        data = board.get_board_data()  
        graph.data_signal.emit(data)
        # make a copy for processing
        processed_data = data.copy()
        
        # Preprocessing
        for eeg_channel in eeg_channels:
            DataFilter.detrend(processed_data[eeg_channel], DetrendOperations.LINEAR.value)
            DataFilter.remove_environmental_noise(processed_data[eeg_channel], sampling_rate, noise_type=1)
            DataFilter.perform_lowpass(processed_data[eeg_channel], sampling_rate, cutoff=40, order=4, filter_type=0, ripple=0)
            DataFilter.perform_highpass(processed_data[eeg_channel], sampling_rate, cutoff=0.1, order=4, filter_type=0, ripple=0)

        # Send data to the graph in the background
        graph.processed_data.emit(processed_data)

if __name__ == "__main__":
    main()