from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from brainflow import BoardShim, DataFilter, DetrendOperations
from brainflow.board_shim import BrainFlowInputParams, BoardIds, LogLevels
import pandas as pd
import time
import sys

# ------------------- Graph Class with QThread ------------------- #
class Graph(QtCore.QThread):
    data_signal = QtCore.pyqtSignal(object)  # Signal to receive raw data
    processed_data = QtCore.pyqtSignal(object)  # Signal to receive processed data

    def __init__(self, eeg_channels, sampling_rate):
        super().__init__()
        self.eeg_channels = eeg_channels
        self.sampling_rate = sampling_rate
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate
        self.running = True

        # Initialize the application and plot window
        self.win = pg.GraphicsLayoutWidget(show=True, title="Real-Time EEG Data (Board 1)")
        self._init_timeseries()
        self._init_processed()

        # Start listening for data signals
        self.data_signal.connect(self.update_plot)
        self.processed_data.connect(self.update_processed)


    def _init_timeseries(self):
        """Initialize the time series plots for each EEG channel."""
        self.plots = []
        self.curves = []
        for i in range(len(self.eeg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('EEG Raw Data')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

    #Plot for processed data
    def _init_processed(self):
        self.plots2 = []
        self.curves2 = []
        for i in range(len(self.eeg_channels)):
            p2 = self.win.addPlot(row=i, col=1)
            p2.showAxis('left', False)
            p2.setMenuEnabled('left', False)
            p2.showAxis('bottom', False)
            p2.setMenuEnabled('bottom', False)
            if i == 0:
                p2.setTitle('Processed Signal')
            self.plots2.append(p2)
            curve2 = p2.plot()
            self.curves2.append(curve2) 

    @QtCore.pyqtSlot(object)
    def update_plot(self, data):
        """Update plot with new data."""
        for count, channel in enumerate(self.eeg_channels):
            self.curves[count].setData(data[channel].tolist())
        QtWidgets.QApplication.processEvents()

    @QtCore.pyqtSlot(object)
    def update_processed(self, data):
        """Update plot with processed data."""
        for count, channel in enumerate(self.eeg_channels):
            self.curves2[count].setData(data[channel].tolist())
            #if count < data.shape[1]:  # Ensure we don't go out of bounds
                #self.curves2[count].setData(data[:, count].tolist())
        QtWidgets.QApplication.processEvents()
        

    # -----------------------------------------------------------
    # Handle close events (user closes window)
    # -----------------------------------------------------------
    def closeEvent(self, event):
        """Called automatically when the window is closed."""
        print("[Graph] Window closed by user — GUI will stop, acquisition continues.")
        self.running = False
        QtCore.QTimer.singleShot(100, self.app_quit)  # Quit Qt after a short delay
        event.accept()

    # -----------------------------------------------------------
    # Public close method (for programmatic shutdown)
    # -----------------------------------------------------------
    @QtCore.pyqtSlot()
    def close_app(self):
        """Called from outside when we want to close GUI gracefully."""
        if self.running:
            print("[Graph] Close requested programmatically — stopping GUI.")
            self.running = False
            QtCore.QTimer.singleShot(100, self.app_quit)

    def app_quit(self):
        """Internal helper to quit the app safely."""
        app = QtWidgets.QApplication.instance()
        if app:
            app.quit()
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