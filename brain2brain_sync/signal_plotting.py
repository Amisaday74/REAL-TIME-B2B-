from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from brainflow import BoardShim, DataFilter, DetrendOperations
from brainflow.board_shim import BrainFlowInputParams, BoardIds, LogLevels
import pandas as pd
import time
import sys

# ------------------- Graph Class with QThread ------------------- #
class Graph(QtCore.QThread):
    data_signal = QtCore.pyqtSignal(object)  # Signal to receive data
    close_signal = QtCore.pyqtSignal()       # Signal to handle closing

    def __init__(self, eeg_channels, sampling_rate):
        super().__init__()
        self.eeg_channels = eeg_channels
        self.sampling_rate = sampling_rate
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate
        self.running = True

        # Initialize the application and plot window
        self.app = QtWidgets.QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(show=True, title="Real-Time EEG Data (Board 1)")
        self._init_timeseries()

        # Start listening for data signals
        self.data_signal.connect(self.update_plot)
        self.close_signal.connect(self.close_app)
        self.start()

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

    @QtCore.pyqtSlot(object)
    def update_plot(self, data):
        """Update plot with new data."""
        for count, channel in enumerate(self.eeg_channels):
            self.curves[count].setData(data[channel].tolist())
        self.app.processEvents()

    @QtCore.pyqtSlot()
    def close_app(self):
        """Handle graceful shutdown of the graph window only."""
        self.running = False     # stop emitting to the graph
        self.app.quit()          # close Qt event loop

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
        time.sleep(4)  # 4-second window
        data = board.get_board_data()  

        # Preprocessing
        for eeg_channel in eeg_channels:
            DataFilter.detrend(data[eeg_channel], DetrendOperations.LINEAR.value)
            DataFilter.remove_environmental_noise(data[eeg_channel], sampling_rate, noise_type=1)
            DataFilter.perform_lowpass(data[eeg_channel], sampling_rate, cutoff=100, order=4, filter_type=0, ripple=0)
            DataFilter.perform_highpass(data[eeg_channel], sampling_rate, cutoff=0.1, order=4, filter_type=0, ripple=0)

        # Send data to the graph in the background
        graph.data_signal.emit(data)

if __name__ == "__main__":
    main()