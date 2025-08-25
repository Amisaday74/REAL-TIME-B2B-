from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from brainflow import BoardShim, DataFilter, DetrendOperations
from brainflow.board_shim import BrainFlowInputParams, BoardIds, LogLevels
import pandas as pd
import time
import sys
import logging

# ------------------- Logging Configuration ------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("eeg_data_stream.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

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
        self.total_frames = 0
        self.dropped_frames = 0
        self.start_time = time.time()

        # Initialize the application and plot window
        self.app = QtWidgets.QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(show=True, title="Real-Time EEG Data")
        self._init_timeseries()
        self._init_stats_widget()

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

    def _init_stats_widget(self):
        """Initialize the stats widget for frame monitoring."""
        self.stats = pg.LabelItem(justify='right')
        self.win.addItem(self.stats, row=len(self.eeg_channels), col=0)

    @QtCore.pyqtSlot(object)
    def update_plot(self, data):
        """Update plot with new data."""
        self.total_frames += 1

        # Update statistics
        if data.shape[1] < 1000:
            self.dropped_frames += 1
        
        elapsed = time.time() - self.start_time
        self.stats.setText(
            f"Total Frames: {self.total_frames} | Dropped Frames: {self.dropped_frames} | Elapsed Time: {elapsed:.1f}s"
        )
        logging.info(f"Frames: {self.total_frames}, Dropped: {self.dropped_frames}, Elapsed: {elapsed:.1f}s")

        for count, channel in enumerate(self.eeg_channels):
            self.curves[count].setData(data[channel].tolist())
        self.app.processEvents()
    
    @QtCore.pyqtSlot()
    def close_app(self):
        """Handle graceful shutdown."""
        logging.info("Closing application gracefully...")
        self.running = False
        self.app.quit()

# ------------------- Main Data Collection Loop ------------------- #
def main():
    logging.info("Initializing BrainFlow BoardShim...")
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    board_id = BoardIds.SYNTHETIC_BOARD.value
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    board = BoardShim(board_id, params)
    
    # Start Board Session
    board.prepare_session()
    board.start_stream(45000)
    logging.info('---- Starting the EEG Data Streaming ----')

    # Initialize the Graph in a separate thread
    graph = Graph(eeg_channels, sampling_rate)

    try:
        while graph.running:
            start_time = time.time()
            data = board.get_current_board_data(1000)  

            # Check the data length:
            if data.shape[1] < 1000:
                logging.warning(f"Data packet too small: {data.shape[1]} samples. Attempting to recover...")
                time.sleep(0.1)
                continue

            logging.info(f"Successfully acquired {data.shape[1]} samples from the board.")

            # Preprocessing
            for eeg_channel in eeg_channels:
                DataFilter.detrend(data[eeg_channel], DetrendOperations.LINEAR.value)
                DataFilter.remove_environmental_noise(data[eeg_channel], sampling_rate, noise_type=1)
                DataFilter.perform_lowpass(data[eeg_channel], sampling_rate, cutoff=100, order=4, filter_type=0, ripple=0)
                DataFilter.perform_highpass(data[eeg_channel], sampling_rate, cutoff=0.1, order=4, filter_type=0, ripple=0)

            # Send data to the graph in the background
            graph.data_signal.emit(data)

            # Ensure the loop runs every ~4 seconds
            elapsed = time.time() - start_time
            if elapsed < 4.0:
                time.sleep(4.0 - elapsed)

    except KeyboardInterrupt:
        logging.info("Keyboard Interrupt Detected. Closing gracefully...")
        graph.close_signal.emit()

    except Exception as e:
        logging.error(f"Critical error occurred: {e}")
        graph.close_signal.emit()

    finally:
        board.stop_stream()
        board.release_session()
        logging.info("Session closed successfully.")

if __name__ == "__main__":
    main()
