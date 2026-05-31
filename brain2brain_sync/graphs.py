from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from brainflow import BoardShim, DataFilter, DetrendOperations
from brainflow.board_shim import BrainFlowInputParams, BoardIds
import time
import numpy as np

# ------------------- Graph Class with QThread ------------------- #
#
# BUFFERED CHUNK-PLOTTING MECHANISM — HOW IT WORKS
# ─────────────────────────────────────────────────
# EEG_device.py collects data in fixed time windows (e.g. timewindow_seconds = 4 s).
# At the end of each window it emits a NumPy block of shape [channels, sampling_rate * timewindow_seconds]
# to this Graph via update_plot() / update_processed().
#
# Rather than drawing the full 4-second block in one jump — which would look like the
# trace snapping forward abruptly — the graph stores the incoming block in an internal
# buffer and replays it in smaller slices called *chunks*:
#
#   chunk_interval_seconds  →  how many seconds of data each tick renders
#   chunk_size              →  samples per chunk = sampling_rate × chunk_interval_seconds
#   _chunk_timer interval   →  fires every chunk_interval_seconds (in ms)
#
# Example  (timewindow = 4 s, chunk_interval = 1 s, sampling_rate = 250 Hz):
#   • EEG_device sends 1000-sample block every 4 s.
#   • _chunk_timer fires every 1 s and draws 250 samples per tick → 4 ticks to drain the buffer.
#   • Result: the display scrolls forward smoothly in 1-second steps.
#
# CONSTRAINT: timewindow_seconds must be an integer multiple of chunk_interval_seconds
# so that every incoming block is divided into a whole number of chunks with no remainder.
# Valid chunk_interval values for a 4-second window: 0.5 s, 1 s, 2 s, 4 s.
#
# INHERENT DISPLAY DELAY:
#   Because EEG_device waits for a complete time window before emitting data, the earliest
#   moment the graph can start drawing a window is when that window has fully elapsed.
#   The display therefore always lags behind live EEG by ~timewindow_seconds.  This is a
#   fundamental trade-off of window-based processing, not a bug.
# ─────────────────────────────────────────────────
class Graph(QtCore.QThread):
    data_signal = QtCore.pyqtSignal(object)  # Signal to receive raw data
    processed_data = QtCore.pyqtSignal(object)  # Signal to receive processed data

    def __init__(self, eeg_channels, sampling_rate, title="Real-Time EEG Data", chunk_interval_seconds=1):
        super().__init__()
        self.eeg_channels = eeg_channels
        self.sampling_rate = sampling_rate
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate
        self.running = True
        # Buffers to hold the latest emitted block(s). They store numpy arrays shape [channels, n_samples]
        self.buffer_raw = None
        self.buffer_processed = None
        # index (in samples) for chunked plotting
        self.buffer_idx = 0
        # ── Chunk-interval configuration ──────────────────────────────────────────────
        # chunk_interval_seconds controls how finely the incoming data block is sliced
        # for display.  Smaller values produce smoother scrolling at the cost of more
        # frequent Qt redraws; larger values reduce CPU load but make updates more steppy.
        # Must evenly divide timewindow_seconds (validated in run_RT_B2B_v3.py).
        self.chunk_interval_seconds = chunk_interval_seconds
        # Number of samples rendered per timer tick
        self.chunk_size = int(self.sampling_rate * chunk_interval_seconds)
        # Qt timer that fires once per chunk_interval to advance the display
        self._chunk_timer = QtCore.QTimer()
        self._chunk_timer.setInterval(int(chunk_interval_seconds * 1000))
        self._chunk_timer.timeout.connect(self._on_chunk_timer)

        # Initialize the application and plot window
        self.win = pg.GraphicsLayoutWidget(show=True, title=title)
        # Prevent the user from closing the window manually:
        # - remove the close button from the window frame
        # - install an event filter to intercept and ignore Close events
        try:
            self.win.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
            # re-show to apply the changed flags
            self.win.show()
        except Exception:
            pass
        self.win.installEventFilter(self)
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
        # Accept numpy arrays with shape [channels, n_samples]
        try:
            arr = np.array(data)
        except Exception:
            return

        # If there's already buffered data, concatenate along time axis
        if self.buffer_raw is None:
            self.buffer_raw = arr.copy()
            self.buffer_idx = 0
        else:
            # concat along axis=1 (time)
            self.buffer_raw = np.concatenate((self.buffer_raw, arr), axis=1)

        # Start (or keep) chunk timer to plot every second
        if not self._chunk_timer.isActive():
            self._chunk_timer.start()
        QtWidgets.QApplication.processEvents()

    @QtCore.pyqtSlot(object)
    def update_processed(self, data):
        """Update plot with processed data."""
        try:
            arr = np.array(data)
        except Exception:
            return

        if self.buffer_processed is None:
            self.buffer_processed = arr.copy()
            self.buffer_idx = 0
        else:
            self.buffer_processed = np.concatenate((self.buffer_processed, arr), axis=1)

        if not self._chunk_timer.isActive():
            self._chunk_timer.start()
        QtWidgets.QApplication.processEvents()

    def _on_chunk_timer(self):
        """Fires every chunk_interval_seconds to advance the display by one chunk.

        Reads the next chunk_size samples from the internal buffers and pushes
        them to the plot curves.  When both buffers have been fully drained the
        timer stops until new data arrives via update_plot / update_processed.
        """
        plotted_any = False

        # Prefer processed buffer for processed plots, raw for raw plots; both may exist
        if self.buffer_raw is not None:
            width = self.buffer_raw.shape[1]
            start = self.buffer_idx
            end = min(start + self.chunk_size, width)
            for count, channel in enumerate(self.eeg_channels):
                try:
                    y = self.buffer_raw[channel, start:end]
                except Exception:
                    # if channel indexing is unexpected, fallback to using count
                    y = self.buffer_raw[count, start:end]
                self.curves[count].setData(y.tolist())
            plotted_any = True

        if self.buffer_processed is not None:
            width_p = self.buffer_processed.shape[1]
            start_p = self.buffer_idx
            end_p = min(start_p + self.chunk_size, width_p)
            for count, channel in enumerate(self.eeg_channels):
                try:
                    y = self.buffer_processed[channel, start_p:end_p]
                except Exception:
                    y = self.buffer_processed[count, start_p:end_p]
                self.curves2[count].setData(y.tolist())
            plotted_any = True

        if plotted_any:
            QtWidgets.QApplication.processEvents()

        # Advance index; if both buffers exhausted, stop timer
        # Use the largest available width to know when to stop
        max_width = 0
        if self.buffer_raw is not None:
            max_width = max(max_width, self.buffer_raw.shape[1])
        if self.buffer_processed is not None:
            max_width = max(max_width, self.buffer_processed.shape[1])

        self.buffer_idx += self.chunk_size
        if self.buffer_idx >= max_width:
            # finished plotting current buffered data
            self._chunk_timer.stop()
            self.buffer_idx = 0
            self.buffer_raw = None
            self.buffer_processed = None
        

    # -----------------------------------------------------------
    # Handle close events (user closes window)
    # -----------------------------------------------------------
    def closeEvent(self, event):
        """Called automatically when the window is closed."""
        print("[Graph] Window closed by user — GUI will stop, acquisition continues.")
        self.running = False
        # stop chunk timer if running
        try:
            if self._chunk_timer.isActive():
                self._chunk_timer.stop()
        except Exception:
            pass
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
            try:
                if self._chunk_timer.isActive():
                    self._chunk_timer.stop()
            except Exception:
                pass
            QtCore.QTimer.singleShot(100, self.app_quit)

    def app_quit(self):
        """Internal helper to quit the app safely."""
        app = QtWidgets.QApplication.instance()
        if app:
            app.quit()

    def eventFilter(self, obj, event):
        """Intercept `Close` events on `self.win` and block them.

        The controlling script should call `graph.close_signal.emit()` or
        `graph.close_app()` to perform a controlled shutdown.
        """
        if obj is getattr(self, 'win', None) and event.type() == QtCore.QEvent.Close:
            print('[Graph] Manual close blocked — use controller to shutdown.')
            try:
                event.ignore()
            except Exception:
                pass
            return True
        return super().eventFilter(obj, event)
# ------------------- Main Data Collection Loop ------------------- #
def main():
    import json
    with open('config.json', 'r') as f:
        config = json.load(f)

    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    board_id = BoardIds.SYNTHETIC_BOARD.value  # Replace with your board ID
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    board = BoardShim(board_id, params)
    chunk_interval = config.get('chunk_interval_seconds', 1)

    # Start Board Session
    board.prepare_session()
    board.start_stream(45000)
    print('---- Starting the EEG Data Streaming ----')

    # Initialize the Graph in a separate thread
    graph = Graph(eeg_channels, sampling_rate, chunk_interval_seconds=chunk_interval)

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