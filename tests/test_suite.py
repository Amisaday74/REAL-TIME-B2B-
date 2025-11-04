import unittest
from unittest.mock import MagicMock, patch
from brainflow import BoardShim, BrainFlowInputParams, BoardIds
from real_time_plotting import Graph, main
import time


class TestRealTimeEEG(unittest.TestCase):

    @patch.object(BoardShim, 'get_current_board_data')
    def test_data_acquisition_and_frame_drop(self, mock_get_data):
        # Setup Mock
        mock_data = MagicMock()
        mock_data.shape = (8, 1000)
        mock_get_data.return_value = mock_data
        
        params = BrainFlowInputParams()
        board_id = BoardIds.SYNTHETIC_BOARD.value
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        sampling_rate = BoardShim.get_sampling_rate(board_id)

        # Initialize Graph
        graph = Graph(eeg_channels, sampling_rate)
        graph.running = True
        
        # Emit mock data
        graph.data_signal.emit(mock_data)
        
        # Verify frame count increment
        self.assertEqual(graph.total_frames, 1)
        self.assertEqual(graph.dropped_frames, 0)

        # Simulate frame drop
        mock_data.shape = (8, 900)
        graph.data_signal.emit(mock_data)

        # Verify dropped frame is counted
        self.assertEqual(graph.total_frames, 2)
        self.assertEqual(graph.dropped_frames, 1)

    @patch.object(BoardShim, 'get_current_board_data')
    def test_gui_update_and_shutdown(self, mock_get_data):
        # Setup Mock
        mock_data = MagicMock()
        mock_data.shape = (8, 1000)
        mock_get_data.return_value = mock_data
        
        params = BrainFlowInputParams()
        board_id = BoardIds.SYNTHETIC_BOARD.value
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        sampling_rate = BoardShim.get_sampling_rate(board_id)

        # Initialize Graph
        graph = Graph(eeg_channels, sampling_rate)
        graph.running = True
        
        # Emit mock data
        graph.data_signal.emit(mock_data)
        
        # Trigger graceful shutdown
        graph.close_signal.emit()
        self.assertFalse(graph.running)
    
    def test_main_function(self):
        # Test main function to ensure initialization
        with patch('RealTimeEEGVisualization.BoardShim') as MockBoardShim:
            mock_instance = MockBoardShim.return_value
            main()
            mock_instance.prepare_session.assert_called_once()
            mock_instance.start_stream.assert_called_once()
            mock_instance.stop_stream.assert_called_once()
            mock_instance.release_session.assert_called_once()


if __name__ == '__main__':
    unittest.main()

