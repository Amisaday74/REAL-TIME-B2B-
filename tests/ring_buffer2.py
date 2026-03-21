from multiprocessing import Array, Process, Event
import numpy as np
import time
import random
N_CHANNELS = 2
BUFFER_LEN = 10   # samples per channel



def writer(shared_array, event):
    buffer = np.frombuffer(shared_array, dtype=np.float64)
    buffer = buffer.reshape(2, -1)

    while True:
        time.sleep(4)

        # Generate two independent channels
        ch1 = np.random.randint(1, 1000, size=buffer.shape[1]).astype(np.float64)
        ch2 = np.random.randint(1, 1000, size=buffer.shape[1]).astype(np.float64)

        # Stack them to create the same (2, N) structure
        new_data = np.vstack((ch1, ch2))

        print("Generated data:")
        print(new_data)

        # Same write operation as before
        buffer[:, :] = new_data

        print("Written to shared buffer:")
        print(buffer)

        event.set()
def reader(shared_array, event):
    buffer = np.frombuffer(shared_array, dtype=np.float64)
    buffer = buffer.reshape(2, -1)

    while True:
        event.wait()

        # Make a safe copy for processing
        data = buffer.copy()

        print("Read from shared buffer:")
        print(data)

        event.clear()


if __name__ == '__main__':
    shared_raw = Array('d', 2 * 10, lock=False)
    event = Event()

    p_writer = Process(target=writer, args=(shared_raw, event))
    p_reader = Process(target=reader, args=(shared_raw, event))

    p_writer.start()
    p_reader.start()

    p_writer.join()
    p_reader.join()

