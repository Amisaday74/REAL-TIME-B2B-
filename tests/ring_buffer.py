from multiprocessing import Array, Process, Event
import time
import random



def test_ring_buffer_array(eno1_buffer, event):
    while (True):
        time.sleep(4)
        # Generate a list of 10 random integers between 1 and 100 (inclusive)
        random_integers = [random.randint(1, 1000) for _ in range(10)]
        print(f"Generated random integers: {random_integers}")
        for i, val in enumerate(random_integers):
            eno1_buffer[i] = val
        print(f"Data written to ring buffer array: {list(eno1_buffer[:len(random_integers)])}")
        event.set()

def read_ring_buffer_array(eno1_buffer, event):
    while (True):
        event.wait()
        data = list(eno1_buffer)
        print(f"Data read from ring buffer array: {data}")
        event.clear()


if __name__ == '__main__':
    eno1_datach1 = Array('d', 10)
    event = Event()

    q1 = Process (target=test_ring_buffer_array, args=[eno1_datach1, event])
    q2 = Process (target=read_ring_buffer_array, args=[eno1_datach1, event])

    q1.start()
    q2.start()
    q1.join()
    q2.join()

