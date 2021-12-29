
import unittest

from numpy import e
from lib.memory import CircularBuffer, UniformMemory, sample_batch_idx

class TestHelperFunctions(unittest.TestCase):
    def test_sample_batch_index(self):
        ### Sampling without replacement
        
        # Assert that all numbers are sampled
        LOW, HIGH, SIZE = 0, 10, 10
        ret = sample_batch_idx(LOW, HIGH, SIZE)
        self.assertTrue(len(ret) == SIZE)
        for num in range(LOW, HIGH):
            self.assertTrue(num in ret) 
        

        ### Sampling with replacement

        # Assert that low to high included
        LOW, HIGH, SIZE = 0, 10, 1000 #<- (1/10000) # chance that this does not happen.
        ret = sample_batch_idx(LOW, HIGH, SIZE)
        self.assertTrue(len(ret) == SIZE)
        self.assertTrue( LOW in ret   )
        self.assertTrue( HIGH-1 in ret)
        

        

class TestCircularBuffer(unittest.TestCase):
    def test_buffer_init(self):
        buffer = CircularBuffer(max_size=10_000)

    def test_buffer_append(self):
        buffer = CircularBuffer(max_size=10_000)
        equivalent_array = []

        HIGH = 10_000
        for i in range(HIGH):
            buffer.append(i)
            equivalent_array.append(i)
        for i in range(HIGH):
            self.assertTrue(buffer[i] == i) # ensure items are appended correctly

        self.assertTrue(len(buffer) == HIGH) # assert that size grows properly
        self.assertTrue(buffer.to_list() == equivalent_array) # assert that to_list works

    def test_buffer_wrapping(self):
        SIZE = 100
        buffer = CircularBuffer(max_size=100)
        equivalent_array = []

        for i in range(SIZE + 1):  # overflow [100, 1, 2, 3, 4]
            buffer.append(i)
            equivalent_array.append(i)
        
        equivalent_array.pop(0) # simulate sliding window

        # test same logic
        self.assertTrue(len(buffer) == SIZE) # assert that size grows properly
        self.assertTrue(buffer.to_list() == equivalent_array) # assert that to_list works

    def test_buffer_wrapping_intense(self):
        SIZE = 100
        buffer = CircularBuffer(max_size=SIZE)
        equivalent_array = []

        for i in range(12312):
            buffer.append(i)
            equivalent_array.append(i)

            if len(equivalent_array) > SIZE:
                equivalent_array.pop(0)
        
        for i in range(SIZE):
            # check if indices work correctly
            self.assertTrue(buffer[i] == equivalent_array[i])

        self.assertTrue(len(buffer) == SIZE) # assert that size grows properly
        self.assertTrue(buffer.to_list() == equivalent_array) # assert that to_list works


class TestUniformMemory(unittest.TestCase):
    def test_memory(self):
        pass


if __name__ == '__main__':
    unittest.main()