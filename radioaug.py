import numpy as np


def Breakage(signal):
    """
    Break a signal into two parts, then apply random width zero-padding to both ends

    Parameters:
    - signal (numpy.ndarray): Input signal sequence, a NumPy array with shape (2, 128) (I/Q signals)

    Returns:
    - sequence_1 (numpy.ndarray): The first new signal
    - sequence_2 (numpy.ndarray): The second new signal
    """
    
    # Get the size of the input signal
    _, input_width = signal.shape

    # Randomly select the break point
    break_point = np.random.randint(int(0.4 * input_width), int(0.6 * input_width))

    # Break the signal
    signal_1 = signal[:, :break_point]
    signal_2 = signal[:, break_point:]

    # Create two zero sequences, each with length 128
    sequence_1 = np.zeros((2, 128), dtype=np.float32)
    sequence_2 = np.zeros((2, 128), dtype=np.float32)

    # Place the broken sub-signals at random positions in the zero sequences
    insert_point_1 = np.random.randint(0, 128 - signal_1.shape[1] + 1)
    sequence_1[:, insert_point_1:insert_point_1 + signal_1.shape[1]] = signal_1

    insert_point_2 = np.random.randint(0, 128 - signal_2.shape[1] + 1)
    sequence_2[:, insert_point_2:insert_point_2 + signal_2.shape[1]] = signal_2

    return sequence_1, sequence_2


def Inversion(signal):
    """
    Reverse a segment of the signal

    Parameters:
    - signal (numpy.ndarray): Input signal sequence, shape (2, 128) (I/Q signals)

    Returns:
    - sig_copy (numpy.ndarray): New signal with a segment reversed
    """
    
    # Get the dimensions of the signal
    _, signal_length = signal.shape

    # Randomly select the length of the segment to reverse
    inv_length = np.random.randint(int(0.8 * signal_length), int(1.0 * signal_length))

    sig_copy = signal.copy()

    # Randomly select the starting position of the segment to reverse
    inv_start = np.random.randint(0, signal_length - inv_length + 1)

    # Get the segment to be reversed
    copy_segment = sig_copy[:, inv_start:inv_start + inv_length].copy()

    # Reverse the segment
    sig_copy[:, inv_start:inv_start + inv_length] = copy_segment[:, ::-1]

    return sig_copy


def Intdel(signal):   
    """
    Generate two new signals by randomly cutting out the middle part of the signal 
    and merging the two ends.

    Parameters:
    - signal (numpy.ndarray): Original signal with shape [2, 128].

    Returns:
    - signal1 (numpy.ndarray): Augmented signal 1.
    - signal2 (numpy.ndarray): Augmented signal 2.
    """

    sig_len = signal.shape[-1]

    # Randomly select the length of the middle segment to delete
    cut_length = np.random.randint(int(0.5 * sig_len), int(0.7 * sig_len))

    # Ensure cut_length is within a valid range
    cut_start = np.random.randint(0, signal.shape[1] - cut_length + 1)

    # Cut the random portion from the middle
    cut_signal = signal[:, cut_start : cut_start + cut_length]
    remain_signal = np.concatenate([signal[:, :cut_start], signal[:, cut_start + cut_length:]], axis=1)

    # Create two zero sequences, each with length 128
    signal_1 = np.zeros((2, 128), dtype=np.float32)
    signal_2 = np.zeros((2, 128), dtype=np.float32)

    # Place the cut segment in a random position in the zero sequence
    insert_point_1 = np.random.randint(0, 128 - cut_signal.shape[1] + 1)
    signal_1[:, insert_point_1:insert_point_1 + cut_signal.shape[1]] = cut_signal

    # Place the remaining signal in a random position in the zero sequence
    insert_point_2 = np.random.randint(0, 128 - remain_signal.shape[1] + 1)
    signal_2[:, insert_point_2:insert_point_2 + remain_signal.shape[1]] = remain_signal

    return signal_1, signal_2

