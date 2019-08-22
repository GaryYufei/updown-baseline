import numpy as np
import torch

class cbs_matrix:

    def __init__(self, vocab_size):
        self.matrix = None
        self.vocab_size = vocab_size

    def init_matrix(self, state_size):
        self.matrix = np.zeros((1, state_size, state_size, self.vocab_size), dtype=np.uint8)

    def add_connect(self, from_state, to_state, w_group):
        assert self.matrix is not None
        for w_index in w_group:
            self.matrix[0, from_state, to_state, w_index] = 1
            self.matrix[0, from_state, from_state, w_index] = 0

    def init_row(self, state_index):
        assert self.matrix is not None
        self.matrix[0, state_index, state_index, :] = 1

    def get_matrix(self):
        return self.matrix

class FreeConstraint:

	def __init__(self, output_size):
		self.M = cbs_matrix(output_size)

	def select_state_func(self, all_predictions):
		return all_predictions[:, 0, 0]

	def get_state_matrix(self, batch_size, device_id):
		self.M.init_matrix(1)
		self.M.init_row(0)
		state_transform = torch.from_numpy(self.M.get_matrix()).to(device_id)
		return state_transform.expand(batch_size, -1, -1, -1)




