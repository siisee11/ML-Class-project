import numpy as np
from time import sleep

class SoftmaxClassifier:
	def __init__(self, num_features, num_label):
		self.num_features = num_features
		self.num_label = num_label
		self.W = np.zeros((self.num_features, self.num_label))

	def train(self, x, y, epochs, batch_size, lr, optimizer):
		"""
		N : # of training data
		D : # of features
		C : # of classes

		[INPUT]
		x : (N, D), input data (first column is bias for all data)
		y : (N, )
		epochs: (int) # of training epoch to execute
		batch_size : (int) # of minibatch size
		lr : (float), learning rate
		optimizer : (Python class) Optimizer

		[OUTPUT]
		final_loss : (float) loss of last training epoch

		[Functionality]
		Given training data, hyper-parameters and optimizer, execute training procedure.
		Training should be done in minibatch (not the whole data at a time)
		Procedure for one epoch is as follow:
		- For each minibatch
			- Compute probability of each class for data
			- Compute softmax loss
			- Compute gradient of weight
			- Update weight using optimizer
		* loss of one epoch = Mean of minibatch losses
		(minibatch losses = [0.5, 1.0, 1.0, 0.5] --> epoch loss = 0.75)

		"""
#		print('========== TRAINING START ==========')
		final_loss = None   # loss of final epoch
		num_data, num_feat = x.shape
		losses = []
		for epoch in range(1, epochs + 1):
			batch_losses = []   # list for storing minibatch losses

		# ========================= EDIT HERE ========================

			for i in np.arange(0, num_data, batch_size):
				batch_x = x[i:i + batch_size]
				batch_y = y[i:i + batch_size]

#				smax, smin = batch_x.max(), batch_x.min()
#				if smax - smin > 0 :
#					batch_x = (batch_x-smin)/(smax - smin)

				score = batch_x.dot(self.W)

	
				prob = self._softmax(score)
				batch_losses.append(self.softmax_loss(prob, batch_y))
				grad = self.compute_grad(batch_x, self.W, prob, batch_y)
				self.W = optimizer.update(self.W, grad, lr)

		# ============================================================
			epoch_loss = sum(batch_losses) / len(batch_losses)  # epoch loss
			# print loss every 10 epoch
#			if epoch % 10 == 0:
#				print('Epoch %d : Loss = %.4f' % (epoch, epoch_loss))
			# store losses
			losses.append(epoch_loss)
		final_loss = losses[-1]

		return final_loss

	def eval(self, x):
		"""

		[INPUT]
		x : (N, D), input data

		[OUTPUT]
		pred : (N, ), predicted label for N test data

		[Functionality]
		Given N test data, compute probability and make predictions for each data.
		"""
		pred = None
		# ========================= EDIT HERE ========================
		score = x.dot(self.W)
		pred = np.argmax(score , 1)

		# ============================================================
		return pred

	def softmax_loss(self, prob, label):
		"""
		N : # of minibatch data
		C : # of classes

		[INPUT]
		prob : (N, C), probability distribution over classes for N data
		label : (N, ), label for each data

		[OUTPUT]
		softmax_loss : scalar, softmax loss for N input

		[Functionality]
		Given probability and correct label, compute softmax loss for N minibatch data
		"""
		softmax_loss = 0.0
		# ========================= EDIT HERE ========================
		N = prob.shape[0]
		softmax_loss = -np.log(prob[range(N), label])
		softmax_loss = np.sum(softmax_loss)/N
		# ============================================================
		return softmax_loss

	def compute_grad(self, x, weight, prob, label):
		"""
		N : # of minibatch data
		D : # of features
		C : # of classes

		[INPUT]
		x : (N, D), input data
		weight : (D, C), Weight matrix of classifier
		prob : (N, C), probability distribution over classes for N data
		label : (N, ), label for each data. (0 <= c < C for c in label)

		[OUTPUT]
		gradient of weight: (D, C), Gradient of weight to be applied (dL/dW)

		[Functionality]
		Given input (x), weight, probability and label, compute gradient of weight.
		"""
		grad_weight = np.zeros_like(weight, dtype=np.float32) # (D, C)
		# ========================= EDIT HERE ========================

		dprob = prob
		N = x.shape[0]
		dprob[range(N), label] -= 1
		grad_weight = np.dot(x.T, dprob) / N

		# ============================================================
		return grad_weight


	def _softmax(self, x):
		"""
		[INPUT]
		x : (N, C), score before softmax

		[OUTPUT]
		softmax : (same shape with x), softmax distribution over axis-1

		[Functionality]
		Given an input x, apply softmax function over axis-1 (classes).
		"""
		softmax = None
		# ========================= EDIT HERE ========================
		ex = np.exp(x - x.max(1, keepdims=True))
		softmax = ex /np.reshape(np.sum(ex, axis=1), (-1, 1))
		# ============================================================
		return softmax
