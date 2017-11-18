import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    s=0
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        dW[:,y[i]]-=X[i,:].T
        dW[:,j]+=X[i,:].T
        loss += margin
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW=dW/num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW = dW + reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #print("inside svm_loss_vectorized")
  #print("W  = ", W)
  #print("X  = ", X)
  #print("y  = ", y)
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  N=X.shape[0]
  D=X.shape[1]
  C=W.shape[1]
    
  #print("N D C", N,D,C)  
  S = X.dot(W)
  #print("S Shape = ",S.shape)
  correct_scores = S[range(N),y]
  correct_scores = correct_scores.reshape(correct_scores.shape[0],1)
  #print(correct_scores)
  S=S-correct_scores+1
  S[range(N),y]=0
  #print(S)
  S[S<0]=0
  #print("After 0 ",S)
  loss = np.sum(S)/N + reg*np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  binary = S
  binary[binary>0]=1
  row_sum=np.sum(binary, axis=1)
  binary[range(N),y]-=row_sum
  dW=X.T.dot(binary)
  dW/=N
  dW+=(reg*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
