import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N=X.shape[0]
  C=W.shape[1]
  D=X.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  den = 0
  z=np.zeros(shape=(N,C))
  for i in range(N):
    den=0
    for j in range(C):
        z[i,j]=X[i,:].dot(W[:,j])
    z[i,:] -= np.max(z[i,:])
    z[i,:] = np.exp(z[i,:])
    den = np.sum(z[i,:])
    z[i,:]/=den
    loss -= np.log(z[i,y[i]])
  z=-z
  for i in range(N):
    for j in range(C):
        if(j==y[i]):
            z[i,j]+=1
        dW[:,j]+=z[i,j]*X[i,:]
        
    
        
  dW/=N
  dW+=reg*W
  loss = loss/N + reg*np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N=X.shape[0]
  C=W.shape[1]
  D=X.shape[1]
  z=np.zeros(shape=(N,C))
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  z=X.dot(W)
  z-=np.max(z)
    
  z_correct = z[range(N), y]

  #print(z)
  loss=np.mean(-np.log( np.exp(z_correct)/np.sum(np.exp(z)) ))
  loss=loss+0.5*reg*np.sum(W*W)

  z=np.exp(z)
  z/=np.sum(z,axis=1).reshape(N,1)
  
  z[range(N),y]-=1
  dW=X.T.dot(z)  
  dW=dW/N+reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

