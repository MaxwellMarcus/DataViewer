import numpy as np

def get_projection(datapoints, v1, v2):
  v2_orthogonal = v2 - (np.dot(v2, v1) / np.dot(v1, v1) * v1) # Get the component of v2 that is orthogonal to v1
  proj_matrix = np.array([v1 / np.dot(v1, v1), v2_orthogonal / np.dot(v2_orthogonal, v2_orthogonal)])
  # proj_matrix = np.array([red_blue/(np.dot(red_blue, red_blue)), red_yellow/np.dot(red_yellow, red_yellow)])
  proj = np.matmul( datapoints, proj_matrix.T ) # Get the components of each of the datapoints that are orthogonal to v1 and v2_orthogonal
  proj[:, 0] = proj[:,0] - proj[:,1] * np.dot(v1, v2) / np.dot(v1, v1) # Add back in the component of v2 along v1
  return proj