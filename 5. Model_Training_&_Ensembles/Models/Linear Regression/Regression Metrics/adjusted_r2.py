def adjusted_r2(r2,X_test):
  r2 = r2
  n = X_test.shape[0]
  k = X_test.shape[1]
  adjusted_r2_score = 1-(((1-r2)*(n-1))/(n-k-1))
  return adjusted_r2_score