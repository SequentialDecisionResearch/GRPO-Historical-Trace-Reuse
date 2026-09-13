Post-hoc alpha mechanism analysis

Research status: EXPLORATORY / POST-HOC. Not a frozen confirmatory gate.
No model calls, no CUDA, no gate refit, and no new pair selection.

Definitions (prompt i):
  alpha_hat_i = sum_j W_ij^2 R_ij / sum_j W_ij^2
  log kappa2_hat_i = log[(1/K) sum_j W_ij^2]
  A_proxy_i(V) = V^2 + (1 - 2V) alpha_hat_i
  log m2_proxy_i = log kappa2_hat_i + log A_proxy_i(V)

The matched pair uses the same target and the same finite on-policy reference v.
Development results are the main exploratory mechanism analysis; test results are exploratory replication.
The program validates raw recomputed median rESS against T04 before accepting any mechanism result.
