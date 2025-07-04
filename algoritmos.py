import numpy as np

def reconstruct_cgnr(H: np.ndarray, g: np.ndarray, max_iterations: int, tol=5e-3, min_iterations=10, lambda_reg: float = 0.0, logger=None) -> tuple:
    m, n = H.shape
    H_norm = np.linalg.norm(H)
    g_norm = np.linalg.norm(g)
    if H_norm == 0 or g_norm == 0:
        raise ValueError("H ou g sao nulos, impossivel normalizar.")
    Hn = H / H_norm
    gn = g.reshape(-1, 1) / g_norm

    if lambda_reg > 0:
        Hn = np.vstack([Hn, np.sqrt(lambda_reg) * np.eye(n)])
        gn = np.vstack([gn, np.zeros((n, 1))])

    tol = max(tol, 1e-12)
    tol = tol * np.linalg.norm(gn)

    f = np.zeros((n, 1))
    r = gn - Hn @ f
    z = Hn.T @ r
    p = z.copy()
    initial_residual_norm = np.linalg.norm(r)
    number_iterations = 0
    min_div = 1e-12

    if logger is not None:
        logger.info(f"CGNR normalizado: H_norm={H_norm:.3e}, g_norm={g_norm:.3e}, tol={tol:.3e}, lambda={lambda_reg:.2e}")

    while number_iterations < max_iterations:
        w = Hn @ p
        z_dot = float(z.T @ z)
        w_dot = float(w.T @ w)

        alpha = z_dot / (w_dot + min_div)
        f_new = f + alpha * p
        r_new = r - alpha * w
        z_new = Hn.T @ r_new
        z_new_dot = float(z_new.T @ z_new)
        beta = z_new_dot / (z_dot + min_div)
        p_new = z_new + beta * p

        current_residual_norm = np.linalg.norm(r_new)
        relative_error = current_residual_norm / (initial_residual_norm + min_div)

        if logger is not None:
            logger.info(
                f"Iteracao {number_iterations + 1}: erro relativo = {relative_error:.6e}, "
                f"residuo = {current_residual_norm:.3e}"
            )
        if relative_error < tol:
            if logger is not None:
                logger.info(f"Convergiu com erro relativo {relative_error:.2e} < {tol:.2e}")
            f = f_new
            break

        f, r, z, p = f_new, r_new, z_new, p_new
        number_iterations += 1

    f = f * (g_norm / H_norm)
    f = np.maximum(f, 0)
    final_residual = g.reshape(-1, 1) - H @ f.reshape(-1, 1)
    final_error = np.linalg.norm(final_residual) / (np.linalg.norm(g) + min_div)

    return f, number_iterations, final_error

def reconstruct_cgne(H: np.ndarray, g: np.ndarray, max_iterations: int, tol=1e-6, min_iterations=10, reg_factor: float = 0.0, logger=None) -> tuple[np.ndarray, int, float]:
    N = H.shape[1]
    f = np.zeros((N, 1), dtype=np.float32)
    g = g.reshape(-1, 1)
    r = g - H @ f
    p = H.T @ r
    initial_residual_norm = np.linalg.norm(r)
    final_iterations = 0
    min_div = 1e-12

    for i in range(1, max_iterations + 1):
        Hp = H @ p
        alpha_num = float(r.T @ r)
        alpha_den = float(Hp.T @ Hp) + min_div
        if alpha_den < min_div:
            break
        alpha = alpha_num / alpha_den
        f_new = f + alpha * p
        r_new = r - alpha * (H @ p)
        beta_num = float(r_new.T @ r_new)
        beta_den = float(r.T @ r) + min_div
        beta = beta_num / beta_den
        p_new = H.T @ r_new + beta * p

        current_residual_norm = np.linalg.norm(r_new)
        relative_error = current_residual_norm / (initial_residual_norm + min_div)

        if logger is not None:
            logger.info(f"Iteracao {i}: erro relativo = {relative_error:.6f}")

        f, r, p = f_new, r_new, p_new
        final_iterations = i

        if i > min_iterations and relative_error < tol:
            if logger is not None:
                logger.info(f"Convergiu com erro relativo {relative_error:.2e} < {tol:.2e}")
            break

    # Regularizacao e nao-negatividade
    if reg_factor > 0.0:
        f = np.maximum(f - reg_factor, 0)
    else:
        f = np.maximum(f, 0)

    final_error = np.linalg.norm(g - H @ f) / (np.linalg.norm(g) + min_div)
    return f.flatten(), final_iterations, final_error
