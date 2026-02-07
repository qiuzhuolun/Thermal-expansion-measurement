"""
Joint Polynomial Fitting: fit original data and its derivative simultaneously.

Given:
- dataCu: original data (x, y) pairs
- dataCu2: derivative data (x, y') pairs

Goal: Find an 8th degree polynomial P(x) such that:
- P(x) fits dataCu
- P'(x) (7th degree) fits dataCu2

Method: Minimize combined least squares error with equal weights.
"""

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def polynomial_and_derivative(coeffs, x):
    """
    Compute polynomial P(x) and its derivative P'(x).
    
    Parameters:
    -----------
    coeffs : array-like
        Polynomial coefficients [a0, a1, a2, ..., an] for P(x) = a0 + a1*x + a2*x^2 + ...
    x : array-like
        x values
    
    Returns:
    --------
    P : ndarray
        P(x) values
    dP : ndarray
        P'(x) values
    """
    n = len(coeffs)
    x = np.asarray(x)
    
    # P(x) = sum(coeffs[i] * x^i)
    P = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coeffs):
        P += c * x**i
    
    # P'(x) = sum(i * coeffs[i] * x^(i-1)) for i >= 1
    dP = np.zeros_like(x, dtype=float)
    for i in range(1, n):
        dP += i * coeffs[i] * x**(i-1)
    
    return P, dP


def residual_function(coeffs, x_data, y_data, x_deriv, y_deriv, weight_data=1.0, weight_deriv=1.0):
    """
    Compute residuals for joint fitting.
    
    Residual = [weight_data * (P(x) - y_data), weight_deriv * (P'(x) - y_deriv)]
    """
    P, dP = polynomial_and_derivative(coeffs, x_data)
    _, dP_deriv = polynomial_and_derivative(coeffs, x_deriv)
    
    res_data = weight_data * (P - y_data)
    res_deriv = weight_deriv * (dP_deriv - y_deriv)
    
    return np.concatenate([res_data, res_deriv])


def joint_polynomial_fit(x_data, y_data, x_deriv, y_deriv, degree=8, 
                         weight_data=1.0, weight_deriv=1.0, verbose=True):
    """
    Perform joint polynomial fitting for data and its derivative.
    
    Parameters:
    -----------
    x_data, y_data : array-like
        Original data points
    x_deriv, y_deriv : array-like
        Derivative data points
    degree : int
        Polynomial degree for P(x), derivative will be (degree-1)
    weight_data, weight_deriv : float
        Weights for data and derivative fitting (default: equal weights)
    verbose : bool
        Print fitting results
    
    Returns:
    --------
    coeffs : ndarray
        Fitted polynomial coefficients [a0, a1, ..., a_degree]
    result : OptimizeResult
        Full optimization result from scipy
    """
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    x_deriv = np.asarray(x_deriv, dtype=float)
    y_deriv = np.asarray(y_deriv, dtype=float)
    
    # Initial guess: fit only the original data first
    initial_coeffs = np.polyfit(x_data, y_data, degree)[::-1]  # reverse to get [a0, a1, ...]
    
    # Optimize using least squares
    result = least_squares(
        residual_function,
        initial_coeffs,
        args=(x_data, y_data, x_deriv, y_deriv, weight_data, weight_deriv),
        method='lm',  # Levenberg-Marquardt
        verbose=0
    )
    
    coeffs = result.x
    
    if verbose:
        print("=" * 60)
        print("Joint Polynomial Fitting Results")
        print("=" * 60)
        print(f"Polynomial degree: {degree} (derivative degree: {degree-1})")
        print(f"Data points: {len(x_data)} (original), {len(x_deriv)} (derivative)")
        print(f"Weights: data={weight_data}, derivative={weight_deriv}")
        print("-" * 60)
        print("Fitted coefficients (a0 + a1*x + a2*x^2 + ...):")
        for i, c in enumerate(coeffs):
            print(f"  a{i} = {c:+.10e}")
        print("-" * 60)
        
        # Compute fitting errors and R-squared
        P_fit, _ = polynomial_and_derivative(coeffs, x_data)
        _, dP_fit = polynomial_and_derivative(coeffs, x_deriv)
        
        # RMSE
        rmse_data = np.sqrt(np.mean((P_fit - y_data)**2))
        rmse_deriv = np.sqrt(np.mean((dP_fit - y_deriv)**2))
        
        # R-squared: R² = 1 - SS_res / SS_tot
        ss_res_data = np.sum((y_data - P_fit)**2)
        ss_tot_data = np.sum((y_data - np.mean(y_data))**2)
        r2_data = 1 - ss_res_data / ss_tot_data
        
        ss_res_deriv = np.sum((y_deriv - dP_fit)**2)
        ss_tot_deriv = np.sum((y_deriv - np.mean(y_deriv))**2)
        r2_deriv = 1 - ss_res_deriv / ss_tot_deriv
        
        # Max absolute error
        max_err_data = np.max(np.abs(P_fit - y_data))
        max_err_deriv = np.max(np.abs(dP_fit - y_deriv))
        
        print("Fitting Quality Metrics:")
        print(f"  Original data:")
        print(f"    R-squared:        {r2_data:.10f}")
        print(f"    RMSE:             {rmse_data:.6e}")
        print(f"    Max abs error:    {max_err_data:.6e}")
        print(f"  Derivative data:")
        print(f"    R-squared:        {r2_deriv:.10f}")
        print(f"    RMSE:             {rmse_deriv:.6e}")
        print(f"    Max abs error:    {max_err_deriv:.6e}")
        print("-" * 60)
        print(f"Optimization success: {result.success}")
        print("=" * 60)
    
    return coeffs, result


def plot_fitting_results(x_data, y_data, x_deriv, y_deriv, coeffs, save_path=None):
    """
    Plot the fitting results.
    """
    # Generate smooth curves
    x_min = min(x_data.min(), x_deriv.min())
    x_max = max(x_data.max(), x_deriv.max())
    x_smooth = np.linspace(x_min, x_max, 500)
    
    P_smooth, dP_smooth = polynomial_and_derivative(coeffs, x_smooth)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot original data and P(x)
    ax1 = axes[0]
    ax1.scatter(x_data, y_data, s=20, alpha=0.6, label='dataCu (original)', color='blue')
    ax1.plot(x_smooth, P_smooth, 'r-', linewidth=2, label='P(x) fitted')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Original Data Fitting', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot derivative data and P'(x)
    ax2 = axes[1]
    ax2.scatter(x_deriv, y_deriv, s=20, alpha=0.6, label="dataCu2 (derivative)", color='green')
    ax2.plot(x_smooth, dP_smooth, 'r-', linewidth=2, label="P'(x) fitted")
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel("y'", fontsize=12)
    ax2.set_title('Derivative Data Fitting', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        default_path = 'fitting_result.png'
        plt.savefig(default_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {default_path}")
    
    return fig


# ============================================================
# Cu Thermal Expansion Data Fitting
# ============================================================

if __name__ == "__main__":
    import pandas as pd
    
    # Load data from Excel file
    excel_path = r"D:\OneDrive\000-Buffer\科研助理\磁致伸缩\仪器\Thermal expansion data of Cu and Al.xlsx"
    df = pd.read_excel(excel_path, header=None)
    
    # Data structure:
    # Column 0: Temperature (K)
    # Column 1: Cu thermal expansion coefficient (derivative: alpha = (1/L) dL/dT)
    # Column 2: Al thermal expansion coefficient
    # Column 3: Cu relative length change (dL/L * 10^6, original function)
    # Column 4: Al relative length change
    
    # Extract Cu data
    T = df[0].values  # Temperature
    alpha_Cu = df[1].values  # Cu thermal expansion coefficient (derivative data)
    dL_L_Cu = df[3].values  # Cu relative length change (original data)
    
    print("\n" + "=" * 60)
    print("Loading Cu Thermal Expansion Data")
    print("=" * 60)
    print(f"Temperature range: {T.min():.1f} K to {T.max():.1f} K")
    print(f"Number of data points: {len(T)}")
    print(f"dL/L (x10^6) range: {dL_L_Cu.min():.1f} to {dL_L_Cu.max():.1f}")
    print(f"Alpha range: {alpha_Cu.min():.3e} to {alpha_Cu.max():.3e}")
    print()
    
    # Note: The derivative relationship
    # If P(T) = dL/L (relative length change)
    # Then dP/dT should approximate alpha (thermal expansion coefficient)
    # However, alpha = (1/L) * dL/dT, which is slightly different
    # For small changes, dL/L is small, so L ~ L0, and alpha ~ d(dL/L)/dT
    
    # The dL/L data appears to be in units of 10^-6, so we use it as is
    # The alpha data is in units of 1/K
    # To make derivative consistent: d(dL/L * 10^6)/dT = alpha * 10^6
    
    # Scale alpha_Cu to match the units of dL/L derivative
    alpha_Cu_scaled = alpha_Cu * 1e6  # Now in units of 10^-6 / K
    
    print(f"Scaled alpha range: {alpha_Cu_scaled.min():.3f} to {alpha_Cu_scaled.max():.3f} (x10^-6 / K)")
    print()
    
    # Perform joint fitting
    # x_data, y_data: Temperature, dL/L (original function)
    # x_deriv, y_deriv: Temperature, alpha (derivative)
    coeffs, result = joint_polynomial_fit(
        T, dL_L_Cu,           # Original function: P(T) = dL/L
        T, alpha_Cu_scaled,   # Derivative: P'(T) ~ alpha (scaled)
        degree=8,             # 8th degree polynomial, derivative is 7th degree
        weight_data=1.0,
        weight_deriv=1.0,
        verbose=True
    )
    
    # Plot results with proper labels
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Generate smooth curves
    T_smooth = np.linspace(T.min(), T.max(), 500)
    P_smooth, dP_smooth = polynomial_and_derivative(coeffs, T_smooth)
    
    # Plot original data: dL/L vs T
    ax1 = axes[0]
    ax1.scatter(T, dL_L_Cu, s=30, alpha=0.7, label='Cu dL/L (data)', color='blue', zorder=5)
    ax1.plot(T_smooth, P_smooth, 'r-', linewidth=2, label='P(T) fitted', zorder=4)
    ax1.set_xlabel('Temperature (K)', fontsize=12)
    ax1.set_ylabel(r'$\Delta L / L$ ($\times 10^{-6}$)', fontsize=12)
    ax1.set_title('Cu Relative Length Change', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot derivative data: alpha vs T
    ax2 = axes[1]
    ax2.scatter(T, alpha_Cu_scaled, s=30, alpha=0.7, label=r'Cu $\alpha$ (data)', color='green', zorder=5)
    ax2.plot(T_smooth, dP_smooth, 'r-', linewidth=2, label="P'(T) fitted", zorder=4)
    ax2.set_xlabel('Temperature (K)', fontsize=12)
    ax2.set_ylabel(r'Thermal Expansion Coeff. ($\times 10^{-6}$ K$^{-1}$)', fontsize=12)
    ax2.set_title('Cu Thermal Expansion Coefficient', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = 'Cu_thermal_expansion_fit.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    
    # Compute detailed statistics
    P_fit, _ = polynomial_and_derivative(coeffs, T)
    _, dP_fit = polynomial_and_derivative(coeffs, T)
    
    # R-squared
    ss_res_data = np.sum((dL_L_Cu - P_fit)**2)
    ss_tot_data = np.sum((dL_L_Cu - np.mean(dL_L_Cu))**2)
    r2_data = 1 - ss_res_data / ss_tot_data
    
    ss_res_deriv = np.sum((alpha_Cu_scaled - dP_fit)**2)
    ss_tot_deriv = np.sum((alpha_Cu_scaled - np.mean(alpha_Cu_scaled))**2)
    r2_deriv = 1 - ss_res_deriv / ss_tot_deriv
    
    # RMSE
    rmse_data = np.sqrt(np.mean((P_fit - dL_L_Cu)**2))
    rmse_deriv = np.sqrt(np.mean((dP_fit - alpha_Cu_scaled)**2))
    
    # Max error
    max_err_data = np.max(np.abs(P_fit - dL_L_Cu))
    max_err_deriv = np.max(np.abs(dP_fit - alpha_Cu_scaled))
    
    # Create comprehensive output report
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("Cu Thermal Expansion Joint Polynomial Fitting Results")
    output_lines.append("=" * 70)
    output_lines.append("")
    output_lines.append("DATA INFORMATION:")
    output_lines.append(f"  Temperature range: {T.min():.2f} K to {T.max():.2f} K")
    output_lines.append(f"  Number of data points: {len(T)}")
    output_lines.append("")
    output_lines.append("POLYNOMIAL MODEL:")
    output_lines.append("  P(T) = a0 + a1*T + a2*T^2 + a3*T^3 + ... + a8*T^8")
    output_lines.append("  P(T) represents: dL/L (relative length change, in units of 10^-6)")
    output_lines.append("  P'(T) represents: thermal expansion coefficient (in units of 10^-6 / K)")
    output_lines.append("")
    output_lines.append("-" * 70)
    output_lines.append("POLYNOMIAL COEFFICIENTS:")
    output_lines.append("-" * 70)
    for i, c in enumerate(coeffs):
        output_lines.append(f"  a{i} = {c:+.15e}")
    output_lines.append("")
    output_lines.append("-" * 70)
    output_lines.append("FITTING QUALITY METRICS:")
    output_lines.append("-" * 70)
    output_lines.append("")
    output_lines.append("  Original Function P(T) = dL/L:")
    output_lines.append(f"    R-squared (R^2):     {r2_data:.12f}")
    output_lines.append(f"    RMSE:                {rmse_data:.10e}")
    output_lines.append(f"    Max absolute error:  {max_err_data:.10e}")
    output_lines.append("")
    output_lines.append("  Derivative Function P'(T) = alpha:")
    output_lines.append(f"    R-squared (R^2):     {r2_deriv:.12f}")
    output_lines.append(f"    RMSE:                {rmse_deriv:.10e}")
    output_lines.append(f"    Max absolute error:  {max_err_deriv:.10e}")
    output_lines.append("")
    output_lines.append("=" * 70)
    output_lines.append("")
    output_lines.append("POLYNOMIAL EXPRESSION (copy-paste ready):")
    output_lines.append("")
    
    # Generate polynomial expression string
    terms = []
    for i, c in enumerate(coeffs):
        if i == 0:
            terms.append(f"{c:+.10e}")
        elif i == 1:
            terms.append(f"{c:+.10e}*T")
        else:
            terms.append(f"{c:+.10e}*T^{i}")
    poly_expr = " ".join(terms)
    output_lines.append(f"P(T) = {poly_expr}")
    output_lines.append("")
    output_lines.append("=" * 70)
    
    # Print to console
    report = "\n".join(output_lines)
    print("\n" + report)
    
    # Save to file
    result_file = 'Cu_thermal_expansion_fit_results.txt'
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nResults saved to: {result_file}")
    
    # Also save coefficients as numpy array for programmatic use
    coeff_file = 'Cu_fit_coefficients.txt'
    np.savetxt(coeff_file, coeffs, 
               header='Polynomial coefficients [a0, a1, ..., a8] for P(T) = dL/L (x10^-6)\n'
                      'Usage: P(T) = sum(a[i] * T^i) for i = 0 to 8',
               fmt='%.15e')
    print(f"Coefficients saved to: {coeff_file}")
