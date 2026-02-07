"""
Piecewise Joint Polynomial Fitting with C1 Continuity

Fits Cu thermal expansion data in two segments:
- Segment 1: 0K to 100K
- Segment 2: 100K to 320K

Constraints at breakpoint (100K):
- Function value continuity: P1(100) = P2(100)
- Derivative continuity: P1'(100) = P2'(100)

Each segment uses joint fitting for both original data and derivative data.
"""

import numpy as np
from scipy.optimize import minimize, least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def polynomial_and_derivative(coeffs, x):
    """
    Compute polynomial P(x) and its derivative P'(x).
    coeffs: [a0, a1, a2, ..., an] for P(x) = a0 + a1*x + a2*x^2 + ...
    """
    n = len(coeffs)
    x = np.asarray(x, dtype=float)
    
    P = np.zeros_like(x)
    for i, c in enumerate(coeffs):
        P += c * x**i
    
    dP = np.zeros_like(x)
    for i in range(1, n):
        dP += i * coeffs[i] * x**(i-1)
    
    return P, dP


def piecewise_residual(all_coeffs, 
                       x1, y1, x1_deriv, y1_deriv,
                       x2, y2, x2_deriv, y2_deriv,
                       degree1, degree2, breakpoint,
                       continuity_weight=1000.0):
    """
    Compute residuals for piecewise fitting with continuity constraints.
    
    all_coeffs: concatenated [coeffs1, coeffs2]
    """
    n1 = degree1 + 1
    n2 = degree2 + 1
    
    coeffs1 = all_coeffs[:n1]
    coeffs2 = all_coeffs[n1:n1+n2]
    
    # Segment 1 residuals
    P1, dP1 = polynomial_and_derivative(coeffs1, x1)
    _, dP1_deriv = polynomial_and_derivative(coeffs1, x1_deriv)
    res1_data = P1 - y1
    res1_deriv = dP1_deriv - y1_deriv
    
    # Segment 2 residuals
    P2, dP2 = polynomial_and_derivative(coeffs2, x2)
    _, dP2_deriv = polynomial_and_derivative(coeffs2, x2_deriv)
    res2_data = P2 - y2
    res2_deriv = dP2_deriv - y2_deriv
    
    # Continuity constraints at breakpoint
    P1_bp, dP1_bp = polynomial_and_derivative(coeffs1, np.array([breakpoint]))
    P2_bp, dP2_bp = polynomial_and_derivative(coeffs2, np.array([breakpoint]))
    
    # C0 continuity: P1(bp) = P2(bp)
    res_c0 = continuity_weight * (P1_bp[0] - P2_bp[0])
    
    # C1 continuity: P1'(bp) = P2'(bp)
    res_c1 = continuity_weight * (dP1_bp[0] - dP2_bp[0])
    
    return np.concatenate([res1_data, res1_deriv, res2_data, res2_deriv, [res_c0, res_c1]])


def piecewise_joint_fit(T, dL_L, alpha, breakpoint=100.0, 
                        degree1=8, degree2=8, verbose=True):
    """
    Perform piecewise joint polynomial fitting.
    
    Parameters:
    -----------
    T : array-like
        Temperature values
    dL_L : array-like
        Relative length change (original function)
    alpha : array-like
        Thermal expansion coefficient (derivative)
    breakpoint : float
        Temperature at which to split the data
    degree1, degree2 : int
        Polynomial degrees for segment 1 and 2
    
    Returns:
    --------
    coeffs1, coeffs2 : arrays
        Polynomial coefficients for each segment
    """
    T = np.asarray(T, dtype=float)
    dL_L = np.asarray(dL_L, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    
    # Split data into two segments
    mask1 = T <= breakpoint
    mask2 = T >= breakpoint
    
    T1, y1, y1_deriv = T[mask1], dL_L[mask1], alpha[mask1]
    T2, y2, y2_deriv = T[mask2], dL_L[mask2], alpha[mask2]
    
    if verbose:
        print("=" * 70)
        print("Piecewise Joint Polynomial Fitting")
        print("=" * 70)
        print(f"Breakpoint: {breakpoint} K")
        print(f"Segment 1: {T1.min():.1f} K to {T1.max():.1f} K ({len(T1)} points)")
        print(f"Segment 2: {T2.min():.1f} K to {T2.max():.1f} K ({len(T2)} points)")
        print(f"Polynomial degrees: {degree1} (segment 1), {degree2} (segment 2)")
        print()
    
    # Initial guess: fit each segment independently first
    init_coeffs1 = np.polyfit(T1, y1, degree1)[::-1]
    init_coeffs2 = np.polyfit(T2, y2, degree2)[::-1]
    initial_guess = np.concatenate([init_coeffs1, init_coeffs2])
    
    # Optimize with continuity constraints
    result = least_squares(
        piecewise_residual,
        initial_guess,
        args=(T1, y1, T1, y1_deriv,
              T2, y2, T2, y2_deriv,
              degree1, degree2, breakpoint, 1000.0),
        method='lm',
        verbose=0
    )
    
    n1 = degree1 + 1
    coeffs1 = result.x[:n1]
    coeffs2 = result.x[n1:n1+degree2+1]
    
    if verbose:
        # Verify continuity at breakpoint
        P1_bp, dP1_bp = polynomial_and_derivative(coeffs1, np.array([breakpoint]))
        P2_bp, dP2_bp = polynomial_and_derivative(coeffs2, np.array([breakpoint]))
        
        print("-" * 70)
        print("Continuity Check at Breakpoint:")
        print("-" * 70)
        print(f"  P1({breakpoint}) = {P1_bp[0]:.10f}")
        print(f"  P2({breakpoint}) = {P2_bp[0]:.10f}")
        print(f"  Difference: {abs(P1_bp[0] - P2_bp[0]):.2e}")
        print()
        print(f"  P1'({breakpoint}) = {dP1_bp[0]:.10f}")
        print(f"  P2'({breakpoint}) = {dP2_bp[0]:.10f}")
        print(f"  Difference: {abs(dP1_bp[0] - dP2_bp[0]):.2e}")
        print()
        
        # Compute fitting metrics for each segment
        print("-" * 70)
        print("Fitting Quality - Segment 1 (0 to 100 K):")
        print("-" * 70)
        P1_fit, dP1_fit = polynomial_and_derivative(coeffs1, T1)
        
        ss_res1 = np.sum((y1 - P1_fit)**2)
        ss_tot1 = np.sum((y1 - np.mean(y1))**2)
        r2_1 = 1 - ss_res1 / ss_tot1
        
        ss_res1_d = np.sum((y1_deriv - dP1_fit)**2)
        ss_tot1_d = np.sum((y1_deriv - np.mean(y1_deriv))**2)
        r2_1_d = 1 - ss_res1_d / ss_tot1_d
        
        print(f"  Original data R^2:    {r2_1:.12f}")
        print(f"  Derivative data R^2:  {r2_1_d:.12f}")
        print()
        
        print("-" * 70)
        print("Fitting Quality - Segment 2 (100 to 320 K):")
        print("-" * 70)
        P2_fit, dP2_fit = polynomial_and_derivative(coeffs2, T2)
        
        ss_res2 = np.sum((y2 - P2_fit)**2)
        ss_tot2 = np.sum((y2 - np.mean(y2))**2)
        r2_2 = 1 - ss_res2 / ss_tot2
        
        ss_res2_d = np.sum((y2_deriv - dP2_fit)**2)
        ss_tot2_d = np.sum((y2_deriv - np.mean(y2_deriv))**2)
        r2_2_d = 1 - ss_res2_d / ss_tot2_d
        
        print(f"  Original data R^2:    {r2_2:.12f}")
        print(f"  Derivative data R^2:  {r2_2_d:.12f}")
        print()
        
        print("-" * 70)
        print("Segment 1 Coefficients (a0 + a1*T + a2*T^2 + ...):")
        print("-" * 70)
        for i, c in enumerate(coeffs1):
            print(f"  a{i} = {c:+.15e}")
        print()
        
        print("-" * 70)
        print("Segment 2 Coefficients (a0 + a1*T + a2*T^2 + ...):")
        print("-" * 70)
        for i, c in enumerate(coeffs2):
            print(f"  a{i} = {c:+.15e}")
        print()
        
        print("=" * 70)
    
    return coeffs1, coeffs2, result


def plot_piecewise_results(T, dL_L, alpha, coeffs1, coeffs2, breakpoint, save_path=None):
    """Plot piecewise fitting results."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Generate smooth curves for each segment
    T1_smooth = np.linspace(T.min(), breakpoint, 300)
    T2_smooth = np.linspace(breakpoint, T.max(), 300)
    
    P1_smooth, dP1_smooth = polynomial_and_derivative(coeffs1, T1_smooth)
    P2_smooth, dP2_smooth = polynomial_and_derivative(coeffs2, T2_smooth)
    
    # Plot original data: dL/L vs T
    ax1 = axes[0]
    ax1.scatter(T, dL_L, s=30, alpha=0.7, label='Cu dL/L (data)', color='blue', zorder=5)
    ax1.plot(T1_smooth, P1_smooth, 'r-', linewidth=2, label='Segment 1 fit', zorder=4)
    ax1.plot(T2_smooth, P2_smooth, 'g-', linewidth=2, label='Segment 2 fit', zorder=4)
    ax1.axvline(x=breakpoint, color='gray', linestyle='--', alpha=0.5, label=f'Breakpoint ({breakpoint} K)')
    ax1.set_xlabel('Temperature (K)', fontsize=12)
    ax1.set_ylabel(r'$\Delta L / L$ ($\times 10^{-6}$)', fontsize=12)
    ax1.set_title('Cu Relative Length Change (Piecewise Fit)', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot derivative data: alpha vs T
    ax2 = axes[1]
    ax2.scatter(T, alpha, s=30, alpha=0.7, label=r'Cu $\alpha$ (data)', color='blue', zorder=5)
    ax2.plot(T1_smooth, dP1_smooth, 'r-', linewidth=2, label='Segment 1 fit', zorder=4)
    ax2.plot(T2_smooth, dP2_smooth, 'g-', linewidth=2, label='Segment 2 fit', zorder=4)
    ax2.axvline(x=breakpoint, color='gray', linestyle='--', alpha=0.5, label=f'Breakpoint ({breakpoint} K)')
    ax2.set_xlabel('Temperature (K)', fontsize=12)
    ax2.set_ylabel(r'Thermal Expansion Coeff. ($\times 10^{-6}$ K$^{-1}$)', fontsize=12)
    ax2.set_title('Cu Thermal Expansion Coefficient (Piecewise Fit)', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


# ============================================================
# Main: Cu Thermal Expansion Piecewise Fitting
# ============================================================

if __name__ == "__main__":
    # Load data
    excel_path = r"D:\OneDrive\000-Buffer\科研助理\磁致伸缩\仪器\Thermal expansion data of Cu and Al.xlsx"
    df = pd.read_excel(excel_path, header=None)
    
    T = df[0].values
    alpha_Cu = df[1].values * 1e6  # Scale to 10^-6 / K
    dL_L_Cu = df[3].values  # Already in 10^-6 units
    
    print("\n" + "=" * 70)
    print("Loading Cu Thermal Expansion Data")
    print("=" * 70)
    print(f"Temperature range: {T.min():.1f} K to {T.max():.1f} K")
    print(f"Number of data points: {len(T)}")
    print()
    
    # Perform piecewise fitting
    breakpoint = 100.0
    coeffs1, coeffs2, result = piecewise_joint_fit(
        T, dL_L_Cu, alpha_Cu,
        breakpoint=breakpoint,
        degree1=8,  # Higher degree for the more complex low-T region
        degree2=8,  # Standard degree for high-T region
        verbose=True
    )
    
    # Plot results
    plot_piecewise_results(
        T, dL_L_Cu, alpha_Cu, 
        coeffs1, coeffs2, breakpoint,
        save_path='Cu_thermal_expansion_piecewise_fit.png'
    )
    
    # Compute fitting metrics for output
    mask1 = T <= breakpoint
    mask2 = T >= breakpoint
    T1, y1, y1_deriv = T[mask1], dL_L_Cu[mask1], alpha_Cu[mask1]
    T2, y2, y2_deriv = T[mask2], dL_L_Cu[mask2], alpha_Cu[mask2]
    
    P1_fit, dP1_fit = polynomial_and_derivative(coeffs1, T1)
    P2_fit, dP2_fit = polynomial_and_derivative(coeffs2, T2)
    
    # Segment 1 metrics
    ss_res1 = np.sum((y1 - P1_fit)**2)
    ss_tot1 = np.sum((y1 - np.mean(y1))**2)
    r2_1 = 1 - ss_res1 / ss_tot1
    rmse1 = np.sqrt(np.mean((P1_fit - y1)**2))
    max_err1 = np.max(np.abs(P1_fit - y1))
    
    ss_res1_d = np.sum((y1_deriv - dP1_fit)**2)
    ss_tot1_d = np.sum((y1_deriv - np.mean(y1_deriv))**2)
    r2_1_d = 1 - ss_res1_d / ss_tot1_d
    rmse1_d = np.sqrt(np.mean((dP1_fit - y1_deriv)**2))
    max_err1_d = np.max(np.abs(dP1_fit - y1_deriv))
    
    # Segment 2 metrics
    ss_res2 = np.sum((y2 - P2_fit)**2)
    ss_tot2 = np.sum((y2 - np.mean(y2))**2)
    r2_2 = 1 - ss_res2 / ss_tot2
    rmse2 = np.sqrt(np.mean((P2_fit - y2)**2))
    max_err2 = np.max(np.abs(P2_fit - y2))
    
    ss_res2_d = np.sum((y2_deriv - dP2_fit)**2)
    ss_tot2_d = np.sum((y2_deriv - np.mean(y2_deriv))**2)
    r2_2_d = 1 - ss_res2_d / ss_tot2_d
    rmse2_d = np.sqrt(np.mean((dP2_fit - y2_deriv)**2))
    max_err2_d = np.max(np.abs(dP2_fit - y2_deriv))
    
    # Continuity check
    P1_bp, dP1_bp = polynomial_and_derivative(coeffs1, np.array([breakpoint]))
    P2_bp, dP2_bp = polynomial_and_derivative(coeffs2, np.array([breakpoint]))
    
    # Save comprehensive results
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("Cu Thermal Expansion PIECEWISE Polynomial Fitting Results")
    output_lines.append("=" * 70)
    output_lines.append("")
    output_lines.append("DATA INFORMATION:")
    output_lines.append(f"  Temperature range: {T.min():.2f} K to {T.max():.2f} K")
    output_lines.append(f"  Total data points: {len(T)}")
    output_lines.append(f"  Breakpoint: {breakpoint} K")
    output_lines.append(f"  Segment 1 points: {len(T1)}")
    output_lines.append(f"  Segment 2 points: {len(T2)}")
    output_lines.append("")
    output_lines.append("-" * 70)
    output_lines.append("CONTINUITY CHECK AT BREAKPOINT:")
    output_lines.append("-" * 70)
    output_lines.append(f"  P1({breakpoint}) = {P1_bp[0]:.12f}")
    output_lines.append(f"  P2({breakpoint}) = {P2_bp[0]:.12f}")
    output_lines.append(f"  Function value difference: {abs(P1_bp[0] - P2_bp[0]):.2e}")
    output_lines.append("")
    output_lines.append(f"  P1'({breakpoint}) = {dP1_bp[0]:.12f}")
    output_lines.append(f"  P2'({breakpoint}) = {dP2_bp[0]:.12f}")
    output_lines.append(f"  Derivative difference: {abs(dP1_bp[0] - dP2_bp[0]):.2e}")
    output_lines.append("")
    output_lines.append("-" * 70)
    output_lines.append("FITTING QUALITY METRICS:")
    output_lines.append("-" * 70)
    output_lines.append("")
    output_lines.append("SEGMENT 1 (0 K to 100 K):")
    output_lines.append("  Original function P(T) = dL/L:")
    output_lines.append(f"    R-squared (R^2):     {r2_1:.12f}")
    output_lines.append(f"    RMSE:                {rmse1:.10e}")
    output_lines.append(f"    Max absolute error:  {max_err1:.10e}")
    output_lines.append("")
    output_lines.append("  Derivative function P'(T) = alpha:")
    output_lines.append(f"    R-squared (R^2):     {r2_1_d:.12f}")
    output_lines.append(f"    RMSE:                {rmse1_d:.10e}")
    output_lines.append(f"    Max absolute error:  {max_err1_d:.10e}")
    output_lines.append("")
    output_lines.append("SEGMENT 2 (100 K to 320 K):")
    output_lines.append("  Original function P(T) = dL/L:")
    output_lines.append(f"    R-squared (R^2):     {r2_2:.12f}")
    output_lines.append(f"    RMSE:                {rmse2:.10e}")
    output_lines.append(f"    Max absolute error:  {max_err2:.10e}")
    output_lines.append("")
    output_lines.append("  Derivative function P'(T) = alpha:")
    output_lines.append(f"    R-squared (R^2):     {r2_2_d:.12f}")
    output_lines.append(f"    RMSE:                {rmse2_d:.10e}")
    output_lines.append(f"    Max absolute error:  {max_err2_d:.10e}")
    output_lines.append("")
    output_lines.append("-" * 70)
    output_lines.append("POLYNOMIAL COEFFICIENTS:")
    output_lines.append("-" * 70)
    output_lines.append("")
    output_lines.append("SEGMENT 1 (0 K to 100 K):")
    output_lines.append("  P1(T) = a0 + a1*T + a2*T^2 + ... + a8*T^8")
    for i, c in enumerate(coeffs1):
        output_lines.append(f"  a{i} = {c:+.15e}")
    output_lines.append("")
    output_lines.append("SEGMENT 2 (100 K to 320 K):")
    output_lines.append("  P2(T) = a0 + a1*T + a2*T^2 + ... + a8*T^8")
    for i, c in enumerate(coeffs2):
        output_lines.append(f"  a{i} = {c:+.15e}")
    output_lines.append("")
    output_lines.append("=" * 70)
    
    report = "\n".join(output_lines)
    
    result_file = 'Cu_thermal_expansion_piecewise_results.txt'
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nResults saved to: {result_file}")
    
    # Save coefficients for programmatic use
    np.savez('Cu_piecewise_coefficients.npz', 
             coeffs1=coeffs1, coeffs2=coeffs2, 
             breakpoint=breakpoint)
    print("Coefficients saved to: Cu_piecewise_coefficients.npz")
