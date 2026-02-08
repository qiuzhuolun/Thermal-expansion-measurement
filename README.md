# Thermal-expansion-measurement
This repository contains the procedure i concluded and the codes i wrote during experiments of measuring the thermal expansion coefficient.

## Background substraction

The dimensions of the dilatometer cell itself change with temperature, contributing a background signal to the measurement. Therefore, the influence of the dilatometer must be subtracted from the raw measurement results to obtain the intrinsic thermal expansion of the sample. The relationship between the measured length change of the sample ($\Delta L_{sample}^{meas}$), the true length change of the sample ($\Delta L_{sample}$), and the length change contribution from the dilatometer ($\Delta L_{cell}$) is given by:

$$ \Delta L_{sample}^{meas} = \Delta L_{sample} - \Delta L_{cell} $$

The measured length change is a superposition of the real expansion of the sample and the dilatometer background. Since the dilatometer is constructed from high-purity copper, which has a well-known small thermal expansion coefficient, we can calibrate the dilatometer's contribution by measuring a standard copper reference sample. The background contribution is derived as:

$$ \Delta L_{cell} = \Delta L_{Cu}^{ref} - \Delta L_{Cu}^{meas} $$

Substituting this expression into the first equation, we can determine the intrinsic length change of the sample:

$$ \Delta L_{sample} = \Delta L_{sample}^{meas} + \Delta L_{cell} = \Delta L_{sample}^{meas} - \Delta L_{Cu}^{meas} + \Delta L_{Cu}^{ref} $$

Dividing both sides by the initial length $L_0$, we obtain the relative length change:

$$ \frac{\Delta L_{sample}}{L_0} = \frac{\Delta L_{sample}^{meas}}{L_0} - \frac{\Delta L_{Cu}^{meas}}{L_0} + \left(\frac{\Delta L_{Cu}}{L_0}\right)_{ref} $$

The last term, $(\Delta L_{Cu}/L_0)_{ref}$, represents the standard relative thermal expansion of copper, which can be obtained from the literature (e.g., *J. Appl. Phys.* **48**, 853–864, 1977).

**Length Dependent Calibration**

The derivation above assumes that the copper reference sample has the exact same length $L_0$ as the sample under test. However, it is impractical to find a copper reference with the identical length for every sample measurement. According to Küchler *et al.* [1], the background length change of the dilatometer relies linearly on the sample length $L$:

$$ \Delta L_{cell} = A(T) + B(T) L $$

Here, $A(T)$ and $B(T)$ are temperature-dependent coefficients. To determine these coefficients, two copper reference samples with different lengths (e.g., a short one of 0.5 mm and a long one of 2 mm) are measured. Using the measured length changes of these two samples and their theoretical values from the literature, the coefficients $A(T)$ and $B(T)$ can be determined by fitting the data to the equation above.

## References

[1] R. Küchler, C. Stingl, and P. Gegenwart, "A uniaxial stress capacitive dilatometer for high-resolution thermal expansion and magnetostriction under multiextreme conditions," *Rev. Sci. Instrum.* **87**, 073903 (2016).




## Data cleaning
