### <h1 align="center" id="title">IGM enthalpy module  </h1>

**Warning: this rather complex module was not much tested so far, use it with care!**

# Description:

This IGM module models the ice enthalpy, which permits to jointly model the ice temperature, as well as the water content created when the temperature hits the pressure melting points, and therefore energy conservation, which is not the case when modelling the sole temperature variable. The model is described in [(Aschwanden and al, JOG, 2012)](https://www.cambridge.org/core/journals/journal-of-glaciology/article/an-enthalpy-formulation-for-glaciers-and-ice-sheets/605D2EC3DE03B82F2A8289220E76EB27). Here, we used a simplified version largely inspired from the one implemented in [PISM](https://www.pism.io/). Other references that have helped are [(Kleiner and al, TC, 2015)](https://tc.copernicus.org/articles/9/217/2015/) and [(Wang and al, 2020)](https://www.sciencedirect.com/science/article/abs/pii/S0098300419311458).

# Model:

## Ice flow

Here we only quickly sketch the components of the ice dynamical modelling necessary for modeling the Enthalpy.

Ice is assumed to be a Non Newtonian fluid, gouverned by Glen's flow law:
$$ \tau = A^{-1/n} | D({\bf u}) |^{1/n-1} D({\bf u}), $$

where $D({\bf U})$ and $\tau$ are the strain rate and deviatoric stress tensor, respectively. While a no-stress condition is applied on the top surface, we assume that
the basal shear stress  $\tau_b$ and the sliding velocity $u_b$ are linked by Weertmann power law at the glacier bed interface:
$$ 
 \tau_b = - c | u_b |^{m-1} u_b,
$$
where $c=\tau_c u_{th}^{-m}$ (unit: $Mpa \; m^{-m} \; y^m$), $\tau_c$ being the yield stress and $u_{th}$ being a parameter homegenous to ice velocity. This formalism is taken from PISM.

(Note that originally, IGM had $c^{-m}$ instead of $c$ above as "slidingco" with $u_{th}=1$. The newest runs under parameters "new_sliding_param", and $c$ has unit $MPa \; m^{-m} \; y^m$)

## Enthalpy

The Enthalpy $E$ is a 3D variable defined everywhere in the ice as a function of the temperature $T$ and the water content $\omega$:

$$
\begin{align}
E(T, \omega,p) = 
\left\{
\begin{array}{ll}
c_i (T- T_{\rm ref}), & {\rm  if } \; T < T_{\rm pmp} , \\ 
E_{\rm pmp} + L \omega, &  
{\rm if } \;  T = T_{\rm pmp} \; {\rm and } \; 0 \le \omega,
\end{array} 
\right.  
\end{align}
$$

where $c_i$ is the heat capacity, $T_{ref}$ is the reference temperature, $T_{\rm pmp} = T_{\rm pmp}(p) = T_0 - \beta p$ is the temperature pressure-melting point of ice, $E_{\rm pmp}(p)$  is the enthalpy pressure-melting point of ice defined as 

$$ E_{\rm pmp}(p) = c_i (T_{\rm pmp}(p) - T_{\rm ref}), $$

where $L$ is the latent heat of fusion. According to the above defintion of enthalpy, we have two possible modes: i) the ice is cold, i.e. below the melting point, and the Enthalpy is simply proportional to the temperature minus a reference temperature ii) the ice is temperate and the Enthalpy continue to grow, the additional component $L \omega$ corresponding to the creation of water content by energy transfer. Therefore, it is straightforward to deduce $E$ from $T$ and $\omega$.

The enthalpy model consists of the following advection-diffusion equation (the horizontal diffusion being neglected):

$$ 
\begin{align}
& \rho_i \left( \frac{\partial E}{ \partial t}
+ u_x \frac{\partial E}{ \partial x}
+ u_y \frac{\partial E}{ \partial y} 
+ u_z \frac{\partial E}{ \partial z} \right) 
 - \frac{\partial }{\partial z} \left(
K_{c,t} \frac{\partial E}{ \partial z} \right)
= \phi - \rho_w L D_w(\omega),
\end{align}
$$

where $\rho_i$ is the ice density, $K_{c,t}$ equals $K_c = k_i/c_i$ if the ice is cold ($E<E_{pmp}$) or $K_t = \epsilon k_i/c_i$ otherwise, $\phi$ is the strain heating defined by (using Glen's flow law)

$$ \phi = D({\bf U}) \tau = A^{-1/n} | D({\bf u}) |^{1+1/n}, $$

where $D({\bf U})$ and $\tau$ are the strain rate and deviatoric stress tensor, respectively. The last source term $- \rho_w L D_w(\omega)$  permits to remove the water in temperate ice $D_w(\omega)$ being a drainage function ((Greve, 1997) and (Aschwanden et al., 2012)).

At the top ice surface, the enthalpy equation is constrained by the surface temperature (or equivalently the Enthalpy) provided by the climate forcing (Dirichlet condition). At the glacier bed, boundary conditions for the enthalpy equation are multiple ([(Aschwanden and al, JOG, 2012)](https://www.cambridge.org/core/journals/journal-of-glaciology/article/an-enthalpy-formulation-for-glaciers-and-ice-sheets/605D2EC3DE03B82F2A8289220E76EB27), [(Kleiner and al, TC, 2015)](https://tc.copernicus.org/articles/9/217/2015/) and [(Wang and al, 2020)](https://www.sciencedirect.com/science/article/abs/pii/S0098300419311458).)

- $K_{c} \frac{\partial E}{ \partial z} = Q_{\rm geo} + Q_{\rm fh}$
if $E_b<E_{\rm pmp}$ and $H_w = 0$, (cold base, dry) 
- $ E_b = E_{\rm pmp} $ if $E_b<E_{\rm pmp}$ and $H_w > 0$, (cold base, wet)
- $ E_b = E_{\rm pmp} $ if $E_b \ge E_{\rm pmp}$ and $W_{till}> 0$, zero temperate basal layer, (temperate base, cold ice)
- $ K_{t} \frac{\partial E}{ \partial z} = 0$ if $E_b \ge E_{\rm pmp}$ and $W_{till} > 0$, non-zero temperate basal layer, (temperate base, temp. ice)

where $H_t$ is the height of the temperate basal layer, $Q_{\rm geo}$ and $Q_{\rm fh}$ are the geothermal heat flux, and the frictional heat flux, respectively. The latter is defined by 
$$ 
Q_{\rm fh} = \tau_b \cdot u_b = c | u_b |^{m+1}.
$$

As a matter of fact, the ice enthalpy (or equivalently temperature and water content) feedbacks the dynamical model in two ways. The Enthalpy directly impacts the sliding basal parametrization, while variations in temperature and water content cause ice softening or hardening. 
  
## Basal melt  

When the temperature hits the pressure-melting point at the glacier bed (i.e. $E \ge E_{\rm pmp}$), the basal melt rate is calculated via the following equation:
$$
\begin{equation}
m_b = \frac{Q_{fr}+Q_{geo} - K_{t,c} \frac{\partial E}{ \partial z} }{\rho_i L}. 
\end{equation}
$$
The basal melt rate is further adjusted positively to account for the drainage of the water content generated along the entire column.

## Water thickness

The basal water thickness in the till $W_{till}$ is computed directly from the basal melt rate as follows:
$$
\begin{equation}
\frac{\partial W_{till} }{ \partial z} = \frac{m_b}{\rho_w} - C,
\end{equation}
$$
where $C$ is a simple drainage parameter. The till is assumed to be saturated when it reaches the value $W_{till}^{max} = 2$ m, therefore, the till water thickness is bounded to this value. The effective thickness of water within the till $N_{till}$ is computed from the saturation ratio $s= W_{till} / W_{till}^{max}$ by the formula [(Bueler and Pelt, GMD, 2015)](https://gmd.copernicus.org/articles/8/1613/2015/gmd-8-1613-2015.html):
$$
\begin{equation}
N_{till} = \min \left\{ P_0, N_0 \left( \frac{\delta P_0}{N_0} \right)^s 10^{(e_0/C_c)(1-s)} \right\},
\end{equation}
$$
where $P_0$ is the ice overburden pressure and the remaining parameters are constant. 

## Sliding parametrization

Last, the sliding coefficient $c$ is defined with the Mohr-Coulomb (Cuffey and Paterson, 2010) sliding law with the effective pressure in the till:
$$
\begin{align}
c  = \tau_c u_{th}^{-m} & = N_{till} \tan(\phi) u_{th}^{-m}, \\
\end{align}
$$
where $\phi$ is the till friction angle.

## Ahrrenius factor

We use the Glen-Paterson-Budd-Lliboutry-Duval law, where

$$A(T,\omega)= A_c(T)(1+C \omega) $$

where $A_c(T)$ is given by the Paterson-Budd law:

$$  A_c(T)= A \exp{( −Q / (R \, T_{pa}) )} $$

where $A$ and $Q$ have different values below and above a threshold temperature. 

$$ A = 3.985 \times 10^{-13} \, s^{-1} Pa^{-3}, \textrm{ if } T <263.15 K$$
$$ A = 1.916 \times 10^3 \, s^{-1} Pa^{-3}, \textrm{else.}$$
and
$$ Q =  60 kJ mol^{-1},  \textrm{ if } T <263.15 K$$
$$ Q = 139 kJ mol^{-1},  \textrm{else.}$$

These values are taken from (Paterson 1994).


## Pressure-adujsuted temperature

Melting point temperature at pressure is adjusted for pressure as follows
$$ T_{pmp} = T_{0} - \beta \rho g d, $$
where $d$ is the depth, $T_{0}=273.15$ is the melting temperate at standart pressure (unit [$K$]),  $\beta = 7.9 \; 10^{-8}$ is Clausius-Clapeyron constant (unit [$K Pa^{-1}$]). Therefore, one define the "pressure-adjusted" temperaure $T_{pa}$ as being the temperature with a shift such that its metling point temperature reference is always zero:
$$ T_{pa} = T + \beta \rho g z. $$


# Numerical scheme

To solve the Enthalpy equation numerically, one makes use of the same horizontal and vertical discretization as used for the ice flow.
Treating the horizontal advection term explicitly using an upwind scheme, the Enthalpy equation with its boundary conditions can be solved column-wise as a one-dimensional advection-diffusion equation. This is achieved implicitly for both the vertical advection and the diffusion term, which are approximated by finite differences. For each column, 
one solves a small tridiagonal using the Tridiagonal Matrix Algorithm (TDMA) aka Thomas Algorithm.

Updating the Enthalpy at time $t^{n+1}$ requires to perform several sub-steps (in function ''update_enthalpy(params,state)''):

- compute the mean surface temperature $T^n_s$ to enforce upper surface Dirichlet Boundary condition,
- compute the vertical discretization with respect to the ice geometry $h^n$,
- compute the temperature $T_{pmp}$ and enthalpy $E_{pmp}$ at pressure meltinf point,
- compute the ice temperature field $T^n$ from the Enthalpy $E^n$,
- compute the Arrhenius factor $A(T^n)$ from temperature $T^n$,
- compute the 3D strain heat $\phi^n$ from ice flow field ${\bf u}^{n+1}$ and rrhenius factor $A(T^n)$,
- compute the 2D basal frictional heat $Q_{\rm fh}^n$, from basal velocity field ${\bf u}$ and sliding coefficient $c^n$,
- compute the $UPWIND^n$ term for the explicit treatement of the horizontal advection,
- compute the surface Enthalpy $E^n_s$ from the surface temperature $T^n_s$,
- compute the new enthalpy $E^{n+1}$ field solving one-dimension column-wise advection-diffusion equation, as well as the basal melt rate, **this is the main updating step**,
- compute the water thickness in the till $W^{n+1}$,
- compute the sliding parametrization $c^{n+1}$. 

# Numerical stability -- time stepping

Here one updates the enthalpy as many times as the ice flow, we assume that the time step for the explicit advection is more restrictive than the implicit diffusion-advection problem.

# Dependencies

- the enthalpy module builds upon the module iceflow
- one needs to have the vertical_iceflow module activated to provide the vertical velocitiy
- make sure to have params.dim_arrhenius = 3
- make sure to have params.new_friction_param = true
- make sure to ave enough retraining retrain_iceflow_emulator_freq: 1, possibly retrain_iceflow_emulator_nbit more than 1.
