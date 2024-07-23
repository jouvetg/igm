### <h1 align="center" id="title">IGM module `vert_flow` </h1>

# Description:

This IGM module computes the vertical component (providing `state.W`) of the velocity from the horizontal components (state.U, computed from an emulation of the Blatter-Pattyn model in the module `iceflow`) by integrating the imcompressibility condition. This module is typically needed prior calling module `particle` for 3D particle trajectory integration, or module `enthalpy` for computing 3D advection-diffusion of the enthalpy.

There are currently two implementations (choose with parameter `vflo_method`). `vflo_method` is by default set to `'kinematic'`.
Here is a quick explanation of those two formulas. For detailled proof, see Claire-Mathilde Stucki's internship report. Those two methods are now implemented with vectorized tensorflow functions to make the code as efficient as possible.

## Notations and hypothesis

We consider a particule within a glacier. 
The glacier is defined by a bedrock $b(x)$ and a surface $s(x,t)$. 
Therefore the thickness of the ice is $h(x,t) = s(x,t) - b(x)$. 
The particle is described by its Cartesion coordinates $(x(t), z(t))$, and velocity vector $\overrightarrow{u} = u \overrightarrow{e_x} + w \overrightarrow{e_z}$, where $u = \frac{dx}{dt}$ and $w= \frac{dz}{dt}$.

We assume that the ice is incompressible, ie 
$
	\overrightarrow{\nabla} . \overrightarrow{u} = 0,
$

We define the relative height of the particle inside the glacier as $r = \frac{z -b}{h}$.

We assume an impermeability boundary condtion on the bedrock, which implies that the velocity along the bedrock is tangent to it. This gives us following relation :
$
	\frac{\partial b}{\partial x} = \frac{w(x,b(x))}{u(x,b(x))},
$

## Formulas

- `'incompressibility'` is a direct integration of the incompressibility condition $\overrightarrow \nabla .\overrightarrow u = 0$ :
$$ w(z) =  u(x,b(x)) \frac{\partial b}{\partial x} - \int_b^z \left( \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} \right) dz. $$

- `'kinematic'`is a rewriting of the `'incompressibility'`formula. It relies on exactly the same physical hypothesis but uses Leibniz formula to switch integrals and derivatives.
$$ w(z) = \frac{\partial z}{\partial x}u + \frac{\partial z}{\partial y}v - \frac{\partial}{\partial x}\left( \int_b^z u dz \right) - \frac{\partial}{\partial y}\left( \int_b^z v dz \right)$$
where $z(x,y)$ is considered as the surface of constant relative height $r= \frac{z-b}{s-b}$, defined for all value of $r$ between 0 and 1.

## Technical detail about computing derivatives

The discretisation used in IGM involves a regular grid in $x$ and $y$ directions, which is very convenient to perform numerical derivations. Concerning the vertical direction $z$, it is a bit more complex : instead of a regular discretization along the $z$ axis, the third dimension is discretized in terms of relative height within the glacier $r = \frac{z-b}{h}$. There are several layers (by default 10) and each layer has a constant relative height within the glacier, which implies a non constant altitude. This explains why the terms $\frac{\partial z}{\partial x}$ and $\frac{\partial z}{\partial y}$ appear in the `'kinematic'` formula when using Leibniz rule. 

It also has consequences when using the `'incompressibility'` method when computing derivatives. The previous formula is written in the orthonormal basis $(x_0,y_0,z_0)$ but because of the non flat shape of the layers, all quantities in IGM are expressed in the local basis $(x,y,z)$ that is not orthonormal : $z$ is the same vertical axis as $z_0$ but $x$ and $y$ are not horizontal, they follow the local slope of the layer. It is therefore necessary to take in account a change of basis, using following matrix (small angles approximation, because the slopes are not steep) :

$$ \begin{pmatrix}
1 & - \frac{\partial z}{\partial x} \\
0 & 1
\end{pmatrix} $$

Finally, to compute derivatives along $x_0$ axis, we have to use following formula (or equivalent formula to derivate along $y_0$) :
$$\frac{\partial u}{\partial x_0} = \frac{\partial u}{ \partial x} - \frac{\partial z}{\partial x} \frac{\partial u}{\partial z}$$

Code originally written by G. Jouvet, improved and tested by Claire-mathilde Stucki
