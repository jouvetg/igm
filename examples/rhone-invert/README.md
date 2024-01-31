
# Overview


This set-up permits to test the optimization case on any glacier given the RGI ID in the most general case (all data used, all control), this on 31 Jannuary 2024 (GJ) the most updated & complete and achived version.

The main notable change compared to former example are (aletsch-invert):

- number of tolerance parameters have been retuned.
- scaling was introducing to optimize each controls with different weights (there are parameters for that, e.g. opti_scaling_thk, opti_scaling_usurf, opti_scaling_slidingco, should not be ncessary to touch them)
- slidingco is added as a control (with strong regalurization), now working! (thanks to the scaling, pont above)
- Millan velocities outlier are now filtered in the oggm_shop
- the cost for the match for ice thickness profiles is now weighted to give spatially uniform weight (typically weigting more isolated profiles)
- Important, the convexity parameter was significantly increasing, the idea behind is that in absence of any data, an ice thickness can be inferred from the RGI mask by solving poission problem -- the convexity parameter being the load in the poisson problem, this weight controls then the volume. Here I imprlmented an heuristic rule to foolow approx the typical scaling btween area and volume found in the litterature.
- a new parameter opti_step_size_decay permits to implement an gently decay in the step size of the Adam optimizer

- a new method was introduced to force smooth flux divergence ("divfluxpen") but did not proved to be relevant, not active yet.