# Lebedev quadrature rules

`lebedev_rules.npz` is vendored from [deepmd-kit](https://github.com/deepmodeling/deepmd-kit)
(`deepmd/dpmodel/utils/lebedev_rules.npz`, LGPL-3.0-or-later).

The underlying sphere Lebedev rules are from John Burkardt:
https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/sphere_lebedev_rule.html

Keys: `points_PPP`, `weights_PPP` (weights sum to 1; integral = 4π Σ w f).
