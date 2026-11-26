{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

{% if module == 'enerzyme.models.efa' %}
{% set public_module = 'enerzyme.models.mp.nonlocal_' %}
{% elif module == 'enerzyme.models.mp.equivariant' and objname in ('GraphSoftmax', 'SoftCap') %}
{% set public_module = 'enerzyme.models.mp.ops' %}
{% endif %}
{% if public_module is defined %}
.. py:class:: {{ objname }}

   See :class:`~{{ public_module }}.{{ objname }}` for the class and its methods.
{% else %}
.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :special-members: __init__
   :undoc-members:
{% endif %}
