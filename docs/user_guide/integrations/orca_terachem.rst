ORCA and TeraChem Bridges
=========================

Enerzyme and Enerzymette integrate with external QC programs in two main ways: batch labeling via :code:`annotate` (TeraChem) and on-the-fly optimization via ORCA ExtOpt.

Batch labeling (Enerzyme annotate)
----------------------------------

:code:`enerzyme annotate` with :code:`QMDriver.engine: TeraChem` runs independent single-point or property calculations on each supplier structure. See :doc:`/user_guide/workflows/qm_annotation`.

Requirements:

- Licensed :code:`terachem` executable
- Scratch space and PCM auxiliary files as configured
- SDF supplier with correct formal charges

ORCA ExtOpt + TeraChem (Enerzymette)
------------------------------------

ORCA can optimize geometries while delegating energy/gradient evaluation to another program (`ORCA ExtOpt tutorial <https://www.faccts.de/docs/orca/6.1/tutorials/workflows/extopt.html>`_).

Enerzymette command:

.. code-block:: bash

    enerzymette orca_terachem_request -i orca.extinp.tmp -t terachem_template.inp

Setup checklist
---------------

1. ORCA input with :code:`! ExtOpt` and optimization keywords
2. ORCA :code:`%method` block:

   .. code-block:: text

       ProgExt "/full/path/to/wrapper.sh"
       end

3. Wrapper script (executable):

   .. code-block:: bash

       #!/bin/bash
       enerzymette orca_terachem_request -i "$1" -t /path/to/template.inp

4. TeraChem template :code:`.inp` — basis, functional, solvent, SCF settings
5. Supporting files (PCM radii, etc.) in expected locations
6. :code:`terachem` available in the environment

The bridge reads ORCA's :code:`.extinp.tmp`, runs TeraChem, writes ORCA-compatible :code:`.engrad`. Subsequent cycles reuse MO guesses when configured.

TeraChem timing QC
------------------

.. code-block:: bash

    enerzymette terachem_timing -f terachem.out

Use before merging large QC campaigns into training data to catch unfinished or hung jobs.

Relationship to training
------------------------

- **annotate** → labeled pickle → Datahub
- **ExtOpt bridge** → optimized geometries you may then label or simulate separately

Neither replaces :code:`enerzyme train`; they feed the data pipeline upstream of it.
