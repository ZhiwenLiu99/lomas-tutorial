Lomas's tutorial documentation!
================================

**Lomas** is a Python library for traffic modeling and generation in DCNs.

Lomas contains the following key modules:

- ``lomas.preprocessor``: A package for preprocessing network trace. This package can deal with 
  many input file formats (e.g. CSV, Excel, Pickle, Parquet, Pcap, etc.).  
- ``trafpy.model``: A package for learning the high dimension distribution of input trace. 
  This package can also be used for generating (or recreating) synthetic network traces for downstream applications.


Getting Started
===============
Follow the :doc:`instructions <install>` to install Lomas, then have a look 
at the :doc:`tutorial <tutorial>` and the `examples <https://github.com/ZhiwenLiu99/lomas-tutorial/tree/main/tests>`_ on the
GitHub page.

Free Software
=============
Lomas is free software; you can redistribute it and/or modify it under the terms
of the Apache License 2.0. Contributions are welcome. Check out the `guidelines <https://trafpy.readthedocs.io/en/latest/Contribute.html>`_
on how to contribute.

Documentation
=============

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Install
   Tutorial
   API Docs
   License
   Citing