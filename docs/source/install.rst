Install
=======

Open Git Bash. Change the current working directory to the location where you want
to clone this `GitHub <https://github.com/ZhiwenLiu99/lomas-tutorial>`_ project, and run::

    $ git clone https://github.com/ZhiwenLiu99/lomas-tutorial.git

In the project's root directory, run::

    $ python setup.py install

Then, still in the root directory, install the required packages with either pip::

    $ pip install -r requirements.txt

or conda::

    $ conda install --file requirements.txt


You should then be able to import Lomas into your Python script from any directory
on your machine::

    >>> import lomas
