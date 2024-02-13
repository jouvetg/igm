# will run all tests in the tests folder and ignore depreciated warnings (numpy).
# This is across all tests - ideally, use a decorator to ignore warnings on specific functions
# -p is to not create pycached from pytest (important for pypi as deployment will mess up the package)
# ! Supposed to be newer -B but confirm!
python -m pytest -W ignore::DeprecationWarning -W ignore::RuntimeWarning -p no:cacheprovider