# coolest_algorithm.py

## Run (local)
***************
1. `workon coolest_algorithm`

For testing a single image:

`python coolest_algorithm.py -i "test_image.jpg"`

This will print a result in the format:

`[<int:squares>, <int:circles>, <int:triangles>]`

For testing a folder of images and corresponding labels:

`python coolest_algorithm.py -t directory_of_tests`

This will print a result in the format:

`[<float:shape-wise-accuracy>, <float:image-wise-accuracy>]`

## Set-up
**********
1. Create environment
    1. `pip install virtualenv`
    1. `pip install virtualenvwrapper-win`
    1. `mkvirtualenv coolest_algorithm`
    1. `pip install -r requirements.txt` - make sure this step is done after the virtual environment is set up
