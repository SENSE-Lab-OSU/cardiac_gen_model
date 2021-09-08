from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

reqs = []

# Get the long description from the README file
long_description= ("A hierarchical generative model for cardiological signals"
                    "(PPG,ECG etc.) that keeps the physiological"
                    " characteristics intact.")

if __name__ == '__main__':
    setup(
        name="CardioGen",

        version='1.0.0',

        package_data={'': ['default.yml']},

        description=("A hierarchical generative model for cardiological signals"
                    "(PPG,ECG etc.) that keeps the physiological"
                    " characteristics intact."),
        long_description_content_type='text/markdown',
        long_description=long_description,

        author='https://sense-lab-osu.github.io/',
        author_email='agarwal.270@buckeyemail.osu.edu',

        license='BSD3',
        url = 'https://github.com/SENSE-Lab-OSU/cardio_gen_model/blob/master/LICENSE',


        classifiers=[

            'Development Status :: 5 - Production/Stable',

            'Intended Audience :: Healthcare Industry',
            'Intended Audience :: Science/Research',

            'License :: OSI Approved :: BSD License',

            'Natural Language :: English',

            'Programming Language :: Python :: 3',

            'Topic :: Scientific/Engineering :: Information Analysis',
            'Topic :: System :: Distributed Computing'
        ],

        keywords='mResearch machine-learning deep-learning GAN generative-model',

        # You can just specify the packages manually here if your project is
        # simple. Or you can use find_packages().
        packages=find_packages(exclude=['contrib', 'docs', 'tests','Examples']),

        # List run-time dependencies here.  These will be installed by pip when
        # your project is installed. For an analysis of "install_requires" vs pip's
        # requirements files see:
        # https://packaging.python.org/en/latest/requirements.html
        install_requires=reqs,

    )