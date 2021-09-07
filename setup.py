from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

reqs = ['absl-py==0.13.0\n', 'aiohttp==3.7.4\n', 'alabaster==0.7.12\n',
        'anyio==2.2.0\n', 'appdirs==1.4.4\n', 'argcomplete==1.12.3\n', 
        'argh==0.26.2\n', 'argon2-cffi==20.1.0\n', 'arrow==0.13.1\n', 
        'astor==0.8.1\n', 'astroid==2.6.6\n', 'astropy==4.3.1\n', 
        'async-generator==1.10\n', 'async-timeout==3.0.1\n', 
        'atomicwrites==1.4.0\n', 'attrs==21.2.0\n', 'autopep8==1.5.6\n', 
        'Babel==2.9.1\n', 'backcall==0.2.0\n', 'bcrypt==3.2.0\n', 
        'binaryornot==0.4.4\n', 'black==19.10b0\n', 'bleach==4.0.0\n', 
        'blinker==1.4\n', 'bokeh==2.3.3\n', 'Bottleneck==1.3.2\n', 
        'branca==0.4.2\n', 'brotlipy==0.7.0\n', 'cached-property==1.5.2\n', 
        'cachetools==4.2.2\n', 'cerebralcortex-kernel==3.3.16\n', 
        'certifi==2021.5.30\n', 'cffi==1.14.6\n', 'chardet==3.0.4\n', 
        'charset-normalizer==2.0.4\n', 'click==8.0.1\n', 
        'cloudpickle==1.1.1\n', 'colorama==0.4.4\n', 'colorlover==0.3.0\n', 
        'cookiecutter==1.7.2\n', 'coverage==5.5\n', 'coveralls==3.2.0\n', 
        'cryptography==3.4.7\n', 'cufflinks==0.17.3\n', 'cycler==0.10.0\n', 
        'Cython==0.29.24\n', 'datascience==0.17.0\n', 'debugpy==1.4.1\n', 
        'decorator==5.0.9\n', 'defusedxml==0.7.1\n', 
        'diff-match-patch==20200713\n', 'docopt==0.6.2\n', 
        'docutils==0.17.1\n', 'entrypoints==0.3\n', 'findspark==1.4.2\n', 
        'flake8==3.9.0\n', 'folium==0.12.1\n', 'fonttools==4.25.0\n', 
        'future==0.18.2\n', 'gast==0.2.2\n', 'gatspy==0.3\n', 
        'geographiclib==1.52\n', 'geopy==2.2.0\n', 'google-auth==1.33.0\n', 
        'google-auth-oauthlib==0.4.4\n', 'google-pasta==0.2.0\n', 
        'graphviz==0.16\n', 'grpcio==1.36.1\n', 'h5py==3.2.1\n', 
        'hdfs3==0.3.1\n', 'idna==3.2\n', 'imagesize==1.2.0\n', 
        'importlib-metadata==3.10.0\n', 'inflection==0.5.1\n', 
        'influxdb==5.3.1\n', 'iniconfig==1.1.1\n', 'intervaltree==3.1.0\n', 
        'ipykernel==6.2.0\n', 'ipython==7.26.0\n', 
        'ipython-genutils==0.2.0\n', 'ipywidgets==7.6.3\n', 'isort==5.9.3\n', 
        'jedi==0.17.2\n', 'Jinja2==2.11.3\n', 'jinja2-time==0.2.0\n', 
        'joblib==1.0.1\n', 'json5==0.9.6\n', 'jsonschema==3.2.0\n', 
        'jupyter-client==6.1.12\n', 'jupyter-core==4.7.1\n', 
        'jupyter-server==1.4.1\n', 'jupyterlab==3.1.7\n', 
        'jupyterlab-pygments==0.1.2\n', 'jupyterlab-server==2.7.1\n', 
        'jupyterlab-widgets==1.0.0\n', 'Keras-Applications==1.0.8\n', 
        'Keras-Preprocessing==1.1.2\n', 'keyring==23.0.1\n', 
        'kiwisolver==1.3.1\n', 'lazy-object-proxy==1.6.0\n', 
        'Markdown==3.3.4\n', 'MarkupSafe==1.1.1\n', 'matplotlib==3.4.2\n', 
        'matplotlib-inline==0.1.2\n', 'mccabe==0.6.1\n', 'mistune==0.8.4\n', 
        'mkl-fft==1.3.0\n', 'mkl-random==1.1.1\n', 'mkl-service==2.3.0\n', 
        'msgpack==1.0.2\n', 'multidict==5.1.0\n', 'munkres==1.1.4\n', 
        'mypy-extensions==0.4.3\n', 'nbclassic==0.2.6\n', 'nbclient==0.5.3\n',
        'nbconvert==6.1.0\n', 'nbformat==5.1.3\n', 'nbsphinx==0.8.7\n', 
        'nest-asyncio==1.5.1\n', 'networkx==2.6.2\n', 'neurokit2==0.1.0\n', 
        'notebook==6.4.3\n', 'numexpr==2.7.3\n', 'numpy==1.19.2\n', 
        'numpydoc==1.1.0\n', 'oauthlib==3.1.1\n', 'olefile==0.46\n', 
        'opt-einsum==3.3.0\n', 'packaging==21.0\n', 'pandas==1.1.5\n', 
        'pandocfilters==1.4.3\n', 'paramiko==2.7.2\n', 'parso==0.7.0\n', 
        'pathspec==0.7.0\n', 'pennprov==2.2.9\n', 'pexpect==4.8.0\n', 
        'pickleshare==0.7.5\n', 'Pillow==8.3.1\n', 'pip==21.2.2\n', 
        'plotly==5.1.0\n', 'pluggy==0.13.1\n', 'poyo==0.5.0\n', 
        'prometheus-client==0.11.0\n', 'prompt-toolkit==3.0.17\n', 
        'protobuf==3.11.2\n', 'psutil==5.8.0\n', 'ptyprocess==0.7.0\n', 
        'py==1.10.0\n', 'py-ecg-detectors==1.0.2\n', 'py4j==0.10.9\n', 
        'pyarrow==4.0.1\n', 'pyasn1==0.4.8\n', 'pyasn1-modules==0.2.8\n', 
        'pycodestyle==2.6.0\n', 'pycparser==2.20\n', 'pydocstyle==6.1.1\n',
        'pydot==1.4.1\n', 'pyerfa==1.7.3\n', 'pyflakes==2.2.0\n', 
        'Pygments==2.10.0\n', 'PyJWT==2.1.0\n', 'pylint==2.9.6\n', 
        'pyls-black==0.4.6\n', 'pyls-spyder==0.3.2\n', 'PyNaCl==1.4.0\n', 
        'pyOpenSSL==20.0.1\n', 'pyparsing==2.4.7\n', 'pyreadline==2.1\n', 
        'pyrsistent==0.17.3\n', 'PySocks==1.7.1\n', 'pyspark==3.1.2\n', 
        'pytest==6.2.4\n', 'python-dateutil==2.8.2\n', 
        'python-jsonrpc-server==0.4.0\n', 'python-language-server==0.36.2\n', 
        'python-slugify==5.0.2\n', 'pytz==2021.1\n', 'PyWavelets==1.1.1\n', 
        'pywin32==228\n', 'pywin32-ctypes==0.2.0\n', 'pywinpty==0.5.7\n', 
        'PyYAML==5.4.1\n', 'pyzmq==22.2.1\n', 'QDarkStyle==3.0.2\n', 
        'qstylizer==0.1.10\n', 'QtAwesome==1.0.2\n', 'qtconsole==5.1.0\n', 
        'QtPy==1.10.0\n', 'regex==2021.8.3\n', 'requests==2.26.0\n', 
        'requests-oauthlib==1.3.0\n', 'rope==0.19.0\n', 'rsa==4.7.2\n', 
        'Rtree==0.9.7\n', 'scikit-learn==0.24.2\n', 'scipy==1.6.2\n', 
        'seaborn==0.11.2\n', 'Send2Trash==1.5.0\n', 
        'setuptools==52.0.0.post20210125\n', 'Shapely==1.7.1\n', 
        'six==1.16.0\n', 'sniffio==1.2.0\n', 'snowballstemmer==2.1.0\n', 
        'sortedcontainers==2.4.0\n', 'Sphinx==4.0.2\n', 
        'sphinxcontrib-applehelp==1.0.2\n', 'sphinxcontrib-devhelp==1.0.2\n', 
        'sphinxcontrib-htmlhelp==2.0.0\n', 'sphinxcontrib-jsmath==1.0.1\n', 
        'sphinxcontrib-qthelp==1.0.3\n', 
        'sphinxcontrib-serializinghtml==1.1.5\n', 'spyder==5.0.5\n', 
        'spyder-kernels==2.0.5\n', 'SQLAlchemy==1.3.24\n', 
        'SQLAlchemy-Utils==0.37.8\n', 'tenacity==8.0.1\n', 
        'tensorboard==2.4.0\n', 'tensorboard-plugin-wit==1.6.0\n', 
        'tensorflow==2.1.0\n', 'tensorflow-addons==0.14.0\n', 
        'tensorflow-estimator==2.1.0\n', 'tensorflow-probability==0.8.0rc0\n',
        'termcolor==1.1.0\n', 'terminado==0.9.4\n', 'testpath==0.5.0\n', 
        'text-unidecode==1.3\n', 'textdistance==4.2.1\n', 
        'threadpoolctl==2.2.0\n', 'three-merge==0.1.1\n', 'tinycss==0.4\n', 
        'toml==0.10.2\n', 'tornado==6.1\n', 'traitlets==5.0.5\n', 
        'typed-ast==1.4.3\n', 'typeguard==2.12.1\n', 
        'typing-extensions==3.10.0.0\n', 'ujson==4.0.2\n', 
        'Unidecode==1.2.0\n', 'urllib3==1.26.6\n', 'watchdog==2.1.3\n', 
        'wcwidth==0.2.5\n', 'webencodings==0.5.1\n', 'Werkzeug==0.16.1\n', 
        'wheel==0.37.0\n', 'whichcraft==0.6.1\n', 
        'widgetsnbextension==3.5.1\n', 'win-inet-pton==1.1.0\n', 
        'wincertstore==0.2\n', 'wrapt==1.12.1\n', 'yapf==0.31.0\n', 
        'yarl==1.6.3\n', 'zipp==3.5.0\n']

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