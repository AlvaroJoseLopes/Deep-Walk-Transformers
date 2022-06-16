import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='deep_walk_transformers',
    version='0.0.0',
    author='',
    author_email='',
    description='Deep Walk Transformers Model',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/AlvaroJoseLopes/Deep-Walk-Transformers',
    license='MIT',
    packages=['deep_walk_transformers', 'deep_walk_transformers.utils'],
    install_requires=['graph-walker==1.0.6']
)