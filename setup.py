from setuptools import setup, find_packages

# Function to read dependencies from requirements.txt
def get_requirements(file_path):
    '''
    Reads the list of dependencies from the requirements.txt file.
    '''
    with open(file_path) as f:
        requirements = [line.strip() for line in f.readlines()]
        # Remove lines that look like comment or empty lines
        return [req for req in requirements if req and not req.startswith('#')]

setup(
    name='loan_risk_prediction',
    version='0.1.0',
    description='A machine learning project for predicting loan default risk.',
    author='Your Name', # Replace with your name/team name
    packages=find_packages(exclude=['tests', 'docs']),
    install_requires=get_requirements('requirements.txt'),
    
    # Entry points define how executable scripts are run after installation.
    # We map 'loan_train' and 'loan_predict' to the main functions in our scripts.
    entry_points={
        'console_scripts': [
            'loan_train = src.train:main',
            'loan_predict = src.predict:main',
        ],
    },
    
    # Metadata for PyPI / distribution
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
)
