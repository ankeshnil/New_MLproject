from setuptools import find_packages, setup
from typing import List

hypen_e_dot = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This function we return the list of all the requirements
    '''
    requirments=[]
    with open (file_path) as file_obj:
        requirments = file_obj.readlines()
        requirments =  [req.strip() for req in requirments] 
        ## with out replacement pyhon read it like "numpy\n", "pandas\n" so beacuse of \n it can"t download this package to for this we should remove it
        
        if hypen_e_dot in requirments:
            requirments.remove(hypen_e_dot) #we do not need this -e . in my setup.py file
            
    return requirments

setup(
    name = 'new_mlProject',
    version= '0.0.1',
    author = 'Ankesh',
    author_email = 'ankeshnil00@gmai.com',
    packages = find_packages(),  # if find- src has __init__.py , So src is a package , and src contail all our project code file
    install_requires = get_requirements('requirements.txt')
)