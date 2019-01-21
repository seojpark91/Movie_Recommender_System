# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name="movie_rsys_pkg",
    packages=find_packages(),
    include_package_data=True,
    version="0.0.1",
    author="Seo Jung Park",
    author_email="seojpark91@gmail.com",
    description="A movie recommender system creation practice package",
    url = "https://github.com/seojpark91/Movie_Recommender_System"
    zip_safe=False
)