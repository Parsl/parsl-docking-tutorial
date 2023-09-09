FROM continuumio/anaconda3:latest
RUN apt update -y
RUN apt install wget build-essential libglu1 git vim -y
RUN conda install -c anaconda jupyter -y
RUN conda install nb_conda_kernels -y
ADD environment.yml .
RUN conda env create --file environment.yml
RUN conda update -c defaults jupyter -y
RUN git clone https://github.com/Parsl/parsl-docking-tutorial
