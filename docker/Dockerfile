FROM continuumio/anaconda3:latest
RUN apt update -y && apt-get install libgl1 -y
RUN apt install wget build-essential libglu1 git vim -y
RUN conda install -c anaconda jupyter -y
RUN conda install nb_conda_kernels -y
COPY . /parsldock
WORKDIR /parsldock
RUN conda env create --file docker/environment.yml
RUN conda update -c defaults jupyter -y

ENV MGLTOOLS_HOME=/opt/conda/envs/parsldockenv

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "parsldockenv", "parsldock"]
