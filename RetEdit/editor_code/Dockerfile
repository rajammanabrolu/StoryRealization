FROM kelvinguu/pytorch:1.2

RUN pip install --upgrade pip
RUN apt-get update

# vim, tmux, and other command line tools
RUN apt-get install -y vim less tmux nmap
COPY .tmux.conf /root

# Jupyter
RUN pip install jupyter
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension  # add Jupyter notebook extension

# vim bindings for Jupyter: https://github.com/lambdalisue/jupyter-vim-binding
RUN mkdir -p $(jupyter --data-dir)/nbextensions
RUN git clone https://github.com/lambdalisue/jupyter-vim-binding $(jupyter --data-dir)/nbextensions/vim_binding
RUN jupyter nbextension enable vim_binding/vim_binding

# autoreload for Jupyter
RUN ipython profile create
RUN echo 'c.InteractiveShellApp.exec_lines = []' >> ~/.ipython/profile_default/ipython_config.py
RUN echo 'c.InteractiveShellApp.exec_lines.append("%load_ext autoreload")' >> ~/.ipython/profile_default/ipython_config.py
RUN echo 'c.InteractiveShellApp.exec_lines.append("%autoreload 2")' >> ~/.ipython/profile_default/ipython_config.py

# NLTK
RUN pip install nltk==3.2.3
# RUN python -m nltk.downloader punkt  # download tokenizer data

# miscellaneous Python packages
RUN pip install requests pyhocon line_profiler pytest tqdm faulthandler python-Levenshtein gitpython futures jsonpickle prettytable tensorboard_logger click pympler

# TensorFlow
RUN pip install tensorflow-gpu==1.3

RUN pip install fabric

RUN git clone https://github.com/kelvinguu/word_forms.git
RUN pip install -e word_forms