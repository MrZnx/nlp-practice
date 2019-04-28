## jupyter notebook in ssh
* install jupyter
```bash
pip install jupyter
```
* create config file
```bash
jupyter notebook --generate-config
```
* create key
```python
from notebook.auth import passwd
passwd()
```
* edit config file
```python
# ~/.jupyter/jupyter_notebook_config.py
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.password = 'sha:ce...'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
```
* start notebook in ssh and coding in localhost
```bash
jupyter notebook
# then input ip:8888 in local web browser
```
## win10 + cuda10 + tf-gpu installation
* [CUDA10](https://developer.nvidia.com/cuda-downloads)
* [cuDNN](https://developer.nvidia.com/rdp/cudnn-download)
```txt
need a nvidia account
unzip, copy file to CUDA path(bin, include, lib)
```
* [tf-gpu](https://github.com/fo40225/tensorflow-windows-wheel/blob/master/1.12.0/py36/GPU/cuda100cudnn73sse2/tensorflow_gpu-1.12.0-cp36-cp36m-win_amd64.whl)
```txt
don't use pip or conda to install the latest TensorFlow.
```