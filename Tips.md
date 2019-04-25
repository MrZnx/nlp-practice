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