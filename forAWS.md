# basic setups 

Most of the libraries requires python 3.10 

Used pyenv to install python3.10 without sudo access

```
curl -fsSL https:/p| bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
```

Create the virtual envrionment and install required libraires,following packages got problem installing from requirment.txt 

```
pip install uv 
uv pip install vllm 
pip install faiss-gpu 
```

# Run Elmtext 
run_elmtext.py 
this will start the vllm and icl services, run the script files, and stop the process afters its complete. 

## 8bit fine Tune support     
8 bit fine tune is only supported with : 
python 3.10.0 
torch==2.7.0 
trl='0.12.0'
peft==0.15.2
transformers=='4.49.0'
datasets=='3.6.0'
bitsandbytes==0.46.1

install python version without sudo using  
```
pyenv install 3.10.10 
```

create virtual environment 
```
pyenv virtualenv 3.10.10 venvname
```

activate virtual environment 
```
pyenv activate venvname
``` 

install all required with same version 

