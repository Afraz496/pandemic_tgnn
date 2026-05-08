FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --upgrade pip && \
    pip install \
        networkx==3.1 \
        numpy==1.23.5 \
        pandas==1.3.5 \
        scikit_learn==1.0.2 \
        scipy==1.11.4 \
        statsmodels==0.14.0 \
        Unidecode==1.3.7 \
        google-cloud-storage

# PyTorch + PyG
RUN pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install torch_geometric==2.4.0
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

CMD ["python", "train.py"]