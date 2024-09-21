FROM python:3.9-bookworm

RUN apt-get update \
  && apt-get install -y libgl1-mesa-glx \
  ffmpeg \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN pip install \
  xgboost \
  dtaidistance \
  scikit-learn \
  seaborn

WORKDIR /app
