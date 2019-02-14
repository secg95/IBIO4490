#!/bin/bash

curl -X POST https://content.dropboxapi.com/2/files/download --header "Authorization: Bearer -ErvsJ8GXjUAAAAAAAAAcuHkFRwzn4wbXY3C9txHNLvOAFnCFJeTTPVUPzb-tK7B" --header "Dropbox-API-Arg: {\"path\": \"/train-jpg.tar\"}" > train-jpg.tar

tar xf train-jpg.tar

curl -X POST https://content.dropboxapi.com/2/files/download --header "Authorization: Bearer -ErvsJ8GXjUAAAAAAAAAcuHkFRwzn4wbXY3C9txHNLvOAFnCFJeTTPVUPzb-tK7B" --header "Dropbox-API-Arg: {\"path\": \"/train_v2.csv\"}" > ./train-jpg/train_v2.csv


python Lab32.py



rmdir ./train-jpg/ImagenesRecortadas