cd Path/to/data

gdown --folder https://drive.google.com/drive/folders/1aqKCzVKh3YBEISO134le1FzrfCwGutbk
for f in *.tar.xz; do
  tar -xf "$f"
done
