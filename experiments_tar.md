

Im repo root - kann dann nach upload gelöscht werden: 
tar -czf experiments.tar.gz experiments

datei hochladen zu google drive und url austauschen in docker-compose.yaml


curl -v --progress-bar \
  --header "PRIVATE-TOKEN:glpat-bwbgE3rsgZB_F6IqaRA6P286MQp1OjUxeQk.01.0z0pjt0yr" \
  --upload-file experiments.tar.gz \
  "https://gitup.uni-potsdam.de/api/v4/projects/21813/packages/generic/beyondliteral-experiments/v1/experiments.tar.gz"



teste: 
curl -L -o /tmp/test.tar.gz "https://drive.google.com/uc?export=download&id=1X_scCj_fxi7nLO-2wqFQWrAyGuwpomWE"
ls -lh /tmp/test.tar.gz