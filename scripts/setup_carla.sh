#!/usr/bin/env bash
set -euo pipefail

mkdir -p 3rd_party/CARLA_0915

cd 3rd_party/CARLA_0915

wget -O CARLA_0915.tar.gz https://tiny.carla.org/carla-0-9-15-linux
tar -xvzf CARLA_0915.tar.gz

# GNU tar 1.35 has a known bug with --keep-newer-files that makes CARLA's
# ImportAssets.sh fail on existing directories in AdditionalMaps_0.9.15.tar.gz.
sed -i 's/tar --keep-newer-files -xvf/tar -xvf/g' ImportAssets.sh

cd Import
wget -O AdditionalMaps_0.9.15.tar.gz https://tiny.carla.org/additional-maps-0-9-15-linux
cd ..
bash ImportAssets.sh
