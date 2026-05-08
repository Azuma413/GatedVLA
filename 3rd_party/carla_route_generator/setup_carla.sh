#!/usr/bin/env bash
set -euo pipefail

# Download and install CARLA
mkdir -p carla
cd carla
wget --content-disposition https://tiny.carla.org/carla-0-9-15-linux
tar -xf CARLA_0.9.15.tar.gz

# GNU tar 1.35 has a known bug with --keep-newer-files that makes CARLA's
# ImportAssets.sh fail on existing directories in AdditionalMaps_0.9.15.tar.gz.
sed -i 's/tar --keep-newer-files -xvf/tar -xvf/g' ImportAssets.sh

cd Import
wget --content-disposition https://tiny.carla.org/additional-maps-0-9-15-linux
cd ..
./ImportAssets.sh
rm CARLA_0.9.15.tar.gz
cd ..
